// server.js - OpenAI to NVIDIA NIM API Proxy (JanitorAI-compatible, pass-through)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware - allow large JanitorAI histories
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// NVIDIA NIM API configuration
// End-to-end NIM flow (https://docs.nvidia.com/nim + https://docs.api.nvidia.com/nim):
//   POST {NIM_API_BASE}/chat/completions with OpenAI-compatible body
//   { model, messages: [{role, content}], temperature, top_p, max_tokens,
//     stream, stream_options, seed, stop, tools, ... }
//   Auth: Authorization: Bearer $NVIDIA_API_KEY
//   Streaming: SSE `data: {...}` chunks + `data: [DONE]` terminator.
//   Reasoning models return `reasoning_content` (also seen as `reasoning` /
//   `thinking`) alongside `content`, in both message and delta objects.
const NIM_API_BASE = (process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1').replace(/\/$/, '');

// SHOW_REASONING controls OUTPUT only (whether reasoning is merged into
// content as <think> blocks). Thinking is still SENT upstream regardless.
const SHOW_REASONING = String(process.env.SHOW_REASONING || 'false').toLowerCase() === 'true';
// THINKING controls whether thinking/reasoning parameters are SENT to NIM.
// Defaults to true per GLM/Kimi/Qwen docs. Set THINKING=false to disable.
const THINKING_ENABLED = String(
  process.env.THINKING ?? process.env.ENABLE_THINKING_MODE ?? 'true'
).toLowerCase() !== 'false';
// Optional override for GLM-style `reasoning_effort` (low|high|max).
// Empty = auto (max for glm family, otherwise omit).
const REASONING_EFFORT_DEFAULT = String(process.env.REASONING_EFFORT || '').toLowerCase() || null;
// Optional alias map for friendly names, e.g.
// MODEL_MAP='{"z-ai/glm-5-3-flash":"z-ai/glm-5.3"}'
// No hardcoded defaults: unknown models pass through verbatim.
let MODEL_MAP_CACHE = null;
function getModelMap() {
  if (MODEL_MAP_CACHE) return MODEL_MAP_CACHE;
  try {
    MODEL_MAP_CACHE = JSON.parse(process.env.MODEL_MAP || '{}');
  } catch {
    console.warn('MODEL_MAP is not valid JSON, ignoring');
    MODEL_MAP_CACHE = {};
  }
  return MODEL_MAP_CACHE;
}
function resolveModel(requested) {
  const map = getModelMap();
  return map[requested] || requested;
}

// Prefer the caller's own key (JanitorAI sends Authorization: Bearer nvapi-...),
// fall back to the server env key so single-user setups keep working.
function getApiKey(req) {
  const h = req.headers.authorization || '';
  const m = h.match(/^Bearer\s+(.+)$/i);
  if (m && m[1] && m[1] !== 'undefined' && m[1] !== 'null') return m[1].trim();
  return process.env.NIM_API_KEY;
}

// Build the thinking payload for a model, respecting anything the client
// already sent. Covers the documented variants:
//   GLM family (incl. z-ai/glm-5.3): extra_body.chat_template_kwargs
//     { enable_thinking: true, clear_thinking: true } + reasoning_effort max
//     (GLM-5.3 card: reasoning_effort low|high|max defaults max;
//      clear_thinking defaults false but chat scenarios should pass true)
//   Kimi / Qwen thinking models: chat_template_kwargs { thinking: true,
//     enable_thinking: true }
//   Generic fallback: extra_body.thinking { type: "enabled" }
// All extensions ride inside extra_body so strict schemas with
// additionalProperties:false (e.g. glm-5.3) don't 422 on unknown top-level keys.
function buildThinking(nimModel, body) {
  if (!THINKING_ENABLED) return {};
  const lower = String(nimModel || '').toLowerCase();
  const clientExtra = (body.extra_body && typeof body.extra_body === 'object') ? body.extra_body : {};
  const clientKwargsTop = (body.chat_template_kwargs && typeof body.chat_template_kwargs === 'object')
    ? body.chat_template_kwargs : {};
  const clientKwargsExtra = (clientExtra.chat_template_kwargs && typeof clientExtra.chat_template_kwargs === 'object')
    ? clientExtra.chat_template_kwargs : {};
  const merged = { ...clientKwargsTop, ...clientKwargsExtra };
  let reasoningEffort = body.reasoning_effort ?? clientExtra.reasoning_effort ?? null;
  let extraThinking = clientExtra.thinking;

  const isGLM = lower.includes('glm');
  const isKimi = lower.includes('kimi');
  const isQwen = lower.includes('qwen');

  if (isGLM) {
    if (merged.enable_thinking === undefined) merged.enable_thinking = true;
    // Chat-safe default per GLM-5.3 docs (clears prior thinking blocks).
    // Agentic callers can override with clear_thinking:false.
    if (merged.clear_thinking === undefined) merged.clear_thinking = true;
    if (merged.thinking === undefined) merged.thinking = true; // kimi-style compat, ignored if unknown
    if (!reasoningEffort) reasoningEffort = REASONING_EFFORT_DEFAULT || 'max';
  } else if (isKimi || isQwen) {
    if (merged.thinking === undefined) merged.thinking = true;
    if (merged.enable_thinking === undefined) merged.enable_thinking = true;
    if (!reasoningEffort && REASONING_EFFORT_DEFAULT) reasoningEffort = REASONING_EFFORT_DEFAULT;
  } else {
    if (merged.enable_thinking === undefined) merged.enable_thinking = true;
    if (extraThinking === undefined) extraThinking = { type: 'enabled' };
    if (!reasoningEffort && REASONING_EFFORT_DEFAULT) reasoningEffort = REASONING_EFFORT_DEFAULT;
  }

  const extraBody = { ...clientExtra, chat_template_kwargs: merged };
  if (extraThinking !== undefined) extraBody.thinking = extraThinking;
  const out = { extra_body: extraBody };
  if (reasoningEffort) {
    out.reasoning_effort = reasoningEffort;
    extraBody.reasoning_effort = reasoningEffort;
  }
  return out;
}

function extractReasoning(obj) {
  if (!obj || typeof obj !== 'object') return null;
  for (const k of ['reasoning_content', 'reasoning', 'thinking']) {
    if (typeof obj[k] === 'string' && obj[k]) return { key: k, text: obj[k] };
  }
  return null;
}
function stripReasoningFields(obj) {
  if (!obj || typeof obj !== 'object') return;
  delete obj.reasoning_content;
  delete obj.reasoning;
  delete obj.thinking;
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    upstream: NIM_API_BASE,
    reasoning_display: SHOW_REASONING,
    thinking: THINKING_ENABLED,
    mode: 'pass-through (any model)',
  });
});

app.get('/', (req, res) => {
  res.json({
    service: 'OpenAI to NVIDIA NIM Proxy',
    usage: 'Point JanitorAI/OpenAI clients at /v1. Any `model` string is forwarded verbatim to NIM.',
    thinking: THINKING_ENABLED,
    reasoning_display: SHOW_REASONING,
    env: ['NIM_API_KEY', 'NIM_API_BASE', 'THINKING', 'SHOW_REASONING', 'REASONING_EFFORT', 'MODEL_MAP'],
  });
});

// List models: proxy the real NIM catalog so pickers show actual IDs.
// Falls back to an empty OpenAI list (clients like JanitorAI accept a
// manually typed custom model even when the list is empty).
let modelsCache = { at: 0, data: null };
app.get('/v1/models', async (req, res) => {
  try {
    if (Date.now() - modelsCache.at < 5 * 60 * 1000 && modelsCache.data) return res.json(modelsCache.data);
    const apiKey = getApiKey(req);
    if (!apiKey) throw new Error('no-key');
    const r = await axios.get(`${NIM_API_BASE}/models`, {
      headers: { Authorization: `Bearer ${apiKey}` },
      timeout: 15000,
      validateStatus: () => true,
    });
    if (r.status >= 200 && r.status < 300 && r.data) {
      const data = r.data.data ? r.data : { object: 'list', data: r.data };
      modelsCache = { at: Date.now(), data };
      return res.json(data);
    }
    throw new Error(`upstream ${r.status}`);
  } catch {
    const single = req.query.model ? String(req.query.model) : null;
    res.json({
      object: 'list',
      data: single
        ? [{ id: single, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy' }]
        : [],
    });
  }
});

async function handleChat(req, res) {
  const apiKey = getApiKey(req);
  if (!apiKey) {
    return res.status(401).json({
      error: { message: 'Missing API key: set Authorization: Bearer nvapi-... or NIM_API_KEY env', type: 'invalid_request_error', code: 401 },
    });
  }
  const body = req.body || {};
  const requestedModel = body.model;
  if (!requestedModel || typeof requestedModel !== 'string') {
    return res.status(400).json({
      error: { message: '`model` is required (any NIM model id, e.g. z-ai/glm-5.3)', type: 'invalid_request_error', code: 400 },
    });
  }
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return res.status(400).json({
      error: { message: '`messages` must be a non-empty array', type: 'invalid_request_error', code: 400 },
    });
  }
  const nimModel = resolveModel(requestedModel);
  const stream = body.stream ?? false;

  const {
    model: _m,
    messages,
    stream: _s,
    extra_body: clientExtraBody,
    chat_template_kwargs: clientChatKwargs, // preserved below via thinking merge; top-level copy kept if client sent it
    ...passthrough
  } = body;

  // Forward every other OpenAI param verbatim (temperature, top_p,
  // max_tokens, stop, seed, tools, ...). No falsy-value bugs: 0/false
  // are preserved, unset fields are omitted so NIM applies its defaults.
  const nimRequest = { model: nimModel, messages, ...passthrough, stream: !!stream };
  if (stream && !nimRequest.stream_options) nimRequest.stream_options = { include_usage: true };

  // Thinking: merge client intent with family-specific defaults.
  const thinking = buildThinking(nimModel, body);
  if (thinking.extra_body) {
    nimRequest.extra_body = { ...(clientExtraBody || {}), ...thinking.extra_body };
  }
  if (thinking.reasoning_effort && nimRequest.reasoning_effort === undefined) {
    nimRequest.reasoning_effort = thinking.reasoning_effort;
  }
  if (clientChatKwargs !== undefined) nimRequest.chat_template_kwargs = clientChatKwargs;

  console.log(`-> ${NIM_API_BASE}/chat/completions model=${nimModel} stream=${!!stream} msgs=${messages.length} thinking=${THINKING_ENABLED}`);

  let upstream;
  try {
    upstream = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        Accept: stream ? 'text/event-stream' : 'application/json',
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 300000,
      validateStatus: () => true,
    });
  } catch (e) {
    console.error('Upstream transport error:', e.message);
    return res.status(502).json({
      error: { message: `Upstream unreachable: ${e.message}`, type: 'api_error', code: 502 },
    });
  }

  if (upstream.status < 200 || upstream.status >= 300) {
    // Error passthrough (JSON or streamed error body).
    if (stream && upstream.data && typeof upstream.data.on === 'function') {
      let errBuf = '';
      upstream.data.on('data', (c) => { errBuf += c.toString(); });
      upstream.data.on('end', () => {
        let parsed;
        try {
          const m = errBuf.match(/data:\s*(\{.*\})/s);
          parsed = JSON.parse(m ? m[1] : errBuf);
        } catch { /* keep raw */ }
        res.status(upstream.status).json(parsed && parsed.error ? parsed : {
          error: { message: (parsed && parsed.detail) || errBuf.slice(0, 2000) || `Upstream ${upstream.status}`, type: 'invalid_request_error', code: upstream.status },
        });
      });
      upstream.data.on('error', () => res.status(502).json({ error: { message: 'Upstream stream error', type: 'api_error', code: 502 } }));
      return;
    }
    const detail = upstream.data && (upstream.data.detail || upstream.data.title);
    return res.status(upstream.status).json(upstream.data && upstream.data.error ? upstream.data : {
      error: { message: detail || upstream.data?.message || `Upstream ${upstream.status}`, type: 'invalid_request_error', code: upstream.status },
    });
  }

  if (stream) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');

    let buffer = '';
    const thinkOpen = {}; // per-choice index <think> state (SHOW_REASONING only)
    let doneSent = false;
    const sendDone = () => { if (!doneSent) { doneSent = true; res.write('data: [DONE]\n\n'); } };

    upstream.data.on('data', (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const raw of lines) {
        const line = raw.trimEnd();
        if (!line) continue;
        if (line.startsWith(':')) { res.write(line + '\n'); continue; } // SSE keepalive/comment
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') { sendDone(); continue; }
        let data;
        try { data = JSON.parse(payload); } catch { res.write(line + '\n'); continue; }
        try {
          if (data.model) data.model = requestedModel;
          const choices = Array.isArray(data.choices) ? data.choices : [];
          let skipChunk = true;
          for (const choice of choices) {
            const delta = choice.delta;
            if (!delta) { skipChunk = false; continue; }
            const r = extractReasoning(delta);
            const content = delta.content;
            const hasTools = delta.tool_calls !== undefined && delta.tool_calls !== null;
            const hasRole = !!delta.role;
            if (SHOW_REASONING && r) {
              const idx = choice.index ?? 0;
              let combined = '';
              if (!thinkOpen[idx]) { combined = '<think>\n' + r.text; thinkOpen[idx] = true; }
              else combined = r.text;
              if (typeof content === 'string' && content) { combined += '</think>\n\n' + content; thinkOpen[idx] = false; }
              delta.content = combined || delta.content;
              stripReasoningFields(delta);
              if (delta.content) skipChunk = false; else if (hasTools || hasRole) skipChunk = false;
            } else {
              const hadReasoning = !!r;
              stripReasoningFields(delta);
              if (typeof content === 'string' && content) skipChunk = false;
              else if (hasTools) skipChunk = false;
              else if (hasRole) skipChunk = false;
              else if (!hadReasoning && (content === null || content === undefined)) skipChunk = false; // e.g. usage-only chunk
              // else: pure reasoning chunk while stripping -> drop (model still thinking)
            }
          }
          if (data.usage) skipChunk = false;
          if (!skipChunk) res.write(`data: ${JSON.stringify(data)}\n\n`);
        } catch {
          res.write(line + '\n');
        }
      }
    });
    upstream.data.on('end', () => { sendDone(); res.end(); });
    upstream.data.on('error', (err) => { console.error('Stream error:', err.message); try { sendDone(); res.end(); } catch {} });
    req.on('close', () => { try { upstream.data.destroy(); } catch {} });
    return;
  }

  // Non-streaming: preserve upstream shape (incl. tool_calls), only adjust reasoning.
  const data = upstream.data || {};
  const choices = Array.isArray(data.choices) ? data.choices.map((choice) => {
    const msg = choice.message || {};
    const r = extractReasoning(msg);
    let content = msg.content ?? '';
    if (SHOW_REASONING && r) content = '<think>\n' + r.text + '\n</think>\n\n' + content;
    stripReasoningFields(msg);
    return { ...choice, message: { ...msg, content } };
  }) : [];
  return res.json({ ...data, model: requestedModel, choices });
}

// Chat completions endpoint (main proxy) + no-prefix alias for SDKs
app.post('/v1/chat/completions', handleChat);
app.post('/chat/completions', handleChat);

// Catch-all for unsupported endpoints (Express 4+5 compatible)
app.use((req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 },
  });
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
    console.log(`Upstream: ${NIM_API_BASE}`);
    console.log(`Mode: pass-through (any model string forwarded to NIM)`);
    console.log(`Thinking sent to NIM: ${THINKING_ENABLED ? 'ENABLED' : 'DISABLED'}`);
    console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED (<think>)' : 'DISABLED (stripped)'}`);
  });
}

module.exports = app;
