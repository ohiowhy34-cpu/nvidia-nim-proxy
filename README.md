# openai-nim-proxy
OpenAI compatible proxy for NVIDIA NIM API. Any `model` string is forwarded verbatim to `https://integrate.api.nvidia.com/v1` (no hardcoded model list), so JanitorAI / OpenAI clients can use any NIM model id such as `z-ai/glm-5.3`.

## Run
```
npm install
NIM_API_KEY=nvapi-... npm start
```

## JanitorAI setup
- Base URL: `http://localhost:3000/v1`
- API key: your `nvapi-...` key (forwarded per-request; falls back to `NIM_API_KEY` env)
- Model: any NIM id, e.g. `z-ai/glm-5.3` (typed manually; `/v1/models` proxies the live NIM catalog)

## Env
| Var | Default | Purpose |
|-----|---------|---------|
| `NIM_API_KEY` | — | Fallback key when client sends no `Authorization` header |
| `NIM_API_BASE` | `https://integrate.api.nvidia.com/v1` | NIM endpoint |
| `THINKING` | `true` | Send thinking params to NIM (`false` disables). Legacy `ENABLE_THINKING_MODE` also read |
| `SHOW_REASONING` | `true` | `true` merges `reasoning_content` into output as `<think>` blocks; `false` strips it |
| `STREAM_DEFAULT` | `true` | Omitted `stream` field defaults to SSE streaming; explicit `false` still returns JSON |
| `REASONING_EFFORT` | auto (`max` for `glm`) | Override `reasoning_effort` (`low`\|`high`\|`max`) |
| `MODEL_MAP` | `{}` | Optional alias JSON, e.g. `{"z-ai/glm-5-3-flash":"z-ai/glm-5.3"}` |

## Thinking (researched from NVIDIA docs)
- GLM family: `extra_body.chat_template_kwargs={enable_thinking:true, clear_thinking:true}` + `reasoning_effort=max` (chat-safe; override `clear_thinking:false` for agentic preserved-thinking)
- Kimi/Qwen thinking models: `chat_template_kwargs={thinking:true, enable_thinking:true}`
- Others: `extra_body.thinking={type:"enabled"}` + `enable_thinking:true`
- Client-sent `chat_template_kwargs` / `reasoning_effort` / `thinking` are respected and merged, never blindly overwritten
- Extensions ride in `extra_body` to avoid 422s on strict schemas (`additionalProperties:false`)
