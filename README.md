# Rajesh AI Resume Assistant

A voice-first, call-screen-style AI assistant grounded strictly in Rajesh Kuna's resume.
Visitors can **speak** or **type** with "Raj" — the AI agent — to learn about Rajesh's
experience, skills, and projects.

---

## Stack

| Layer | Dev / Production |
|---|---|
| Real-time transport | LiveKit Cloud (`wss://...livekit.cloud`) |
| Frontend | React + Vite + TypeScript + `livekit-client` |
| STT | Deepgram Nova-2 |
| LLM | Groq API with auto-fallback (`openai/gpt-oss-120b` → `openai/gpt-oss-20b` → `qwen/qwen3.8-27b`) |
| TTS | Cartesia Sonic-3 |
| Token server | FastAPI (Python, same venv as agent) |

---

## Prerequisites

- Node.js ≥ 18
- Python 3.11+ with venv at `agent/venv/`
- LiveKit Cloud credentials (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` in `.env`)
- Groq API Key (`GROQ_API_KEY` in `.env`)

---

## One-time setup

```powershell
# 1. Copy env file (already done — keys are present)
copy .env.example .env

# 2. Install Python packages
.\agent\venv\Scripts\pip.exe install -r .\agent\requirements.txt

# 3. Install frontend packages (already done)
cd frontend && npm install
cd ..
```

---

## Running (3 terminals — Docker not required when using LiveKit Cloud)

### Terminal 1 — AI voice agent
```powershell
.\agent\venv\Scripts\python.exe .\agent\agent.py dev
```

### Terminal 2 — Token server
```powershell
.\agent\venv\Scripts\uvicorn.exe agent.token_server:app --port 8880 --reload
```

### Terminal 3 — Frontend
```powershell
cd frontend
npm run dev
```

Then open **http://localhost:5173** — allow microphone access and the call starts automatically.

---

## Architecture

```
Browser (React)
  │   fetch /token?greeting=Good+afternoon
  ▼
Token Server (FastAPI :8880)
  │   mints LiveKit JWT, embeds greeting in room metadata
  ▼
LiveKit Cloud
  │   WebSocket signaling + WebRTC audio
  ▼
Agent Worker (Python)
  ├── Deepgram STT  ─→  transcript
  ├── Groq API LLM  ─→  grounded response (resume.md in system prompt)
  └── Cartesia TTS  ─→  audio back to visitor
```

Text fallback: browser sends `{ type: "text_input" }` over LiveKit DataChannel →
agent routes through same LLM session → replies with `{ type: "text_response" }`.

---

## Changing the LLM Model

In `agent/agent.py`, the LLM is powered by Groq API (`llama-3.1-8b-instant`). You can set `GROQ_MODEL` in `.env` to switch to `llama-3.3-70b-versatile` or any other Groq model without changing code.

---

## Updating Rajesh's resume

Edit `shared/resume.md` only — no code changes required anywhere else.

---

## Project structure

```
/rajesh-ai-assistant
  /agent
    agent.py          Voice agent (STT → LLM → TTS + DataChannel text)
    token_server.py   FastAPI JWT minting endpoint
    requirements.txt
    venv/
  /frontend
    src/
      App.tsx
      hooks/useLiveKitRoom.ts
      components/
        CallScreen.tsx
        CaptionsStrip.tsx
        TextInput.tsx
    .env.local        VITE_TOKEN_SERVER_URL=http://localhost:8880
  /shared
    resume.md         Single source of truth for agent knowledge
  docker-compose.yml
  .env               API keys (never commit)
  .env.example
```
