"""
Rajesh AI Resume Assistant — Voice Agent
Phases 0 + 2 + 3 implemented here.

Run:  python agent.py dev
Env:  ../.env  (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET,
                DEEPGRAM_API_KEY, CARTESIA_API_KEY)
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import asyncio
import json
import logging
import ssl                      # pre-import: avoids 578ms event-loop block
import re
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# ---------------------------------------------------------------------------
# Pre-warm heavy imports that would otherwise block the async event loop
# on first use (numpy FFT, ssl certs, etc.)
# ---------------------------------------------------------------------------
try:
    import numpy as np           # pre-import: avoids 277ms block on first FFT
    np.fft.rfft(np.zeros(512))   # trigger lazy sub-module load
except ImportError:
    pass

try:
    ssl.create_default_context()  # pre-load system trust store
except Exception:
    pass

_agent_dir = Path(__file__).parent
_root_dir = _agent_dir.parent
if str(_agent_dir) not in sys.path:
    sys.path.insert(0, str(_agent_dir))
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

logger = logging.getLogger("rajesh-agent")

from dotenv import load_dotenv

# Load .env from the project root (one level above /agent)
load_dotenv(Path(__file__).parent.parent / ".env")

import certifi
import httpx
import openai as openai_api

from livekit import rtc
from livekit.agents import (
    APIConnectOptions,
    Agent,
    AgentSession,
    JobContext,
    JobExecutorType,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import FallbackAdapter
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.room_io import AudioInputOptions, RoomOptions, TextOutputOptions
from livekit.agents.voice.turn import (
    EndpointingOptions,
    InterruptionOptions,
    PreemptiveGenerationOptions,
    TurnHandlingOptions,
)
from livekit.plugins import deepgram, openai

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Ordered list of Groq-hosted models to try (primary → fallback → last-resort).
# The FallbackAdapter will try each in order if the previous one fails.
# openai/gpt-oss-120b is the proven accurate model.
# qwen/qwen3.8-27b provides ultra-fast (~0.23s) fallback with no reasoning token overhead.
GROQ_MODELS = [
    os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]

# Shared HTTP client with pre-cached SSL context (prevents 1.7s event-loop stalls during active calls)
_ssl_context = ssl.create_default_context(cafile=certifi.where())

_shared_http_client = httpx.AsyncClient(
    verify=_ssl_context,
    timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=10.0),
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
)

_shared_openai_client = openai_api.AsyncClient(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    http_client=_shared_http_client,
)

GROQ_LLMS = [
    openai.LLM(
        model=model_name,
        client=_shared_openai_client,
    )
    for model_name in dict.fromkeys(GROQ_MODELS)
]

FALLBACK_LLM = FallbackAdapter(
    GROQ_LLMS,
    attempt_timeout=10.0,
    max_retry_per_llm=1,
    retry_interval=0.5,
)

# ---------------------------------------------------------------------------
# Resume grounding — load the single source of truth at startup
# ---------------------------------------------------------------------------
_RESUME_PATH = Path(__file__).parent.parent / "shared" / "resume.md"

def _load_resume() -> str:
    try:
        raw = _RESUME_PATH.read_text(encoding="utf-8")
        # Strip developer/agent HTML comments so the agent doesn't take meta-instructions overly literally
        return re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL).strip()
    except FileNotFoundError:
        raise RuntimeError(f"resume.md not found at {_RESUME_PATH}. "
                           "Ensure shared/resume.md exists.")

RESUME_CONTENT = _load_resume()

# The system instructions embed the resume with clear conversational voice guidelines
SYSTEM_INSTRUCTIONS = f"""
You are Raj — the intelligent, friendly, and articulate AI voice assistant representing Rajesh Kuna.
You speak naturally, warmly, and concisely, like an engaging tech professional having an interactive conversation.

Rajesh's actual verified background is provided below:
---RESUME START---
{RESUME_CONTENT}
---RESUME END---

═══════════════════════════════════════════════════════
NAME COLLECTION (MANDATORY FIRST STEP):
═══════════════════════════════════════════════════════
- After your greeting, your VERY FIRST action must be to ask for the visitor's name.
- If they share their name, remember it and use it naturally in responses (e.g., "Great question, Amit!").
- If they refuse or skip, gently insist ONCE more (e.g., "No worries! Just a first name so I can make our chat more personal. What should I call you?").
- If they refuse again, accept it gracefully and proceed, addressing them as "friend" or "there".
- Do NOT jump into resume content until you have asked for the name at least once.

═══════════════════════════════════════════════════════
STRICT GUARDRAILS — RESUME CONTENT ONLY:
═══════════════════════════════════════════════════════
- You MUST answer questions ONLY from Rajesh's resume content above.
- GREETINGS, NAME COLLECTION, AND FAREWELLS are natural conversational parts of the interaction — handle them politely and directly without forcing resume content into them.
- If a user asks about ANYTHING outside the resume (e.g., weather, politics, Elon Musk, current events, sports, other people, general knowledge, coding tutorials, opinions on non-resume topics), you MUST:
  a) Politely decline in ONE short sentence.
  b) Steer the conversation BACK to Rajesh's background.
  Example: "I appreciate the curiosity, but I'm here to help you explore Rajesh's professional journey! Want to know about his Spring Boot expertise or his TCS projects?"
- EXCEPTION: You may share a quick, witty tech joke if asked (keep it to 1-2 sentences max), then immediately redirect back to the resume.
- NEVER discuss, debate, or provide information about topics unrelated to Rajesh's resume, even if the user insists.
- If a user repeatedly tries to divert you, stay firm but polite: "I'd love to chat about that, but my specialty is all things Rajesh Kuna! What can I tell you about his work?"

CONVERSATIONAL RULES & VOICE PERSONALITY:
1. Spoken Brevity (CRITICAL):
   - Keep answers to 1 to 3 short, punchy sentences (under 35 words).
   - This is a real-time voice call, NOT a written essay. Never overwhelm the user with walls of text.
2. Natural Skill Highlights (NO KEYWORD DUMPING):
   - When asked about skills or tech stack, NEVER recite long comma-separated lists of 20+ technologies.
   - Summarize into 3-4 key pillars (e.g., "Rajesh is a Java Backend Specialist with 3 years at TCS, specializing in Spring Boot microservices, Kafka event streaming, and GenAI using Spring AI.").
3. Interactive & Personable:
   - Use the visitor's name naturally in responses once you know it.
   - If greeted (e.g. "hi", "what is your name?", "are you there?"), respond warmly and naturally.
   - If the user's input is unclear, garbled, or gibberish (e.g., "Gkjcf", background noise), politely ask for clarification in one short sentence without repeating previous answers.
   - NEVER repeat, resume, or re-explain what you were saying before if the user changed the topic or interrupted you. Address the user's latest utterance directly.
4. Factual Grounding:
   - For all factual questions about Rajesh's work history, companies (TCS), projects (OmniSaina), education, and metrics (70% API latency cut, 30% query throughput optimization), rely strictly on the facts above.
   - Never invent arbitrary companies, projects, or employment history.
   - Share contact details (phone: +91 9553941055, email: rajeshkuna70@gmail.com, LinkedIn, GitHub) ONLY when the visitor specifically asks how to reach or contact Rajesh, or during the farewell.
5. End with a Brief Hook:
   - Where appropriate during normal inquiries, keep the conversation moving with a short conversational prompt (e.g., "Would you like to hear about his TCS onsite experience or his OmniSaina project?").

═══════════════════════════════════════════════════════
FAREWELL & CONTACT COLLECTION (TOP PRIORITY):
═══════════════════════════════════════════════════════
- When the user says "bye", "goodbye", "exit", "that's all", "thanks I'm done", "see you", or indicates they want to end/leave:
  * CRITICAL: NEVER repeat, resume, summarize, or finish what you were saying previously. If the user interrupted you, DROP that previous thought completely!
  * CRITICAL: DO NOT dump or mention ANY resume skills, metrics, projects, or background in your farewell.
  * Your ENTIRE reply must ONLY be the polite farewell and contact collection (under 40 words):
    a) Thank them warmly for their time and interest in Rajesh's profile.
    b) Ask: "Before you go, would you like to share your email or phone number? I'll make sure Rajesh reaches out to you personally!"
    c) Always share Rajesh's contact info: "You can reach Rajesh directly at rajeshkuna70@gmail.com or call +91 9553941055. Have a great day!"
    d) If they shared their name, use it warmly in the farewell (e.g., "Take care, Ramu!").
- If they subsequently share contact details, confirm warmly: "Thanks! Rajesh will get back to you soon."
- If they decline to share contact details, respect it: "No worries at all! Whenever you'd like to connect with Rajesh, just come back and share your details. You can reach Rajesh directly at rajeshkuna70@gmail.com. Have a wonderful day!"
""".strip()

# ---------------------------------------------------------------------------
# Deterministic opener — greets, then asks for name
# ---------------------------------------------------------------------------
def _build_opener(greeting: str, visitor_name: str | None = None) -> str:
    if visitor_name:
        return (
            f"{greeting}! I'm Raj — Rajesh Kuna's AI voice assistant. "
            f"Nice to meet you, {visitor_name}! "
            "I can tell you about his experience, skills, projects, and background. "
            "What would you like to know?"
        )
    return (
        f"{greeting}! I'm Raj — Rajesh Kuna's AI voice assistant. "
        "Before we dive in, may I know your name?"
    )

_PHASE0_TEST_LINE = (
    "Hello! I'm Raj, Rajesh's AI voice assistant. "
    "The pipeline is connected and working correctly. Please go ahead and speak!"
)

# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------
class ResumeAgent(Agent):
    def __init__(self, opener: str, room) -> None:
        super().__init__(instructions=SYSTEM_INSTRUCTIONS)
        self._opener = opener
        self._room = room

    async def on_enter(self) -> None:  # called when agent session is ready
        # Brief delay to let the client fully subscribe to tracks & events
        await asyncio.sleep(0.8)

        # Speak greeting audio — LiveKit automatically publishes transcription to room
        await self.session.say(self._opener)


# ---------------------------------------------------------------------------
# Email utility — sends chat history to Rajesh on session end
# ---------------------------------------------------------------------------
NON_NAME_WORDS = {
    "skills", "skill", "experience", "exp", "projects", "project", "resume", "work", "tcs",
    "education", "hello", "hi", "hey", "hola", "greetings", "good", "morning", "afternoon",
    "evening", "yes", "no", "nope", "nah", "never", "why", "what", "who", "where", "how",
    "when", "ok", "okay", "sure", "fine", "cool", "yeah", "yep", "thanks", "thank", "you",
    "bye", "goodbye", "help", "contact", "email", "phone", "details", "rajesh", "raj",
    "assistant", "ai", "bot", "here", "there", "java", "spring", "springboot", "kafka",
    "aws", "docker", "kubernetes", "microservices", "angular", "react", "node", "python",
    "sql", "oracle", "database", "backend", "frontend", "nothing", "none", "later", "skip",
}


def _extract_visitor_info(dialogue: list[tuple[str, str]]) -> tuple[str, str]:
    """Extract visitor name and contact info (email/phone) from conversation dialogue."""
    visitor_name = "Unknown Visitor"
    contacts: list[str] = []

    for i, (speaker, text) in enumerate(dialogue):
        # Extract contact info from visitor messages
        if speaker == "VISITOR":
            emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
            phones = re.findall(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+?\d{10,14}", text)
            for e in emails:
                if e not in contacts:
                    contacts.append(e)
            for p in phones:
                cleaned = re.sub(r"[\s\-()]", "", p)
                if len(cleaned) >= 10 and cleaned not in contacts:
                    contacts.append(p)

        # Detect visitor name
        if visitor_name == "Unknown Visitor":
            if speaker == "VISITOR":
                # Pattern: 'My name is X' or 'I am X' or 'Call me X'
                m = re.search(r"(?:my name is|i am|i'm|this is|call me)\s+([a-zA-Z\s]{2,25})", text, re.IGNORECASE)
                if m:
                    cand = m.group(1).strip().split()
                    cand_words = [w for w in cand if w.lower() not in NON_NAME_WORDS]
                    if cand_words:
                        visitor_name = " ".join(cand_words[:2]).title()
                elif i > 0 and "name" in dialogue[i - 1][1].lower():
                    # Direct short answer to 'what is your name?'
                    words = [w for w in re.split(r"[\s,!.?]+", text) if w]
                    if 1 <= len(words) <= 2 and all(w.isalpha() and w.lower() not in NON_NAME_WORDS for w in words):
                        visitor_name = " ".join(words).title()
            elif speaker == "RAJ (AI)":
                # Raj recognized and greeted by name: 'Nice to meet you, Amit!'
                m = re.search(r"(?:nice to meet you|welcome|pleased to meet you),?\s+([A-Z][a-z]+)", text, re.IGNORECASE)
                if m and m.group(1).lower() not in NON_NAME_WORDS:
                    visitor_name = m.group(1).title()

    visitor_contact = ", ".join(contacts) if contacts else "Not shared"
    return visitor_name, visitor_contact


async def _send_chat_email(
    session: AgentSession,
    room_name: str,
    fallback_transcript: list[tuple[str, str]] | None = None,
) -> None:
    """Extract chat history from the session (with fallback) and email it to Rajesh."""
    smtp_email = os.environ.get("SMTP_EMAIL", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    smtp_to = os.environ.get("NOTIFICATION_EMAIL", smtp_email)

    if not smtp_email or not smtp_password:
        logger.info("SMTP not configured — skipping chat email (set SMTP_EMAIL & SMTP_PASSWORD)")
        return

    dialogue: list[tuple[str, str]] = []

    # 1. Attempt extraction from LiveKit AgentSession history
    try:
        chat_ctx = getattr(session, "history", None) or getattr(session, "_chat_ctx", None)
        if chat_ctx is not None:
            raw_items = getattr(chat_ctx, "items", None)
            items = raw_items() if callable(raw_items) else (raw_items if raw_items is not None else [])
            if not items and callable(getattr(chat_ctx, "messages", None)):
                try:
                    items = chat_ctx.messages()
                except Exception:
                    pass

            for item in items:
                role = str(getattr(item, "role", "")).lower()
                if "system" in role:
                    continue
                if getattr(item, "type", "") in ("function_call", "tool_call", "function_call_output", "tool_call_output"):
                    continue

                text = getattr(item, "text_content", None)
                if not text:
                    content = getattr(item, "content", "")
                    if isinstance(content, list):
                        text = " ".join(str(c) for c in content if c)
                    elif content:
                        text = str(content)
                text = (text or "").strip()
                if not text:
                    continue

                speaker = "VISITOR" if "user" in role else "RAJ (AI)"
                dialogue.append((speaker, text))
    except Exception as e:
        logger.warning(f"Failed to extract session.history items: {e}", exc_info=True)

    # 2. Integrate fallback transcript
    if not dialogue and fallback_transcript:
        dialogue = list(fallback_transcript)
    elif fallback_transcript:
        # If the opener greeting was missed in session.history, prepend it
        if dialogue and dialogue[0][0] == "VISITOR":
            for spk, txt in fallback_transcript:
                if spk == "RAJ (AI)":
                    dialogue.insert(0, (spk, txt))
                    break

    visitor_name, visitor_contact = _extract_visitor_info(dialogue)

    now = datetime.now()
    if dialogue:
        chat_text = "\n\n".join(f"{spk}: {txt}" for spk, txt in dialogue)
    else:
        chat_text = "(No conversation messages recorded)"

    # Build email body
    subject = f"🤖 Portfolio Chat — {visitor_name} — {now.strftime('%d %b %Y %I:%M %p')}"
    body = f"""Hi Rajesh,

Someone just chatted with your AI Portfolio Assistant!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 VISITOR DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Name:    {visitor_name}
  Contact: {visitor_contact}
  Room:    {room_name}
  Date:    {now.strftime('%d %b %Y, %I:%M:%S %p')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬 CHAT HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chat_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This email was sent automatically by your AI Portfolio Assistant.
Check your LiveKit dashboard for more details.
"""

    # Send via Gmail SMTP in a thread (blocking I/O)
    def _smtp_send() -> None:
        msg = MIMEMultipart()
        msg["From"] = smtp_email
        msg["To"] = smtp_to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.send_message(msg)

    try:
        await asyncio.to_thread(_smtp_send)
        logger.info(f"✅ Chat email sent for visitor '{visitor_name}' in room {room_name}")
    except Exception as e:
        logger.error(f"❌ Failed to send chat email: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    # ------------------------------------------------------------------
    # Read time-of-day greeting from job metadata or room metadata
    # ------------------------------------------------------------------
    room_meta_raw = ctx.job.metadata or ctx.room.metadata or "{}"
    try:
        room_meta: dict = json.loads(room_meta_raw)
    except json.JSONDecodeError:
        room_meta = {}

    greeting = room_meta.get("greeting", "Hello")
    opener = _build_opener(greeting)

    # In-memory conversation log to ensure 100% transcript capture for emails
    conversation_log: list[tuple[str, str]] = [("RAJ (AI)", opener)]

    # ------------------------------------------------------------------
    # Build the agent session (STT → LLM → TTS pipeline)
    # ------------------------------------------------------------------

    # ── No local VAD ──────────────────────────────────────────────────
    # Silero VAD runs an ONNX model on every audio frame. On Render free
    # tier (~0.1 CPU) it can't keep up with realtime audio, causing the
    # delay to snowball (0.18s → 14.87s in logs) and audio to break.
    #
    # Instead we use turn_detection="stt" which delegates end-of-speech
    # detection to Deepgram's server-side endpointing — zero local CPU.
    # ──────────────────────────────────────────────────────────────────

    session = AgentSession(
        stt=deepgram.STT(
            model="nova-2",
            language="en-US",
            endpointing_ms=250,     # Deepgram server-side end-of-utterance detection
            vad_events=True,        # Deepgram sends VAD events over websocket
        ),
        llm=FALLBACK_LLM,           # Pre-warmed shared LLM adapter (zero instantiation latency)
        tts=deepgram.TTS(model="aura-helios-en"),
        vad=None,                   # MUST be None: prevents LiveKit from auto-loading Silero ONNX VAD locally
        conn_options=SessionConnectOptions(
            llm_conn_options=APIConnectOptions(timeout=20.0, max_retry=2),
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",   # server-side turn detection via Deepgram
            endpointing=EndpointingOptions(min_delay=0.25, max_delay=1.5),
            interruption=InterruptionOptions(
                enabled=True,
                min_words=1,
                min_duration=0.3,
                resume_false_interruption=False,
            ),
            preemptive_generation=PreemptiveGenerationOptions(enabled=True, preemptive_tts=True),
        ),
    )

    # ------------------------------------------------------------------
    # Phase 3: DataChannel text-input handler
    # Routes visitor's typed messages through the same session / chat history
    # Coordinates turn claiming, cancels in-flight replies, and interrupts speech
    # ------------------------------------------------------------------
    active_text_task: asyncio.Task[None] | None = None

    @ctx.room.on("data_received")
    def on_data_received(packet: rtc.DataPacket) -> None:
        nonlocal active_text_task
        try:
            msg = json.loads(bytes(packet.data).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        if msg.get("type") != "text_input":
            return

        user_text: str = msg.get("text", "").strip()
        if not user_text:
            return

        logger.info(f"Received text input from visitor: '{user_text}'")
        conversation_log.append(("VISITOR", user_text))

        # 1. Immediately cancel any pending in-flight text reply task
        if active_text_task and not active_text_task.done():
            active_text_task.cancel()

        # 2. Immediately interrupt any in-flight agent speech
        int_fut = None
        try:
            int_fut = session.interrupt(force=True)
        except Exception as e:
            logger.debug("Interrupt notice: %s", e)

        async def _reply_to_text(text: str) -> None:
            try:
                if int_fut and not int_fut.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(int_fut), timeout=0.3)
                    except Exception:
                        pass
                async with session._claim_user_turn():
                    await session.generate_reply(user_input=text)
                logger.info("Successfully generated reply for text input")
            except asyncio.CancelledError:
                logger.info(f"Text reply task cancelled for: '{text}'")
            except Exception as ex:
                logger.error(f"Error generating reply for text input: {ex}", exc_info=True)

        active_text_task = asyncio.create_task(_reply_to_text(user_text))

    # ------------------------------------------------------------------
    # Start agent
    # Disable server-side APM (AGC/AEC) since browser client already runs hardware AGC/AEC/NS;
    # this completely prevents livekit_ffi event-loop stalls under concurrent load.
    # Disable transcription sync math (sync_transcription=False) to eliminate speaking rate FFT.
    # ------------------------------------------------------------------
    await session.start(
        agent=ResumeAgent(opener=opener, room=ctx.room),
        room=ctx.room,
        room_options=RoomOptions(
            audio_input=AudioInputOptions(
                auto_gain_control=False,
            ),
            text_output=TextOutputOptions(
                sync_transcription=False,
            ),
        ),
    )

    # ------------------------------------------------------------------
    # Phase 4: Auto-email chat history on visitor disconnect
    # ------------------------------------------------------------------
    email_sent = False  # guard against duplicate sends

    def _trigger_email_send(reason: str) -> None:
        nonlocal email_sent
        if email_sent:
            return
        email_sent = True
        logger.info(f"Triggering chat email ({reason}) for room '{ctx.room.name}'...")
        asyncio.create_task(
            _send_chat_email(
                session,
                ctx.room.name or "unknown-room",
                fallback_transcript=conversation_log,
            )
        )

    @ctx.room.on("participant_disconnected")
    def on_participant_left(participant) -> None:
        # Only trigger for the visitor (not the agent itself)
        if participant.identity == ctx.room.local_participant.identity:
            return
        _trigger_email_send(f"visitor '{participant.identity}' disconnected")

    @ctx.room.on("disconnected")
    def on_room_left() -> None:
        _trigger_email_send("room disconnected")


# ---------------------------------------------------------------------------
# Prewarm function — runs during worker startup before any jobs arrive.
# Eliminates the 2.3+ second event-loop freeze on the first user query!
# ---------------------------------------------------------------------------
def prewarm(proc: JobProcess) -> None:
    logger.info("Running agent prewarm to avoid realtime event-loop stalls...")
    try:
        import livekit.agents.llm.async_toolset  # noqa: F401
    except Exception as e:
        logger.debug("Toolset prewarm notice: %s", e)
    try:
        import anyio
        import anyio.abc._testing                # noqa: F401
        import anyio._core._sockets             # noqa: F401
        import anyio._backends._asyncio         # noqa: F401
    except Exception as e:
        logger.debug("Anyio prewarm notice: %s", e)
    try:
        from livekit.agents.tokenize import basic
        basic.hyphenate_word("prewarm")
    except Exception as e:
        logger.debug("Hyphenate prewarm notice: %s", e)
    try:
        from livekit.agents.voice.transcription import _speaking_rate  # noqa: F401
    except Exception as e:
        logger.debug("Speaking rate prewarm notice: %s", e)
    try:
        import ssl
        _p_ssl = ssl.create_default_context()
        _p_ssl.load_default_certs()
    except Exception as e:
        logger.debug("SSL default certs prewarm notice: %s", e)
    try:
        import asyncio
        asyncio.to_thread(lambda: None)
    except Exception as e:
        logger.debug("ThreadPool prewarm notice: %s", e)
    try:
        import livekit.agents.utils.http_context
        livekit.agents.utils.http_context._create_ssl_context()
    except Exception as e:
        logger.debug("SSL prewarm notice: %s", e)
    try:
        import numpy as np
        np.fft.rfft(np.zeros(512))
        np.divide(np.array([1.0]), np.array([1.0]))
    except Exception as e:
        logger.debug("Numpy prewarm notice: %s", e)
    logger.info("Prewarm complete.")


# ---------------------------------------------------------------------------
# Embedded Token Server & Entry point
# ---------------------------------------------------------------------------
import threading

def start_embedded_token_server() -> None:
    # Only run if explicitly requested; in Render production, the start command
    # already runs `uvicorn agent.token_server:app --host 0.0.0.0 --port $PORT` directly.
    if os.environ.get("START_EMBEDDED_SERVER", "").lower() != "true":
        return

    port = os.environ.get("PORT")
    if not port:
        return

    def _run() -> None:
        import uvicorn
        try:
            from token_server import app as token_app
        except ModuleNotFoundError:
            from agent.token_server import app as token_app
        logger.info(f"Starting embedded token server thread on port {port}...")
        try:
            uvicorn.run(token_app, host="0.0.0.0", port=int(port), log_level="info")
        except OSError as e:
            logger.warning(f"Embedded token server skipped — port {port} already in use: {e}")

    t = threading.Thread(target=_run, daemon=True, name="embedded_token_server")
    t.start()


if __name__ == "__main__":
    start_embedded_token_server()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="rajesh-agent",
            num_idle_processes=1,
            job_executor_type=JobExecutorType.THREAD,
            # Raise threshold so worker stays available on CPU-constrained hosts like Render free tier
            load_threshold=0.95,
        )
    )
