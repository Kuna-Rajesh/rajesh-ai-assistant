"""
Token-minting server for the Rajesh AI Resume Assistant frontend.

Run:  uvicorn token_server:app --port 8880 --reload
      (from inside /agent with the venv active)

GET /token?identity=visitor-xyz&greeting=Good+afternoon
Returns: { "token": "<jwt>", "url": "<livekit_ws_url>", "room": "<room_name>" }
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, VideoGrants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("token_server")

import asyncio
import time

dispatch_lock = asyncio.Lock()
last_dispatch_time: dict[str, float] = {}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")
ROOM_NAME = "rajesh-resume-room"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Rajesh AI Assistant — Token Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/token")
async def get_token(
    identity: str = Query(default="visitor", description="Unique participant identity"),
    greeting: str = Query(default="Hello", description="Time-of-day greeting for the agent opener"),
) -> dict:
    """
    Mint a LiveKit room-join JWT.
    The greeting is embedded in the room's metadata so the agent can
    deliver the correct time-of-day opener without needing clock access.
    """
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(status_code=500, detail="Server credentials not configured")

    import json
    room_metadata = json.dumps({"greeting": greeting})

    # Dispatch agent worker to room if not already active (thread-safe with cooldown)
    async with dispatch_lock:
        now = time.time()
        if now - last_dispatch_time.get(ROOM_NAME, 0) > 10:
            try:
                from livekit.api import LiveKitAPI
                from livekit.protocol import agent_dispatch, room as proto_room
                async with LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET) as api:
                    # 1. Ensure room exists on LiveKit Cloud
                    try:
                        await api.room.create_room(proto_room.CreateRoomRequest(name=ROOM_NAME, empty_timeout=300))
                    except Exception as ex:
                        logger.debug(f"Room create notice: {ex}")

                    # 2. Update room metadata with time-of-day greeting
                    try:
                        await api.room.update_room_metadata(proto_room.UpdateRoomMetadataRequest(room=ROOM_NAME, metadata=room_metadata))
                    except Exception as ex:
                        logger.debug(f"Room metadata update notice: {ex}")

                    # 3. Clean up stale agent dispatches
                    try:
                        existing = await api.agent_dispatch.list_dispatch(ROOM_NAME)
                        for d in existing:
                            try:
                                await api.agent_dispatch.delete_dispatch(d.id, ROOM_NAME)
                                logger.info(f"Cleaned up stale agent dispatch {d.id}")
                            except Exception:
                                pass
                    except Exception as ex:
                        logger.debug(f"List dispatch notice: {ex}")

                    # 4. Create fresh agent dispatch
                    dispatch = await api.agent_dispatch.create_dispatch(
                        agent_dispatch.CreateAgentDispatchRequest(
                            agent_name="rajesh-agent",
                            room=ROOM_NAME,
                            metadata=room_metadata,
                        )
                    )
                    last_dispatch_time[ROOM_NAME] = now
                    logger.info(f"Successfully created agent dispatch {dispatch.id} for room {ROOM_NAME}")
            except Exception as e:
                logger.warning(f"Agent dispatch notice: {e}")
        else:
            logger.info(f"Skipped duplicate agent dispatch request within cooldown for room {ROOM_NAME}")

    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=ROOM_NAME,
                # Allow frontend to set room metadata on join
                room_admin=False,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_metadata(room_metadata)
        .to_jwt()
    )

    return {"token": token, "url": LIVEKIT_URL, "room": ROOM_NAME, "greeting": greeting}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "room": ROOM_NAME, "livekit_url": LIVEKIT_URL}
