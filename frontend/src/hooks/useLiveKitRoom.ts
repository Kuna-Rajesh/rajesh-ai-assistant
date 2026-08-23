import { useEffect, useRef, useState, useCallback } from 'react';
import {
  Room,
  RoomEvent,
  ConnectionState,
  LocalParticipant,
  Participant,
  TrackPublication,
  Track,
  RemoteTrack,
} from 'livekit-client';
import type { TranscriptionSegment } from 'livekit-client';
import type { TranscriptEntry } from '../components/CaptionsStrip';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export type RoomConnectionState = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error';

interface UseLiveKitRoomReturn {
  connectionState: RoomConnectionState;
  isMuted: boolean;
  transcript: TranscriptEntry[];
  toggleMute: () => void;
  disconnect: () => void;
  sendTextMessage: (text: string) => Promise<void>;
  agentSpeaking: boolean;
  errorMessage: string | null;
}

// ---------------------------------------------------------------------------
// Time-of-day greeting helper
// ---------------------------------------------------------------------------
function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return 'Good morning';
  if (hour < 17) return 'Good afternoon';
  return 'Good evening';
}

// ---------------------------------------------------------------------------
// Unique ID helper
// ---------------------------------------------------------------------------
let _idCounter = 0;
function uid(): string {
  return `t-${Date.now()}-${++_idCounter}`;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------
const TOKEN_SERVER_URL = import.meta.env.VITE_TOKEN_SERVER_URL ?? 'http://localhost:8880';

export function useLiveKitRoom(): UseLiveKitRoomReturn {
  const roomRef = useRef<Room | null>(null);

  const [connectionState, setConnectionState] = useState<RoomConnectionState>('idle');
  const [isMuted, setIsMuted] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [agentSpeaking, setAgentSpeaking] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // -------------------------------------------------------------------------
  // Transcript helpers
  // -------------------------------------------------------------------------
  const addEntry = useCallback((speaker: 'agent' | 'user', text: string) => {
    setTranscript((prev) => {
      // Merge with last entry of same speaker if it came in rapidly (streaming chunks)
      const last = prev[prev.length - 1];
      if (last && last.speaker === speaker && Date.now() - Number(last.id.split('-')[1]) < 1500) {
        return [...prev.slice(0, -1), { ...last, text: last.text + ' ' + text }];
      }
      return [...prev, { id: uid(), speaker, text }];
    });
  }, []);

  // -------------------------------------------------------------------------
  // Connect on mount
  // -------------------------------------------------------------------------
  useEffect(() => {
    let room: Room;
    let cancelled = false;

    async function connect() {
      setConnectionState('connecting');
      setErrorMessage(null);

      try {
        const greeting = getGreeting();
        const identity = `visitor-${Math.random().toString(36).slice(2, 8)}`;

        const res = await fetch(
          `${TOKEN_SERVER_URL}/token?identity=${encodeURIComponent(identity)}&greeting=${encodeURIComponent(greeting)}`
        );
        if (!res.ok) throw new Error(`Token server error: ${res.status}`);
        const { token, url } = await res.json();

        if (cancelled) return;

        room = new Room({
          adaptiveStream: true,
          dynacast: true,
          audioCaptureDefaults: { echoCancellation: true, noiseSuppression: true },
        });
        roomRef.current = room;

        // --- Connection state ---
        room.on(RoomEvent.ConnectionStateChanged, (state: ConnectionState) => {
          if (state === ConnectionState.Connected) setConnectionState('connected');
          else if (state === ConnectionState.Disconnected) setConnectionState('disconnected');
          else if (state === ConnectionState.Reconnecting) setConnectionState('connecting');
        });

        // --- Transcription events (LiveKit sends these when agent speaks) ---
        room.on(
          RoomEvent.TranscriptionReceived,
          (
            segments: TranscriptionSegment[],
            participant?: Participant,
            _publication?: TrackPublication
          ) => {
            segments.forEach((seg) => {
              if (!seg.final) return; // only log completed sentences
              const isLocal = participant?.identity === room.localParticipant.identity;
              addEntry(isLocal ? 'user' : 'agent', seg.text);
            });
          }
        );

        // --- Track Subscription (play remote audio in browser speakers) ---
        room.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
          if (track.kind === Track.Kind.Audio) {
            const element = track.attach();
            document.body.appendChild(element);
          }
        });
        room.on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
          track.detach().forEach((el) => el.remove());
        });

        // --- Real-time speech detection ---
        room.on(RoomEvent.ActiveSpeakersChanged, (speakers: Participant[]) => {
          const isSpeaking = speakers.some(
            (p) => p.identity !== room.localParticipant.identity
          );
          setAgentSpeaking(isSpeaking);
        });

        room.on(RoomEvent.TrackMuted, (_pub: TrackPublication, participant: Participant) => {
          if (participant !== room.localParticipant) setAgentSpeaking(false);
        });
        room.on(RoomEvent.TrackUnmuted, (_pub: TrackPublication, participant: Participant) => {
          if (participant !== room.localParticipant) setAgentSpeaking(true);
        });

        // --- DataChannel: text responses from agent ---
        room.on(RoomEvent.DataReceived, (payload: Uint8Array, _participant?: Participant) => {
          try {
            const msg = JSON.parse(new TextDecoder().decode(payload));
            if (msg.type === 'text_response' && msg.text) {
              addEntry('agent', msg.text);
            }
          } catch {
            // ignore malformed packets
          }
        });

        await room.connect(url, token, { autoSubscribe: true });

        if (cancelled) {
          room.disconnect();
          return;
        }

        // Try unlocking audio context for mobile browsers
        try {
          await room.startAudio();
        } catch {
          // Audio playback will unlock on user's first tap
        }

        // Try enabling microphone gracefully (mobile browsers may require explicit tap)
        try {
          await room.localParticipant.setMicrophoneEnabled(true);
          setIsMuted(false);
        } catch (micErr) {
          console.warn('Microphone auto-enable restricted by browser policy:', micErr);
          setIsMuted(true);
          // Keep room connected so user can hear agent and interact via text or tap mic
        }

      } catch (err) {
        if (!cancelled) {
          setConnectionState('error');
          setErrorMessage(err instanceof Error ? err.message : 'Failed to connect');
        }
      }
    }

    connect();

    return () => {
      cancelled = true;
      roomRef.current?.disconnect();
      roomRef.current = null;
    };
  }, []); // connect once on mount

  // -------------------------------------------------------------------------
  // Controls
  // -------------------------------------------------------------------------
  const toggleMute = useCallback(async () => {
    const room = roomRef.current;
    const lp = room?.localParticipant;
    if (!lp) return;

    try {
      await room.startAudio();
    } catch {}

    const nextMuted = !isMuted;
    try {
      await lp.setMicrophoneEnabled(!nextMuted);
      setIsMuted(nextMuted);
    } catch (err) {
      console.error('Failed to toggle microphone:', err);
    }
  }, [isMuted]);

  const disconnect = useCallback(() => {
    roomRef.current?.disconnect();
    setConnectionState('disconnected');
  }, []);

  // -------------------------------------------------------------------------
  // Send text message via DataChannel
  // -------------------------------------------------------------------------
  const sendTextMessage = useCallback(async (text: string) => {
    const lp: LocalParticipant | undefined = roomRef.current?.localParticipant;
    if (!lp) return;

    // Optimistically show user message in transcript
    addEntry('user', text);

    const payload = new TextEncoder().encode(
      JSON.stringify({ type: 'text_input', text })
    );
    await lp.publishData(payload, { reliable: true });
  }, [addEntry]);

  return {
    connectionState,
    isMuted,
    transcript,
    toggleMute,
    disconnect,
    sendTextMessage,
    agentSpeaking,
    errorMessage,
  };
}
