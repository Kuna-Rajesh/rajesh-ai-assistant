import React, { useState } from 'react';
import type { TranscriptEntry } from './CaptionsStrip';
import CaptionsStrip from './CaptionsStrip';
import TextInput from './TextInput';

interface CallScreenProps {
  connectionState: 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error';
  isMuted: boolean;
  agentSpeaking: boolean;
  transcript: TranscriptEntry[];
  errorMessage: string | null;
  onToggleMute: () => void;
  onHangup: () => void;
  onSendText: (text: string) => void;
}

// ---- Inline SVG icons ----

const MicOnIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
    <line x1="12" y1="19" x2="12" y2="23" />
    <line x1="8" y1="23" x2="16" y2="23" />
  </svg>
);

const MicOffIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="1" y1="1" x2="23" y2="23" />
    <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6" />
    <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23" />
    <line x1="12" y1="19" x2="12" y2="23" />
    <line x1="8" y1="23" x2="16" y2="23" />
  </svg>
);

const PhoneOffIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.68 13.31a16 16 0 0 0 3.41 2.6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7 2 2 0 0 1 1.72 2v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07" />
    <path d="M14.46 14.46l-9.1-9.1" />
    <path d="M3.35 7c-.34 1.03-.52 2.1-.52 3.19" />
    <path d="M5.05 4.05A10 10 0 0 1 21.95 10" />
  </svg>
);

const TextIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
);

// ---- Profile photo with initials fallback ----

const PhotoOrFallback: React.FC = () => {
  const [imgError, setImgError] = React.useState(false);
  if (imgError) {
    return (
      <div className="profile-avatar-fallback" aria-label="Rajesh Kuna profile photo">
        RK
      </div>
    );
  }
  return (
    <img
      src="/rajesh.png"
      alt="Rajesh Kuna"
      className="profile-photo"
      onError={() => setImgError(true)}
    />
  );
};

// ---- Status badge ----

function connLabel(state: CallScreenProps['connectionState']): string {
  switch (state) {
    case 'connecting': return 'Connecting…';
    case 'connected': return 'Connected';
    case 'disconnected': return 'Call ended';
    case 'error': return 'Error';
    default: return 'Starting…';
  }
}

// ---- Component ----

const CallScreen: React.FC<CallScreenProps> = ({
  connectionState,
  isMuted,
  agentSpeaking,
  transcript,
  errorMessage,
  onToggleMute,
  onHangup,
  onSendText,
}) => {
  const [textOpen, setTextOpen] = useState(false);

  const isConnected = connectionState === 'connected';
  const isEnded = connectionState === 'disconnected';

  return (
    <div className="call-bg">
      <div className="call-card">

        {/* ── Profile section ── */}
        <div className="photo-section">
          <div className={`photo-ring${agentSpeaking ? ' speaking' : ''}`}>
            <PhotoOrFallback />
          </div>

          <div className="profile-info">
            <div className="profile-name">Rajesh Kuna</div>
            <div className="profile-title">Java Backend Developer · TCS · Hyderabad</div>

            <div className={`conn-badge ${agentSpeaking ? 'speaking' : isConnected ? 'connected' : isEnded ? 'disconnected' : 'connecting'}`}
              aria-live="polite" aria-atomic="true">
              {agentSpeaking ? (
                <>
                  <div className="equalizer-bars">
                    <span className="eq-bar bar1" />
                    <span className="eq-bar bar2" />
                    <span className="eq-bar bar3" />
                  </div>
                  Speaking…
                </>
              ) : (
                <>
                  <span className="dot" />
                  {connLabel(connectionState)}
                </>
              )}
            </div>

            {errorMessage && (
              <div className="status-error" role="alert">{errorMessage}</div>
            )}
          </div>
        </div>

        {/* ── Captions ── */}
        <CaptionsStrip entries={transcript} />

        {/* ── Controls ── */}
        {!isEnded && (
          <div className="controls-bar" role="toolbar" aria-label="Call controls">

            {/* Mute */}
            <button
              id="btn-mute"
              className={`ctrl-btn mute${isMuted ? ' muted' : ''}`}
              onClick={onToggleMute}
              disabled={!isConnected}
              aria-label={isMuted ? 'Unmute microphone' : 'Mute microphone'}
              title={isMuted ? 'Unmute' : 'Mute'}
            >
              {isMuted ? <MicOffIcon /> : <MicOnIcon />}
            </button>

            {/* End call */}
            <button
              id="btn-end-call"
              className="ctrl-btn end-call"
              onClick={onHangup}
              aria-label="End call"
              title="End call"
            >
              <PhoneOffIcon />
            </button>

            {/* Text toggle */}
            <button
              id="btn-text-toggle"
              className={`ctrl-btn text-toggle${textOpen ? ' active' : ''}`}
              onClick={() => setTextOpen((v) => !v)}
              aria-label={textOpen ? 'Close text input' : 'Open text input'}
              aria-expanded={textOpen}
              title="Text fallback"
            >
              <TextIcon />
            </button>

          </div>
        )}

        {/* Call-ended message */}
        {isEnded && (
          <div className="status-overlay" style={{ flex: 'unset', padding: '12px 0' }}>
            <p className="status-text">Call ended. Refresh to start a new conversation.</p>
          </div>
        )}

        {/* ── Text input panel ── */}
        <TextInput
          open={textOpen && isConnected}
          onSend={onSendText}
          disabled={!isConnected}
        />

      </div>
    </div>
  );
};

export default CallScreen;
