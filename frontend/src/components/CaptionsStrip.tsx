import React, { useEffect, useRef } from 'react';

export interface TranscriptEntry {
  id: string;
  speaker: 'agent' | 'user';
  text: string;
  timestamp: number;  // epoch ms — set when entry is created
}

interface CaptionsStripProps {
  entries: TranscriptEntry[];
}

/** Format timestamp as "Sep 27, 2026 · 3:45:12 PM" */
function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  const date = d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
  const time = d.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  });
  return `${date} · ${time}`;
}

const CaptionsStrip: React.FC<CaptionsStripProps> = ({ entries }) => {
  const listRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new entries arrive
  useEffect(() => {
    const el = listRef.current;
    if (el) {
      requestAnimationFrame(() => {
        el.scrollTo({
          top: el.scrollHeight,
          behavior: 'smooth',
        });
      });
    }
  }, [entries]);

  if (entries.length === 0) {
    return (
      <div className="captions-wrap">
        <div className="captions-empty">
          Your conversation will appear here…
        </div>
      </div>
    );
  }

  return (
    <div className="captions-wrap">
      <div className="captions-list" ref={listRef} role="log" aria-live="polite">
        {entries.map((entry) => (
          <div
            key={entry.id}
            className={`caption-item ${entry.speaker}`}
            aria-label={`${entry.speaker === 'agent' ? 'Raj' : 'You'}: ${entry.text}`}
          >
            <div className="caption-header">
              <span className="caption-label">
                {entry.speaker === 'agent' ? 'Raj' : 'You'}
              </span>
              <span className="caption-time">
                {formatTimestamp(entry.timestamp)}
              </span>
            </div>
            <span className="caption-text">{entry.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CaptionsStrip;
