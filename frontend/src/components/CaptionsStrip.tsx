import React, { useEffect, useRef } from 'react';

export interface TranscriptEntry {
  id: string;
  speaker: 'agent' | 'user';
  text: string;
}

interface CaptionsStripProps {
  entries: TranscriptEntry[];
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
            <span className="caption-label">
              {entry.speaker === 'agent' ? 'Raj' : 'You'}
            </span>
            <span className="caption-text">{entry.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CaptionsStrip;
