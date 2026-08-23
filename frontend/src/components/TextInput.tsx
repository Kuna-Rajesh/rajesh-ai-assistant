import React, { useRef, useState } from 'react';

interface TextInputProps {
  open: boolean;
  onSend: (text: string) => void;
  disabled?: boolean;
}

const TextInput: React.FC<TextInputProps> = ({ open, onSend, disabled = false }) => {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
    // reset height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = `${Math.min(el.scrollHeight, 90)}px`;
    }
  };

  return (
    <div className={`text-panel${open ? ' open' : ''}`} aria-hidden={!open}>
      <div className="text-input-row">
        <textarea
          ref={textareaRef}
          id="text-input-field"
          placeholder="Type a message… (Enter to send)"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          rows={1}
          disabled={disabled || !open}
          aria-label="Text message to Raj"
        />
        <button
          id="send-text-btn"
          className="send-btn"
          onClick={handleSend}
          disabled={!value.trim() || disabled}
          aria-label="Send message"
          title="Send (Enter)"
        >
          {/* Send arrow icon */}
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default TextInput;
