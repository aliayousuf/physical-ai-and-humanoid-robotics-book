import React from 'react';

interface ChatInterfaceProps {
  inputValue: string;
  setInputValue: (value: string) => void;
  isLoading: boolean;
  handleSubmit: (e: React.FormEvent) => void;
  currentMode: string;
  selectedText: string | null;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  inputValue,
  setInputValue,
  isLoading,
  handleSubmit,
  currentMode,
  selectedText
}) => {
  return (
    <div className="chat-input-container">
      {selectedText && (
        <div className="selected-text-preview">
          <strong>Selected Text:</strong> {selectedText.substring(0, 100)}...
          <span className="mode-indicator selected-text-mode">Selected Text Mode</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="chat-input-form">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder={
            currentMode === 'selected_text'
              ? "Ask about the selected text..."
              : "Ask about the book content..."
          }
          disabled={isLoading}
          className="chat-input"
        />
        <button
          type="submit"
          disabled={isLoading || !inputValue.trim()}
          className="chat-submit-btn"
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
};