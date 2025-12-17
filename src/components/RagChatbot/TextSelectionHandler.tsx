import React from 'react';

interface TextSelectionHandlerProps {
  selectedText: string | null;
  onClearSelection: () => void;
}

export const TextSelectionHandler: React.FC<TextSelectionHandlerProps> = ({
  selectedText,
  onClearSelection
}) => {
  if (!selectedText) {
    return null;
  }

  return (
    <div className="text-selection-overlay">
      <div className="selected-text-info">
        <span className="selected-text-preview">
          Selected: "{selectedText.substring(0, 50)}{selectedText.length > 50 ? '...' : ''}"
        </span>
        <button
          className="clear-selection-btn"
          onClick={onClearSelection}
          title="Clear selection"
        >
          Clear
        </button>
      </div>
    </div>
  );
};