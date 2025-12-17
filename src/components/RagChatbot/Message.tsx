import React from 'react';
import { ChatMessage } from './api';

interface MessageProps {
  message: ChatMessage;
}

export const Message: React.FC<MessageProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`message-container ${isUser ? 'user-message' : 'assistant-message'}`}>
      <div className="message-content">
        {message.content}
      </div>

      {message.sources && message.sources.length > 0 && (
        <div className="message-sources">
          <details>
            <summary>Sources</summary>
            <ul>
              {message.sources.map((source, index) => (
                <li key={index} className="source-item">
                  <a
                    href={source.page_reference}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {source.title}
                  </a>
                  <span className="relevance-score">Score: {source.relevance_score?.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </details>
        </div>
      )}
    </div>
  );
};