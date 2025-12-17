import React, { useState, useEffect, useRef } from 'react';
import { ChatInterface } from './ChatInterface';
import { Message } from './Message';
import { TextSelectionHandler } from './TextSelectionHandler';
import { apiClient, ChatMessage, Session, QueryMode } from './api';
import '../../css/rag-chatbot.css';

interface RagChatbotProps {
  backendUrl?: string;
}

// Define interface for session state in localStorage
interface StoredSessionState {
  sessionId: string;
  messages: ChatMessage[];
  currentMode: QueryMode;
  selectedText: string | null;
  lastActivePage: string;
  timestamp: number; // Unix timestamp for expiration
}

export const RagChatbot: React.FC<RagChatbotProps> = ({ backendUrl = '/api/v1' }) => {
  const [session, setSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentMode, setCurrentMode] = useState<QueryMode>('general');
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const [currentPageUrl, setCurrentPageUrl] = useState<string>('');
  const [isOpen, setIsOpen] = useState<boolean>(false); // Track if chatbot is open/closed (starts minimized)
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Load session state from localStorage on component mount
  useEffect(() => {
    const loadStoredSession = () => {
      try {
        const stored = localStorage.getItem('ragChatbotSession');
        if (stored) {
          const storedState: StoredSessionState = JSON.parse(stored);

          // Check if session is still valid (not older than 24 hours)
          const now = Date.now();
          const sessionAge = now - storedState.timestamp;
          const maxAge = 24 * 60 * 60 * 1000; // 24 hours in milliseconds

          if (sessionAge < maxAge) {
            // Attempt to validate session with backend
            apiClient.getSession(storedState.sessionId)
              .then(validSession => {
                if (validSession) {
                  setSession(validSession);
                  setMessages(storedState.messages);
                  setCurrentMode(storedState.currentMode);
                  setSelectedText(storedState.selectedText);
                } else {
                  // Session no longer valid, clear storage
                  localStorage.removeItem('ragChatbotSession');
                }
              })
              .catch(() => {
                // If backend validation fails, clear storage
                localStorage.removeItem('ragChatbotSession');
              });
          } else {
            // Session expired, clear storage
            localStorage.removeItem('ragChatbotSession');
          }
        }
      } catch (error) {
        console.error('Error loading stored session:', error);
        // Clear any corrupted storage
        localStorage.removeItem('ragChatbotSession');
      }
    };

    loadStoredSession();

    // Initialize session if none loaded
    if (!session) {
      const initSession = async () => {
        try {
          const newSession = await apiClient.createSession();
          setSession(newSession);

          // Get current page URL
          setCurrentPageUrl(window.location.pathname);
        } catch (error) {
          console.error('Failed to create session:', error);
        }
      };

      initSession();
    } else {
      // Update page URL if we loaded a session
      setCurrentPageUrl(window.location.pathname);
    }

    // Set up text selection handler
    const handleSelection = () => {
      const selectedText = window.getSelection()?.toString().trim();
      if (selectedText) {
        setSelectedText(selectedText);
        setCurrentMode('selected_text');
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  // Save session state to localStorage whenever it changes
  useEffect(() => {
    if (session) {
      const storedState: StoredSessionState = {
        sessionId: session.session_id,
        messages,
        currentMode,
        selectedText,
        lastActivePage: currentPageUrl,
        timestamp: Date.now()
      };

      try {
        localStorage.setItem('ragChatbotSession', JSON.stringify(storedState));
      } catch (error) {
        console.error('Error saving session to localStorage:', error);
      }
    }
  }, [session, messages, currentMode, selectedText, currentPageUrl]);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || !session || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date().toISOString(),
      sources: []
    };

    // Add user message to UI immediately
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      let response;
      if (currentMode === 'selected_text' && selectedText) {
        // Use selected text query
        response = await apiClient.querySelectedText(
          session.session_id,
          inputValue,
          selectedText,
          { page_url: currentPageUrl, section_context: selectedText.substring(0, 200) }
        );
      } else {
        // Use general query
        response = await apiClient.queryGeneral(
          session.session_id,
          inputValue,
          { page_url: currentPageUrl, selected_mode: false }
        );
      }

      const assistantMessage: ChatMessage = {
        id: response.response_id,
        role: 'assistant',
        content: response.response,
        timestamp: response.timestamp,
        sources: response.sources || []
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Clear selected text after query if in selected text mode
      if (currentMode === 'selected_text') {
        setSelectedText(null);
        setCurrentMode('general');
      }
    } catch (error: any) {
      console.error('Error sending message:', error);

      // Check for specific error types and provide appropriate feedback
      let errorMessageContent = 'Sorry, I encountered an error processing your request. Please try again.';

      if (error.message.includes('rate limit')) {
        errorMessageContent = 'Rate limit exceeded. Please wait a moment before sending another message.';
      } else if (error.message.includes('session')) {
        errorMessageContent = 'Session expired. Please refresh the page to continue.';
      } else if (error.message.includes('network') || error.message.includes('fetch')) {
        errorMessageContent = 'Unable to connect to the server. Please check your connection and try again.';
      } else if (error.message.includes('400') || error.message.includes('Bad Request')) {
        errorMessageContent = 'Invalid request. Please try rephrasing your question.';
      }

      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: errorMessageContent,
        timestamp: new Date().toISOString(),
        sources: []
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearContext = async () => {
    if (!session) return;

    try {
      await apiClient.clearContext(session.session_id);
      setSelectedText(null);
      setCurrentMode('general');
    } catch (error) {
      console.error('Error clearing context:', error);
    }
  };

  const handleLoadHistory = async () => {
    if (!session) return;

    try {
      const history = await apiClient.getHistory(session.session_id);
      setMessages(history.messages);
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  return (
    <div className={`rag-chatbot-container ${isOpen ? 'open' : 'minimized'}`}>
      {isOpen ? (
        <>
          <div className="rag-chatbot-header">
            <h3>Physical AI & Humanoid Robotics Assistant</h3>
            <div className="chatbot-mode-indicator">
              Mode: <span className={`mode-${currentMode}`}>{currentMode === 'general' ? 'General' : 'Selected Text'}</span>
              {selectedText && (
                <button
                  className="clear-context-btn"
                  onClick={handleClearContext}
                  title="Clear selected text context"
                >
                  Clear
                </button>
              )}
              <button
                className="minimize-btn"
                onClick={() => setIsOpen(false)}
                title="Minimize chatbot"
              >
                −
              </button>
            </div>
          </div>

          <div className="rag-chatbot-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <p>Hello! I'm your Physical AI & Humanoid Robotics book assistant.</p>
                <p>You can ask me questions about the book content, or select text on the page and ask questions about it specifically.</p>
              </div>
            ) : (
              messages.map((message) => (
                <Message
                  key={message.id}
                  message={message}
                />
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          <ChatInterface
            inputValue={inputValue}
            setInputValue={setInputValue}
            isLoading={isLoading}
            handleSubmit={handleSubmit}
            currentMode={currentMode}
            selectedText={selectedText}
          />
        </>
      ) : (
        // Minimized view - just a button to expand
        <button
          className="expand-btn"
          onClick={() => setIsOpen(true)}
          title="Open chatbot"
        >
          🤖
        </button>
      )}

      <TextSelectionHandler
        selectedText={selectedText}
        onClearSelection={() => {
          setSelectedText(null);
          setCurrentMode('general');
        }}
      />
    </div>
  );
};