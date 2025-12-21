// Types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources: Array<{
    content_id: string;
    title: string;
    page_reference: string;
    relevance_score?: number;
  }>;
}

export interface Session {
  session_id: string;
  created_at: string;
  expires_at: string;
}

export interface QueryContext {
  page_url?: string;
  selected_mode?: boolean;
  section_context?: string;
}

export interface QueryResponse {
  response_id: string;
  session_id: string;
  query: string;
  response: string;
  sources: Array<{
    content_id: string;
    title: string;
    page_reference: string;
    relevance_score: number;
  }>;
  timestamp: string;
  query_mode: string;
}

export interface HistoryResponse {
  session_id: string;
  messages: ChatMessage[];
  pagination: {
    limit: number;
    offset: number;
    total: number;
  };
}

export type QueryMode = 'general' | 'selected_text';

// API Client
class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = '/api/v1') {
    this.baseUrl = baseUrl;
  }

  async createSession(): Promise<Session> {
    const response = await fetch(`${this.baseUrl}/chat/session`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ initial_context: 'Physical AI and Humanoid Robotics documentation' }),
    });

    if (!response.ok) {
      throw new Error(`Failed to create session: ${response.statusText}`);
    }

    return response.json();
  }

  async getSession(session_id: string): Promise<Session> {
    const response = await fetch(`${this.baseUrl}/chat/session/${session_id}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to get session: ${response.statusText}`);
    }

    return response.json();
  }

  async queryGeneral(
    session_id: string,
    query: string,
    context?: QueryContext
  ): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/chat/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id,
        query,
        mode: 'general',
        context: context || { page_url: window.location.pathname, selected_mode: false },
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = errorData.error?.message || `Query failed: ${response.statusText}`;
      throw new Error(errorMessage);
    }

    return response.json();
  }

  async querySelectedText(
    session_id: string,
    query: string,
    selected_text: string,
    context?: QueryContext
  ): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/chat/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id,
        query,
        mode: 'selected_text',
        selected_text,
        context: context || { page_url: window.location.pathname },
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = errorData.error?.message || `Selected text query failed: ${response.statusText}`;
      throw new Error(errorMessage);
    }

    return response.json();
  }

  async getHistory(
    session_id: string,
    limit: number = 20,
    offset: number = 0
  ): Promise<HistoryResponse> {
    const response = await fetch(
      `${this.baseUrl}/chat/session/${session_id}/history?limit=${limit}&offset=${offset}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = errorData.error?.message || `Failed to get history: ${response.statusText}`;
      throw new Error(errorMessage);
    }

    return response.json();
  }

  async clearContext(session_id: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/chat/session/${session_id}/context`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = errorData.error?.message || `Failed to clear context: ${response.statusText}`;
      throw new Error(errorMessage);
    }

    return response.json();
  }
}

// Create a singleton instance
// Use environment variable for production, fallback to relative path for development
const getBackendUrl = (): string => {
  // In browser environment
  if (typeof window !== 'undefined') {
    // If we're in production (not localhost), use the production backend URL
    if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      // IMPORTANT: Replace this URL with your actual Railway backend URL after deployment
      // Example: 'https://your-project-production.up.railway.app/api/v1'
      const PRODUCTION_BACKEND_URL = 'https://physical-ai-and-humanoid-robotics-book-production-f3fc.up.railway.app/api/v1';
      return PRODUCTION_BACKEND_URL;
    } else {
      // For local development, use the specific port where backend is running
      return 'http://127.0.0.1:8003/api/v1';
    }
  }

  // Default to relative path for development (will be proxied by Docusaurus)
  return '/api/v1';
};

export const apiClient = new ApiClient(getBackendUrl());

// Export for direct use if needed
export default apiClient;