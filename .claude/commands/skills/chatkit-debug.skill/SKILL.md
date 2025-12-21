

skill:
  name: "ChatKit Debug & Troubleshooting"
  metadata:
    id: chatkit-debug
    category: debug
    tags:
      - chatkit
      - debug
      - troubleshooting
      - errors
    description: >
      Diagnoses and fixes common ChatKit integration issues across
      backend and frontend implementations.

quick_diagnostic_checklist:
  backend_health_check:
    - step: "Start backend server"
      command: "python backend/main.py"
    - step: "Test health endpoint"
      command: "curl http://localhost:8000/health"
    - step: "Check logs"
      description: "Look for import errors or stack traces"

  frontend_health_check:
    - step: "Start frontend dev server"
      command: "cd frontend && npm run dev"
    - step: "Check browser console"
      description: "Look for JavaScript runtime errors"
    - step: "Inspect network tab"
      description: "Verify /chatkit requests are sent and receive responses"

error_database:
  backend_import_errors:
    - error: "ModuleNotFoundError: chatkit.stores"
      wrong_code: "from chatkit.stores import Store"
      correct_code: "from chatkit.store import Store"

    - error: "ModuleNotFoundError: chatkit.models"
      wrong_code: "from chatkit.models import ..."
      correct_code: "from chatkit.types import ..."

    - error: "ImportError: Event"
      wrong_code: "from chatkit.server import Event"
      correct_code: "Remove import (Event does not exist)"

    - error: "ImportError: ClientToolCallOutputItem"
      wrong_code: "from chatkit.types import ClientToolCallOutputItem"
      correct_code: "Use Any type"

    - error: "ImportError: FilePart"
      wrong_code: "from chatkit.types import FilePart"
      correct_code: "Use Any type"

  backend_runtime_errors:
    - error: "Can't instantiate abstract class"
      cause: "Missing Store method implementations"
      solution: >
        Implement all 14 Store methods including save_attachment,
        load_attachment, and delete_attachment

    - error: "Agent doesn't remember conversation"
      cause: "Only current message passed to agent"
      solution: "Build full conversation history before agent run"

    - error: "TypeError: object is not subscriptable"
      cause: "Incorrect type access"
      solution: "Inspect item content structure before indexing"

  frontend_errors:
    - error: "FatalAppError: Invalid input at api"
      cause: "Missing domainKey"
      solution: "Add domainKey: 'localhost' to API config"

    - error: "Unrecognized key 'name'"
      cause: "Wrong prompt schema"
      solution: "Use label instead of name"

    - error: "Unrecognized key 'icon'"
      cause: "Invalid property"
      solution: "Remove icon from prompts"

    - error: "Blank screen / no chat UI"
      cause: "Missing CDN script"
      solution: >
        Add:
        <script src="https://cdn.platform.openai.com/deployments/chatkit/chatkit.js" async></script>

    - error: "History not loading"
      cause: "Thread not persisted"
      solution: "Persist thread ID in localStorage"

    - error: "CORS error"
      cause: "Missing CORS middleware"
      solution: "Add CORSMiddleware to FastAPI app"

diagnostic_scripts:
  check_all_imports:
    description: "Verify all ChatKit and Agents imports work"
    script: |
      try:
          from chatkit.server import ChatKitServer, StreamingResult
          print("✓ chatkit.server imports OK")
      except ImportError as e:
          print(f"✗ chatkit.server: {e}")

      try:
          from chatkit.store import Store
          print("✓ chatkit.store imports OK")
      except ImportError as e:
          print(f"✗ chatkit.store: {e}")

      try:
          from chatkit.types import ThreadMetadata, ThreadItem, Page
          print("✓ chatkit.types imports OK")
      except ImportError as e:
          print(f"✗ chatkit.types: {e}")

      try:
          from chatkit.agents import AgentContext, stream_agent_response
          print("✓ chatkit.agents imports OK")
      except ImportError as e:
          print(f"✗ chatkit.agents: {e}")

      try:
          from agents import Agent, Runner
          from agents.extensions.models.litellm_model import LitellmModel
          print("✓ agents imports OK")
      except ImportError as e:
          print(f"✗ agents: {e}")

  check_store_implementation:
    description: "Verify all abstract Store methods are implemented"
    script: |
      import inspect
      from chatkit.store import Store

      required_methods = [
          'generate_thread_id',
          'generate_item_id',
          'load_thread',
          'save_thread',
          'load_thread_items',
          'add_thread_item',
          'save_item',
          'load_item',
          'delete_thread_item',
          'load_threads',
          'delete_thread',
          'save_attachment',
          'load_attachment',
          'delete_attachment',
      ]

      from your_module import YourStore

      for method in required_methods:
          if hasattr(YourStore, method):
              print(f"✓ {method}")
          else:
              print(f"✗ {method} - MISSING!")

  test_conversation_memory:
    description: "Debug conversation history loading"
    script: |
      async def _build_conversation_history(self, thread, current_input, context):
          page = await self.store.load_thread_items(thread.id, None, 100, "asc", context)

          print("\n=== CONVERSATION DEBUG ===")
          print(f"Thread: {thread.id}")
          print(f"Items in store: {len(page.data)}")

          for i, item in enumerate(page.data):
              item_type = type(item).__name__
              text = self._extract_text(item)
              preview = text[:50] + "..." if len(text) > 50 else text
              print(f"  {i+1}. [{item_type}] {preview}")

          print("=========================\n")

network_debugging:
  browser_fetch_logger:
    description: "Log all fetch requests and responses"
    script: |
      const originalFetch = window.fetch;
      window.fetch = async (...args) => {
        console.log('Fetch:', args[0]);
        const response = await originalFetch(...args);
        console.log(
          'Response:',
          response.status,
          response.headers.get('content-type')
        );
        return response;
      };

  expected_response_types:
    streaming: "text/event-stream"
    json: "application/json"

common_fixes_summary:
  backend:
    - "Use chatkit.store, not chatkit.stores"
    - "Use chatkit.types, not chatkit.models"
    - "Implement all 14 Store methods"
    - "Build full conversation history for the agent"

  frontend:
    - "Add ChatKit CDN script to index.html"
    - "Add domainKey: 'localhost' to config"
    - "Use label instead of name in prompts"
    - "Remove icon property from prompts"
    - "Persist thread ID in localStorage"

related_skills:
  - chatkit-backend
  - chatkit-frontend
  - chatkit-store
  - chatkit-agent-memory
