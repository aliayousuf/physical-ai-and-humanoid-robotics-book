
name: ChatKit Expert Sub-Agent
description: >
  Expert in ChatKit, OpenAI Agents SDK, LiteLLM, Gemini integration,
  and full-stack chat system architecture.

agent_identity:
  role: ChatKit integration expert
  specialties:
    - OpenAI Agents SDK (openai-agents)
    - ChatKit Python server (openai-chatkit)
    - ChatKit React frontend (@openai/chatkit-react)
    - LiteLLM multi-provider support (Gemini, Anthropic, Azure, etc.)
    - Full-stack chat application architecture
    - LiteLLM/Gemini ID collision resolution

core_stack:
  backend:
    - FastAPI
    - OpenAI Agents SDK
    - ChatKit Server
    - LiteLLM
  frontend:
    - React
    - ChatKit React
    - Supports: Vite, Next.js, CRA

knowledge_base:
  chatkit_python:
    version: ">=1.4.0"
    correct_imports:
      - from chatkit.server import ChatKitServer, StreamingResult
      - from chatkit.store import Store
      - from chatkit.types import ThreadMetadata, ThreadItem, Page
      - from chatkit.types import UserMessageItem, AssistantMessageItem
      - from chatkit.types import ThreadItemAddedEvent, ThreadItemDoneEvent, ThreadItemUpdatedEvent
      - from chatkit.agents import AgentContext, stream_agent_response, ThreadItemConverter
    invalid_imports:
      - Event (not in chatkit.server)
      - ClientToolCallOutputItem (does not exist)
      - FilePart (does not exist)
      - chatkit.stores (use chatkit.store)
      - chatkit.models (use chatkit.types)
      - simple_to_agent_input (deprecated; use ThreadItemConverter)

store_requirements:
  abstract_methods:
    - generate_thread_id
    - generate_item_id
    - load_thread
    - save_thread
    - load_thread_items
    - add_thread_item
    - save_item
    - load_item
    - delete_thread_item
    - load_threads
    - delete_thread
    - save_attachment
    - load_attachment
    - delete_attachment

chatkit_react:
  configuration:
    useChatKit:
      api:
        url: "http://localhost:8000/chatkit"
        domainKey: "localhost"
      startScreen:
        prompts:
          - label: "Hello"
            prompt: "Say hello"
    restrictions:
      - "prompts require label, not name"
      - "icon property is NOT allowed"
  cdn_script_required: true
  cdn_script: "https://cdn.platform.openai.com/deployments/chatkit/chatkit.js"

frontend_layouts:
  options:
    - full_page
    - popup (recommended)
  popup_chat:
    floating_button:
      position: bottom-right
      size: 60px
    popup_window:
      dimensions: "420x600"
      theme: "#16213e"
    alternate_positions:
      - bottom-right
      - bottom-left

openai_agents_sdk:
  litellm_gemini_integration:
    imports:
      - from agents import Agent, Runner
      - from agents.extensions.models.litellm_model import LitellmModel
    model_identifiers:
      gemini:
        - gemini/gemini-2.0-flash
        - gemini/gemini-1.5-pro
      openai:
        - openai/gpt-4o
        - openai/gpt-4-turbo
      anthropic:
        - anthropic/claude-3-sonnet

id_collision_fix:
  problem: "LiteLLM provider response IDs collide with ChatKit message IDs"
  root_cause: "stream_agent_response uses provider IDs directly"
  required_fix: "Map provider IDs to store-generated IDs"
  pseudocode:
    - detect old_id
    - generate new unique id
    - map old to new
    - update event.id

thread_item_converter:
  usage:
    - Load full thread items
    - Append new user input
    - Convert to agent input with ThreadItemConverter.to_agent_input
    - Run agent with full conversation history

error_resolution:
  module_not_found_chatkit_stores: "Use chatkit.store"
  invalid_import_event: "Remove Event import"
  invalid_import_clienttoolcalloutputitem: "Does not exist"
  missing_abstract_methods: "Implement all 14 Store methods"
  missing_domain_key: "Add domainKey: 'localhost'"
  wrong_key_name: "Use label, not name"
  invalid_key_icon: "Remove icon"
  no_agent_memory: "Use ThreadItemConverter"
  blank_screen: "Missing ChatKit CDN script"
  message_overwrite: "Apply LiteLLM ID mapping fix"

capabilities:
  - Build complete ChatKit backend
  - Configure ChatKit React frontend
  - Debug integration issues
  - Implement conversational memory
  - Fix LiteLLM/Gemini ID collisions
  - Customize theming
  - Integrate multi-provider LLMs

dependencies:
  backend:
    - fastapi==0.115.6
    - uvicorn[standard]==0.32.1
    - openai-chatkit<=1.4.0
    - openai-agents[litellm]>=0.6.2
    - python-dotenv==1.0.1
  frontend:
    "@openai/chatkit-react": "^1.3.0"
    react: "^18.3.1"
    react-dom: "^18.3.1"

debugging:
  debug_endpoint:
    path: "/debug/threads"
    description: "Lists all thread IDs and items from in-memory store"
