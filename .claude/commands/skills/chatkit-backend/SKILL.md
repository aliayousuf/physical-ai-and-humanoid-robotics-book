

skill:
  name: "ChatKit Backend Setup"
  metadata:
    id: chatkit-backend
    category: setup
    tags:
      - chatkit
      - backend
      - python
      - fastapi
      - gemini
      - litellm
    description: >
      Creates a production-ready ChatKit Python backend using FastAPI,
      OpenAI Agents SDK, and LiteLLM for multi-provider AI support.
      Includes a critical fix for the LiteLLM/Gemini message ID collision issue.

inputs:
  provider:
    type: string
    required: false
    default: gemini
    description: "AI provider: gemini, openai, anthropic"
  model:
    type: string
    required: false
    default: auto
    description: "Specific model ID"
  port:
    type: number
    required: false
    default: 8000
    description: "Server port"

outputs:
  files:
    - path: backend/main.py
      description: "FastAPI server with ChatKit"
    - path: backend/requirements.txt
      description: "Python dependencies"
    - path: .env.example
      description: "Environment variable template"

prerequisites:
  - Python 3.10+
  - pip

execution_steps:
  - step: "Create directory structure"
    structure: |
      backend/
      ├── main.py
      ├── requirements.txt
      └── .venv/

  - step: "Generate requirements.txt"
    file: backend/requirements.txt
    contents: |
      fastapi==0.115.6
      uvicorn[standard]==0.32.1
      openai-chatkit<=1.4.0
      openai-agents[litellm]>=0.6.2
      python-dotenv==1.0.1

  - step: "Generate main.py"
    notes:
      - "CRITICAL: Import ThreadItemConverter for proper history handling"
    file: backend/main.py
    template: |
      import os
      import uuid
      from pathlib import Path
      from datetime import datetime, timezone
      from typing import Any, AsyncIterator
      from dataclasses import dataclass, field

      from dotenv import load_dotenv
      from fastapi import FastAPI, Request
      from fastapi.middleware.cors import CORSMiddleware
      from fastapi.responses import Response, StreamingResponse
      from fastapi.staticfiles import StaticFiles

      from agents import Agent, Runner
      from agents.extensions.models.litellm_model import LitellmModel

      from chatkit.server import ChatKitServer, StreamingResult
      from chatkit.store import Store
      from chatkit.types import ThreadMetadata, ThreadItem, Page
      from chatkit.agents import AgentContext, stream_agent_response, ThreadItemConverter

      ROOT_DIR = Path(__file__).parent.parent
      load_dotenv(ROOT_DIR / ".env")

      # [INSERT STORE IMPLEMENTATION - use chatkit-store skill]
      # [INSERT MODEL CONFIGURATION - based on provider param]
      # [INSERT SERVER IMPLEMENTATION - use chatkit-agent-memory skill]

      app = FastAPI(title="ChatKit Server")
      app.add_middleware(
          CORSMiddleware,
          allow_origins=["*"],
          allow_credentials=True,
          allow_methods=["*"],
          allow_headers=["*"],
      )

      store = MemoryStore()
      server = ChatKitServerImpl(store)

      @app.post("/chatkit")
      async def chatkit_endpoint(request: Request):
          result = await server.process(await request.body(), {})
          if isinstance(result, StreamingResult):
              return StreamingResponse(result, media_type="text/event-stream")
          return Response(content=result.json, media_type="application/json")

      @app.get("/health")
      async def health():
          return {"status": "ok"}

      @app.get("/debug/threads")
      async def debug_threads():
          result = {}
          for thread_id, state in store._threads.items():
              items = []
              for item in state.items:
                  item_data = {"id": item.id, "type": type(item).__name__}
                  if hasattr(item, "content") and item.content:
                      content_parts = []
                      for part in item.content:
                          if hasattr(part, "text"):
                              content_parts.append(part.text)
                      item_data["content"] = content_parts
                  items.append(item_data)
              result[thread_id] = {"items": items, "count": len(items)}
          return result

      if __name__ == "__main__":
          import uvicorn
          uvicorn.run(app, host="0.0.0.0", port={{PORT}})

model_configuration:
  gemini:
    code: |
      model = LitellmModel(
          model="gemini/gemini-2.0-flash",
          api_key=os.getenv("GEMINI_API_KEY"),
      )
  openai:
    code: |
      model = LitellmModel(
          model="openai/gpt-4o",
          api_key=os.getenv("OPENAI_API_KEY"),
      )
  anthropic:
    code: |
      model = LitellmModel(
          model="anthropic/claude-3-sonnet-20240229",
          api_key=os.getenv("ANTHROPIC_API_KEY"),
      )

critical_fix:
  name: "LiteLLM/Gemini ID Collision Fix"
  problem: >
    stream_agent_response reuses provider-generated message IDs when using
    LiteLLM with non-OpenAI providers, causing message overwrites.
  solution: "Map provider IDs to unique store-generated IDs"
  implementation: |
    async def respond(self, thread, input, context):
        from chatkit.types import (
            ThreadItemAddedEvent, ThreadItemDoneEvent,
            ThreadItemUpdatedEvent, AssistantMessageItem
        )

        id_mapping: dict[str, str] = {}

        async for event in stream_agent_response(agent_context, result):
            if event.type == "thread.item.added":
                if isinstance(event.item, AssistantMessageItem):
                    old_id = event.item.id
                    if old_id not in id_mapping:
                        new_id = self.store.generate_item_id("message", thread, context)
                        id_mapping[old_id] = new_id
                    event.item.id = id_mapping[old_id]
            elif event.type == "thread.item.done":
                if isinstance(event.item, AssistantMessageItem):
                    old_id = event.item.id
                    if old_id in id_mapping:
                        event.item.id = id_mapping[old_id]
            elif event.type == "thread.item.updated":
                if event.item_id in id_mapping:
                    event.item_id = id_mapping[event.item_id]

            yield event

validation:
  checklist:
    - Backend starts without import errors
    - "/health endpoint returns HTTP 200"
    - "/chatkit endpoint accepts POST requests"
    - Messages do not overwrite each other
    - "/debug/threads shows unique IDs"
    - Agent remembers conversation context

related_skills:
  - chatkit-store
  - chatkit-agent-memory
  - chatkit-frontend
