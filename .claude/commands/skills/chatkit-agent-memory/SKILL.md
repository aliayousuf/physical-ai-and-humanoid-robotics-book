

skill:
  name: "ChatKit Agent Memory (Conversation History)"
  metadata:
    id: chatkit-agent-memory
    category: component
    tags:
      - chatkit
      - agent
      - memory
      - conversation
      - history
      - litellm
      - gemini
    description: >
      Implements conversation history loading for ChatKit agents with a
      LiteLLM/Gemini ID collision fix. This is CRITICAL—without it, the agent
      only sees the current message and cannot remember previous context.

problem:
  summary: "Conversation memory loss and LiteLLM/Gemini message ID collisions"
  issues:
    - name: No Conversation Memory
      details:
        - ChatKit's simple_to_agent_input(input) only passes the current message
        - Agent forgets user's name
        - Agent loses prior context
        - Agent repeats already-answered questions
    - name: LiteLLM/Gemini ID Collision
      details:
        - stream_agent_response reuses provider message IDs
        - IDs may collide across responses
        - Messages overwrite each other instead of creating new ones

solution:
  overview:
    - Use ThreadItemConverter.to_agent_input() to load full conversation history
    - Map incoming provider message IDs to unique store-generated IDs
  key_components:
    - ThreadItemConverter
    - Store.generate_item_id
    - stream_agent_response event interception

implementation:
  title: "Complete ChatKitServer with Memory AND ID Fix"
  language: python
  code: |
    from typing import Any, AsyncIterator

    from agents import Agent, Runner
    from agents.extensions.models.litellm_model import LitellmModel

    from chatkit.server import ChatKitServer, StreamingResult
    from chatkit.store import Store
    from chatkit.types import (
        ThreadMetadata, ThreadItem,
        ThreadItemAddedEvent, ThreadItemDoneEvent, ThreadItemUpdatedEvent,
        AssistantMessageItem
    )
    from chatkit.agents import AgentContext, stream_agent_response, ThreadItemConverter


    class ChatKitServerWithMemory(ChatKitServer[dict]):
        """ChatKit server with full conversation memory and LiteLLM ID fix"""

        def __init__(self, data_store: Store, model: LitellmModel, instructions: str):
            super().__init__(data_store)

            self.agent = Agent[AgentContext](
                name="Assistant",
                instructions=instructions,
                model=model,
            )
            self.converter = ThreadItemConverter()

        async def respond(
            self,
            thread: ThreadMetadata,
            input: Any,
            context: dict
        ) -> AsyncIterator:
            """Generate response with full conversation context and unique IDs"""

            agent_context = AgentContext(
                thread=thread,
                store=self.store,
                request_context=context,
            )

            page = await self.store.load_thread_items(
                thread.id,
                after=None,
                limit=100,
                order="asc",
                context=context
            )
            all_items = list(page.data)

            if input:
                all_items.append(input)

            agent_input = await self.converter.to_agent_input(all_items) if all_items else []

            result = Runner.run_streamed(
                self.agent,
                agent_input,
                context=agent_context,
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

usage:
  example:
    language: python
    code: |
      from agents.extensions.models.litellm_model import LitellmModel

      model = LitellmModel(
          model="gemini/gemini-2.0-flash",
          api_key=os.getenv("GEMINI_API_KEY"),
      )

      store = MemoryStore()

      server = ChatKitServerWithMemory(
          data_store=store,
          model=model,
          instructions="You are a helpful assistant. Remember what the user tells you."
      )

key_points:
  - Use ThreadItemConverter instead of manual history construction
  - Load messages in ascending chronological order
  - Generate unique IDs to prevent LiteLLM/Gemini collisions
  - Handle added, done, and updated thread item events
  - Limit conversation history (default: 100 items)

id_mapping_rationale:
  explanation: >
    stream_agent_response uses provider-generated IDs, which may be reused
    or non-unique when using LiteLLM with Gemini or Anthropic.
  consequences:
    - Messages overwrite each other
    - Store state becomes corrupted
  fix:
    - Map provider IDs to store-generated unique IDs

testing:
  scenario:
    - user: "My name is Alice"
      assistant: "Nice to meet you, Alice!"
    - user: "What's my name?"
      assistant: "Your name is Alice."
  expected_behavior:
    - Agent remembers prior messages
    - Responses are stored as separate thread items

debug:
  endpoint:
    path: /debug/threads
    description: "Inspect stored thread items and verify unique IDs"
    code: |
      @app.get("/debug/threads")
      async def debug_threads():
          result = {}
          for thread_id, state in store._threads.items():
              items = [{"id": i.id, "type": type(i).__name__} for i in state.items]
              result[thread_id] = {"items": items, "count": len(items)}
          return result

validation:
  checklist:
    - Agent remembers user's name
    - Agent recalls previous topics
    - Messages do not overwrite each other
    - Debug endpoint shows unique message IDs
    - No "Updated existing item" logs for new messages

related_skills:
  - chatkit-store
  - chatkit-backend
  - chatkit-debug
