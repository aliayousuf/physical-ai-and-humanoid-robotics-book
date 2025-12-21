
skill:
  name: "ChatKit Store Implementation"
  metadata:
    id: chatkit-store
    category: component
    tags:
      - chatkit
      - store
      - persistence
      - memory
    description: >
      Generates a complete ChatKit Store implementation with all 14 required
      abstract methods. Supports in-memory storage by default and can be
      extended for Redis or database-backed persistence.

inputs:
  type:
    type: string
    required: false
    default: memory
    description: "Storage type: memory, redis, postgres"
  context_type:
    type: string
    required: false
    default: dict
    description: "Generic request context type"

outputs:
  description: "Store class implementation with all required abstract methods"

critical_knowledge:
  required_imports:
    correct:
      - "from chatkit.store import Store"
      - "from chatkit.types import ThreadMetadata, ThreadItem, Page"
    incorrect:
      - "from chatkit.stores import Store"
      - "from chatkit.models import ..."
      - "from chatkit.types import FilePart"
      - "from chatkit.types import ClientToolCallOutputItem"

required_methods:
  description: "Every Store subclass MUST implement ALL of the following"
  methods:
    - "generate_thread_id(context) -> str"
    - "generate_item_id(item_type, thread, context) -> str"
    - "load_thread(thread_id, context) -> ThreadMetadata"
    - "save_thread(thread, context) -> None"
    - "load_thread_items(thread_id, after, limit, order, context) -> Page[ThreadItem]"
    - "add_thread_item(thread_id, item, context) -> None"
    - "save_item(thread_id, item, context) -> None"
    - "load_item(thread_id, item_id, context) -> ThreadItem"
    - "delete_thread_item(thread_id, item_id, context) -> None"
    - "load_threads(limit, after, order, context) -> Page[ThreadMetadata]"
    - "delete_thread(thread_id, context) -> None"
    - "save_attachment(attachment, context) -> None"
    - "load_attachment(attachment_id, context) -> Any"
    - "delete_attachment(attachment_id, context) -> None"

implementation:
  title: "In-Memory Store Reference Implementation"
  language: python
  template: |
    import uuid
    from datetime import datetime, timezone
    from typing import Any
    from dataclasses import dataclass, field

    from chatkit.store import Store
    from chatkit.types import ThreadMetadata, ThreadItem, Page


    @dataclass
    class ThreadState:
        """Internal state for a thread"""
        thread: ThreadMetadata
        items: list[ThreadItem] = field(default_factory=list)


    class MemoryStore(Store[dict]):
        """Thread-safe in-memory store for ChatKit"""

        def __init__(self) -> None:
            self._threads: dict[str, ThreadState] = {}
            self._attachments: dict[str, Any] = {}

        # ==================== ID Generation ====================

        def generate_thread_id(self, context: dict) -> str:
            return f"thread_{uuid.uuid4().hex[:12]}"

        def generate_item_id(self, item_type: str, thread: ThreadMetadata, context: dict) -> str:
            return f"{item_type}_{uuid.uuid4().hex[:12]}"

        # ==================== Thread Operations ====================

        async def load_thread(self, thread_id: str, context: dict) -> ThreadMetadata:
            state = self._threads.get(thread_id)
            if state:
                return state.thread.model_copy(deep=True)

            thread = ThreadMetadata(
                id=thread_id,
                created_at=datetime.now(timezone.utc),
                metadata={}
            )
            self._threads[thread_id] = ThreadState(
                thread=thread.model_copy(deep=True),
                items=[]
            )
            return thread

        async def save_thread(self, thread: ThreadMetadata, context: dict) -> None:
            state = self._threads.get(thread.id)
            if state:
                state.thread = thread.model_copy(deep=True)
            else:
                self._threads[thread.id] = ThreadState(
                    thread=thread.model_copy(deep=True),
                    items=[]
                )

        async def load_threads(self, limit: int, after: str | None, order: str, context: dict) -> Page[ThreadMetadata]:
            threads = [s.thread.model_copy(deep=True) for s in self._threads.values()]
            threads.sort(key=lambda t: t.created_at, reverse=(order == "desc"))
            return Page(data=threads[:limit], has_more=len(threads) > limit)

        async def delete_thread(self, thread_id: str, context: dict) -> None:
            self._threads.pop(thread_id, None)

        # ==================== Item Operations ====================

        def _get_items(self, thread_id: str) -> list[ThreadItem]:
            state = self._threads.get(thread_id)
            return state.items if state else []

        async def load_thread_items(
            self,
            thread_id: str,
            after: str | None,
            limit: int,
            order: str,
            context: dict,
        ) -> Page[ThreadItem]:
            items = [item.model_copy(deep=True) for item in self._get_items(thread_id)]
            items.sort(
                key=lambda i: getattr(i, "created_at", datetime.now(timezone.utc)),
                reverse=(order == "desc"),
            )

            start = 0
            if after:
                index_map = {item.id: idx for idx, item in enumerate(items)}
                start = index_map.get(after, -1) + 1

            slice_items = items[start: start + limit + 1]
            has_more = len(slice_items) > limit
            result_items = slice_items[:limit]

            return Page(
                data=result_items,
                has_more=has_more,
                after=slice_items[-1].id if has_more and slice_items else None
            )

        async def add_thread_item(self, thread_id: str, item: ThreadItem, context: dict) -> None:
            state = self._threads.get(thread_id)
            if not state:
                await self.load_thread(thread_id, context)
                state = self._threads[thread_id]

            for i, existing in enumerate(state.items):
                if existing.id == item.id:
                    state.items[i] = item.model_copy(deep=True)
                    return

            state.items.append(item.model_copy(deep=True))

        async def save_item(self, thread_id: str, item: ThreadItem, context: dict) -> None:
            await self.add_thread_item(thread_id, item, context)

        async def load_item(self, thread_id: str, item_id: str, context: dict) -> ThreadItem:
            for item in self._get_items(thread_id):
                if item.id == item_id:
                    return item.model_copy(deep=True)
            raise ValueError(f"Item {item_id} not found")

        async def delete_thread_item(self, thread_id: str, item_id: str, context: dict) -> None:
            state = self._threads.get(thread_id)
            if state:
                state.items = [i for i in state.items if i.id != item_id]

        # ==================== Attachment Operations ====================

        async def save_attachment(self, attachment: Any, context: dict) -> None:
            self._attachments[attachment.id] = attachment

        async def load_attachment(self, attachment_id: str, context: dict) -> Any:
            if attachment_id not in self._attachments:
                raise ValueError(f"Attachment {attachment_id} not found")
            return self._attachments[attachment_id]

        async def delete_attachment(self, attachment_id: str, context: dict) -> None:
            self._attachments.pop(attachment_id, None)

validation:
  checklist:
    - "All 14 Store methods are implemented"
    - "No abstract method errors on instantiation"
    - "Thread items persist within session"
    - "Pagination works correctly"

common_errors:
  - error: "Can't instantiate abstract class"
    cause: "Missing required Store methods"
    fix: "Implement all 14 required methods"
  - error: "chatkit.stores not found"
    cause: "Incorrect module import"
    fix: "Use chatkit.store (singular)"

related_skills:
  - chatkit-backend
  - chatkit-agent-memory
