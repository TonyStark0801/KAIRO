"""ChromaDB vector memory client — subscribes to MemoryWriteEvent."""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config.models import MemoryConfig
    from memory.vector.embedder import Embedder
    from runtime.event_bus import EventBus

logger = logging.getLogger(__name__)


class VectorMemoryClient:
    def __init__(self, config: MemoryConfig, embedder: Embedder, event_bus: EventBus) -> None:
        self._config = config
        self._embedder = embedder
        self._bus = event_bus
        self._collection = None
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def initialize(self) -> bool:
        try:
            import chromadb
            from pathlib import Path
            path = Path(self._config.chroma_path)
            path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(path))
            self._collection = client.get_or_create_collection("kairo_commands")
            self._healthy = True
            return True
        except Exception:
            logger.exception("Failed to initialize ChromaDB")
            return False

    async def on_memory_write(self, event) -> None:
        if not self._healthy or self._collection is None:
            return
        try:
            embedding = await self._embedder.embed(event.command_text)
            if embedding is None:
                return
            doc_id = f"{event.session_id}_{event.timestamp}"
            self._collection.upsert(
                ids=[doc_id], embeddings=[embedding], documents=[event.command_text],
                metadatas=[{"tool_name": event.tool_name, "timestamp": str(event.timestamp), "session_id": event.session_id}],
            )
        except Exception:
            logger.exception("Vector memory write failed")

    async def search(self, query: str, n_results: int = 5) -> list[dict]:
        if not self._healthy or self._collection is None:
            return []
        try:
            embedding = await self._embedder.embed(query)
            if embedding is None:
                return []
            results = self._collection.query(query_embeddings=[embedding], n_results=n_results)
            return [{"document": doc, "metadata": meta} for doc, meta in zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0])]
        except Exception:
            logger.exception("Vector search failed")
            return []
