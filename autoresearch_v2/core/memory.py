"""
core/memory.py — Vectorized Research Graph

Provides persistent storage for research findings, failed experiments,
and agent lessons using ChromaDB (default) or an in-memory fallback.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResearchMemory:
    """
    Vectorized Research Graph backed by ChromaDB.

    Falls back to a simple JSON file store when ChromaDB is unavailable.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        mem_cfg = config.get("memory", {})
        self.persist_dir = mem_cfg.get("persist_directory", "./research_memory")
        self.collection_name = mem_cfg.get("collection_name", "autoresearch_v2")
        self.embedding_model = mem_cfg.get("embedding_model", "all-MiniLM-L6-v2")
        self._client = None
        self._collection = None
        self._fallback_store: List[Dict[str, Any]] = []
        self._use_chroma = False
        self._init_backend()

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_backend(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
            logger.info(
                "[Memory] ChromaDB backend initialised (collection=%s).",
                self.collection_name,
            )
        except ImportError:
            logger.warning(
                "[Memory] chromadb not installed — using in-memory fallback."
            )
        except Exception as exc:
            logger.warning("[Memory] ChromaDB init failed (%s) — using fallback.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Persist a text fragment with optional metadata. Returns the doc ID."""
        doc_id = doc_id or str(uuid.uuid4())
        meta = metadata or {}
        meta.setdefault("timestamp", time.time())

        if self._use_chroma and self._collection is not None:
            self._collection.add(
                documents=[text],
                metadatas=[meta],
                ids=[doc_id],
            )
        else:
            self._fallback_store.append(
                {"id": doc_id, "text": text, "metadata": meta}
            )
        logger.debug("[Memory] Stored doc %s.", doc_id)
        return doc_id

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant stored documents.

        Returns a list of dicts with keys 'id', 'text', 'metadata', 'distance'.
        """
        if self._use_chroma and self._collection is not None:
            kwargs: Dict[str, Any] = {
                "query_texts": [query_text],
                "n_results": min(n_results, max(1, self._collection.count())),
            }
            if where:
                kwargs["where"] = where
            results = self._collection.query(**kwargs)
            docs = []
            for i, doc_id in enumerate(results["ids"][0]):
                docs.append(
                    {
                        "id": doc_id,
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                        if results.get("distances")
                        else None,
                    }
                )
            return docs

        # Fallback: naive substring / keyword match with optional metadata filter
        q_lower = query_text.lower()
        scored = []
        for entry in self._fallback_store:
            if not self._fallback_where_match(entry["metadata"], where):
                continue
            score = sum(
                1 for token in q_lower.split() if token in entry["text"].lower()
            )
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": e["id"],
                "text": e["text"],
                "metadata": e["metadata"],
                "distance": None,
            }
            for _, e in scored[:n_results]
        ]

    @staticmethod
    def _fallback_where_match(
        metadata: Dict[str, Any], where: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Evaluate a ChromaDB-style *where* filter against *metadata*.

        Supports the ``{"field": {"$eq": value}}`` form used internally.
        Returns True when *where* is None (no filter).
        """
        if not where:
            return True
        for field, condition in where.items():
            actual = metadata.get(field)
            if isinstance(condition, dict):
                op, expected = next(iter(condition.items()))
                if op == "$eq" and actual != expected:
                    return False
                if op == "$ne" and actual == expected:
                    return False
            else:
                # Plain equality shorthand
                if actual != condition:
                    return False
        return True

    def store_skill(self, skill_name: str, description: str, source_run: str) -> str:
        """Persist a reusable skill (cross-run learning)."""
        return self.store(
            text=description,
            metadata={
                "type": "skill",
                "name": skill_name,
                "source_run": source_run,
            },
        )

    def get_skills(self, topic: str, n: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the most relevant skills for a topic."""
        return self.query(topic, n_results=n, where={"type": {"$eq": "skill"}})

    def store_failure(self, agent: str, context: str, error: str) -> str:
        """Record a failed experiment so the system avoids repeating it."""
        return self.store(
            text=f"Agent: {agent}\nContext: {context}\nError: {error}",
            metadata={"type": "failure", "agent": agent},
        )

    def count(self) -> int:
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._fallback_store)

    def save_fallback(self, path: str = "./research_memory/fallback.json") -> None:
        """Persist the fallback store to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._fallback_store, f, indent=2)

    def load_fallback(self, path: str = "./research_memory/fallback.json") -> None:
        """Load the fallback store from disk."""
        p = Path(path)
        if p.exists():
            with open(p) as f:
                self._fallback_store = json.load(f)
