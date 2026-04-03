"""
agents/base_agent.py — Abstract base class for all ToM-enabled agents.

Every concrete agent inherits from BaseAgent and gains:
  - A typed system_prompt that the HyperKernel can read/write.
  - A run() entry-point.
  - Built-in failure logging (writes to logs/<agent_name>.log).
  - Optional Theory-of-Mind (ToM) introspection helpers.
"""

from __future__ import annotations

import abc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """Abstract base for all AutoResearch v2 agents."""

    #: Override in subclasses with the default persona prompt.
    DEFAULT_SYSTEM_PROMPT: str = "You are a helpful AI research assistant."

    def __init__(
        self,
        config: dict,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        log_dir: str = "./logs",
    ) -> None:
        self.config = config
        self.name = name or self.__class__.__name__
        self.system_prompt: str = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._run_count: int = 0
        self._failure_count: int = 0
        self._execution_log: List[Dict[str, Any]] = []
        self._setup_file_logger()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_file_logger(self) -> None:
        log_path = self._log_dir / f"{self.name.lower()}.log"
        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        self._file_logger = logging.getLogger(f"agent.{self.name}")
        if not self._file_logger.handlers:
            self._file_logger.addHandler(fh)
        self._file_logger.setLevel(logging.DEBUG)

    def log(self, msg: str, level: str = "info") -> None:
        getattr(self._file_logger, level, self._file_logger.info)(
            "[%s] %s", self.name, msg
        )

    # ------------------------------------------------------------------
    # Execution lifecycle
    # ------------------------------------------------------------------

    async def run(self, **kwargs: Any) -> Any:
        """
        Entry point for every agent.

        Wraps _execute() with error counting and telemetry.
        """
        self._run_count += 1
        start = time.time()
        self.log(f"Starting run #{self._run_count} with kwargs={list(kwargs.keys())}")
        try:
            result = await self._execute(**kwargs)
            elapsed = time.time() - start
            self._execution_log.append(
                {"run": self._run_count, "status": "ok", "elapsed": elapsed}
            )
            self.log(f"Run #{self._run_count} completed in {elapsed:.2f}s.")
            return result
        except Exception as exc:
            self._failure_count += 1
            elapsed = time.time() - start
            self._execution_log.append(
                {
                    "run": self._run_count,
                    "status": "error",
                    "error": str(exc),
                    "elapsed": elapsed,
                }
            )
            self.log(
                f"Run #{self._run_count} FAILED ({exc}) after {elapsed:.2f}s.",
                level="error",
            )
            raise

    @abc.abstractmethod
    async def _execute(self, **kwargs: Any) -> Any:
        """Subclasses implement their core logic here."""

    # ------------------------------------------------------------------
    # Theory-of-Mind helpers
    # ------------------------------------------------------------------

    def simulate_reviewer_perspective(self, draft: str, persona: str) -> str:
        """
        A lightweight ToM shim: pre-read the draft through a reviewer persona.

        Returns a critique string.  Full implementation delegates to LLMTool.
        """
        from autoresearch_v2.env.tools import LLMTool

        llm = LLMTool(self.config)
        prompt = (
            f"You are '{persona}'.  Read the following research draft and provide "
            f"a brief, critical review from your perspective.\n\n{draft}"
        )
        try:
            return llm.complete(prompt)
        except Exception as exc:
            self.log(f"ToM simulation failed: {exc}", level="warning")
            return f"[ToM unavailable] Could not simulate '{persona}' perspective."

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def failure_rate(self) -> float:
        if self._run_count == 0:
            return 0.0
        return self._failure_count / self._run_count

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "runs": self._run_count,
            "failures": self._failure_count,
            "failure_rate": round(self.failure_rate, 3),
        }
