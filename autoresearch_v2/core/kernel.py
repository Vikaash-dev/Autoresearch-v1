"""
core/kernel.py — DGM-H Hyper-Kernel (Self-Modification Logic)

The Hyper-Kernel is the "Brain of Brains." It:
  1. Monitors sub-agent logs for Reasoning Loops (repeated failures).
  2. Uses an LLM to patch the offending agent's system prompt.
  3. Exposes read/write access to sub-agent prompts/tool configs.
  4. Runs a background monitoring thread via start() / stop().
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import yaml

if TYPE_CHECKING:
    from autoresearch_v2.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class HyperKernel:
    """DGM-H Hyper-Kernel: monitors agents and patches their prompts."""

    def __init__(self, config: dict, log_dir: str = "./logs") -> None:
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._agent_registry: Dict[str, "BaseAgent"] = {}
        self._failure_counts: Dict[str, int] = {}
        self._patch_history: list[dict] = []
        self._check_interval: int = config.get("compute", {}).get(
            "kernel_check_interval", 5
        )
        self._max_retries: int = config.get("compute", {}).get("max_debug_retries", 3)
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        # Track byte offsets already read per log file to avoid re-counting.
        self._log_offsets: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Agent registry
    # ------------------------------------------------------------------

    def register_agent(self, name: str, agent: "BaseAgent") -> None:
        """Register a sub-agent so the kernel can monitor and patch it."""
        self._agent_registry[name] = agent
        self._failure_counts[name] = 0
        logger.info("[HyperKernel] Registered agent: %s", name)

    def get_agent(self, name: str) -> Optional["BaseAgent"]:
        return self._agent_registry.get(name)

    # ------------------------------------------------------------------
    # Prompt read / write (self-referential codebase access)
    # ------------------------------------------------------------------

    def read_agent_prompt(self, name: str) -> Optional[str]:
        """Return the current system prompt of a registered agent."""
        agent = self._agent_registry.get(name)
        if agent is None:
            logger.warning("[HyperKernel] Agent '%s' not found.", name)
            return None
        return agent.system_prompt

    def write_agent_prompt(self, name: str, new_prompt: str) -> bool:
        """Overwrite the system prompt of a registered agent."""
        agent = self._agent_registry.get(name)
        if agent is None:
            logger.warning("[HyperKernel] Cannot patch unknown agent '%s'.", name)
            return False
        old_prompt = agent.system_prompt
        agent.system_prompt = new_prompt
        self._patch_history.append(
            {
                "agent": name,
                "timestamp": time.time(),
                "old_prompt": old_prompt,
                "new_prompt": new_prompt,
            }
        )
        logger.info("[HyperKernel] Patched system prompt for agent '%s'.", name)
        return True

    # ------------------------------------------------------------------
    # Failure detection
    # ------------------------------------------------------------------

    def record_failure(self, agent_name: str) -> None:
        """Increment the failure counter for an agent."""
        self._failure_counts[agent_name] = (
            self._failure_counts.get(agent_name, 0) + 1
        )

    def record_success(self, agent_name: str) -> None:
        """Reset the failure counter after a successful operation."""
        self._failure_counts[agent_name] = 0

    def _detect_reasoning_loop(self, agent_name: str) -> bool:
        """Return True when the agent has failed > max_retries times.

        Using strict greater-than means the agent gets exactly max_retries
        attempts before the kernel intervenes (consistent with the name).
        """
        return self._failure_counts.get(agent_name, 0) > self._max_retries

    # ------------------------------------------------------------------
    # Patch generation
    # ------------------------------------------------------------------

    def _generate_patch(self, agent_name: str, failure_summary: str) -> str:
        """
        Use the config-specified LLM to generate an improved system prompt.

        Falls back to a heuristic patch when no LLM is available.
        """
        current_prompt = self.read_agent_prompt(agent_name) or ""
        try:
            from autoresearch_v2.env.tools import LLMTool

            llm = LLMTool(self.config)
            patch_instruction = (
                f"You are a Prompt Engineer. The following agent has been failing "
                f"repeatedly.\n\nAgent: {agent_name}\n"
                f"Failure summary: {failure_summary}\n\n"
                f"Current system prompt:\n{current_prompt}\n\n"
                "Rewrite the system prompt to address the root cause of the failures "
                "while preserving the original intent. Output ONLY the new prompt."
            )
            new_prompt = llm.complete(patch_instruction)
            logger.info(
                "[HyperKernel] LLM-generated patch for '%s'.", agent_name
            )
            return new_prompt
        except Exception as exc:
            logger.warning(
                "[HyperKernel] LLM patch generation failed (%s); using heuristic.", exc
            )
            return (
                current_prompt
                + "\n\n[KERNEL PATCH] If a previous approach has failed multiple times, "
                "try an alternative strategy. Be concise, concrete, and avoid loops."
            )

    def reflect_and_patch(self, agent_name: str, failure_summary: str) -> bool:
        """
        Run one reflection-and-patch cycle for an agent in a Reasoning Loop.

        Returns True if a patch was applied.
        """
        if not self._detect_reasoning_loop(agent_name):
            return False
        logger.warning(
            "[HyperKernel] Reasoning loop detected for '%s' — patching.", agent_name
        )
        new_prompt = self._generate_patch(agent_name, failure_summary)
        patched = self.write_agent_prompt(agent_name, new_prompt)
        if patched:
            self._failure_counts[agent_name] = 0
        return patched

    # ------------------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------------------

    def _scan_log_file(self, log_path: Path) -> Dict[str, int]:
        """Count new error/failure lines in *log_path* since last scan."""
        agent_errors: Dict[str, int] = {}
        key = str(log_path)
        offset = self._log_offsets.get(key, 0)
        try:
            with log_path.open("rb") as fh:
                fh.seek(offset)
                new_bytes = fh.read()
                self._log_offsets[key] = offset + len(new_bytes)
            text = new_bytes.decode("utf-8", errors="replace")
            for line in text.splitlines():
                m = re.search(
                    r"\[(coder|writer|reviewer|ideator)\].*?(error|fail)", line, re.I
                )
                if m:
                    agent_key = m.group(1).lower()
                    agent_errors[agent_key] = agent_errors.get(agent_key, 0) + 1
        except OSError:
            pass
        return agent_errors

    def monitor_once(self) -> None:
        """Single monitoring pass: scan *new* log lines and patch agents in loops."""
        for log_file in self.log_dir.glob("*.log"):
            errors = self._scan_log_file(log_file)
            for agent_name, count in errors.items():
                self._failure_counts[agent_name] = (
                    self._failure_counts.get(agent_name, 0) + count
                )
                if self._detect_reasoning_loop(agent_name):
                    self.reflect_and_patch(
                        agent_name,
                        failure_summary=f"Detected {count} new errors in {log_file.name}",
                    )

    def run_monitoring_loop(self) -> None:
        """Blocking monitoring loop — normally called via start()."""
        self._running = True
        logger.info("[HyperKernel] Monitoring loop started.")
        while self._running:
            try:
                self.monitor_once()
            except Exception as exc:
                logger.error("[HyperKernel] Monitor error: %s", exc)
            time.sleep(self._check_interval)

    def start(self) -> None:
        """Launch the monitoring loop in a daemon background thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_thread = threading.Thread(
            target=self.run_monitoring_loop,
            name="HyperKernel-Monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("[HyperKernel] Background monitor thread started.")

    def stop(self) -> None:
        """Signal the monitoring loop to stop and wait for the thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self._check_interval + 2)
        logger.info("[HyperKernel] Stopped.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_patch_history(self, path: str = "./logs/patch_history.yaml") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._patch_history, f)

    @property
    def patch_count(self) -> int:
        return len(self._patch_history)
