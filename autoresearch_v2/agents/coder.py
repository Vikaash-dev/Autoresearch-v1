"""
agents/coder.py — Self-Healing Python Coder Agent (Docker-integrated)

Responsibilities:
  - Generate Python experiment code from a hypothesis description.
  - Execute the code in a Docker sandbox (or subprocess fallback).
  - Automatically debug tracebacks up to `max_retries` times.
  - Report numerical results with traceable execution logs.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from autoresearch_v2.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    """Self-healing Python coder with Docker-integrated execution."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert Python research engineer. "
        "Given a hypothesis, write clean, reproducible Python code to test it. "
        "Always include detailed print statements for all numerical results so they "
        "can be traced back to specific lines. "
        "If previous code raised an exception, analyse the traceback carefully and "
        "produce corrected code — do NOT repeat the same mistake."
    )

    def __init__(self, config: dict, **kwargs: Any) -> None:
        super().__init__(config, name="CoderAgent", **kwargs)
        self._max_retries: int = config.get("compute", {}).get("max_debug_retries", 3)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def _execute(
        self,
        hypothesis: str,
        branch_id: Optional[str] = None,
        previous_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate and execute experiment code for *hypothesis*.

        Returns a dict with keys:
          - code: str       — the generated Python source
          - output: str     — stdout / stderr from execution
          - success: bool
          - retries: int
        """
        from autoresearch_v2.env.sandbox import DockerSandbox
        from autoresearch_v2.env.tools import LLMTool

        llm = LLMTool(self.config)
        sandbox = DockerSandbox(self.config)

        code: str = ""
        output: str = ""
        retries = 0
        last_error: Optional[str] = previous_error

        while retries <= self._max_retries:
            # 1. Generate code
            code = self._generate_code(llm, hypothesis, last_error)
            self.log(f"Generated code (attempt {retries + 1}):\n{code[:200]}...")

            # 2. Execute in sandbox
            exec_result = sandbox.run(code, timeout=self.config.get("compute", {}).get(
                "sandbox_timeout", 120
            ))
            output = exec_result.get("output", "")
            success = exec_result.get("success", False)

            if success:
                self.log("Code executed successfully.")
                return {
                    "code": code,
                    "output": output,
                    "success": True,
                    "retries": retries,
                    "branch_id": branch_id,
                }

            # 3. Self-healing: analyse error and retry
            last_error = exec_result.get("error", output)
            self.log(f"Execution failed (attempt {retries + 1}): {last_error[:200]}", level="error")
            retries += 1

        # All retries exhausted
        self.log("All retries exhausted — returning failure result.", level="error")
        return {
            "code": code,
            "output": output,
            "success": False,
            "retries": retries,
            "branch_id": branch_id,
            "last_error": last_error,
        }

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def _generate_code(
        self,
        llm: Any,
        hypothesis: str,
        previous_error: Optional[str] = None,
    ) -> str:
        prompt = self.system_prompt + "\n\n"
        prompt += f"Hypothesis to test:\n{hypothesis}\n\n"
        if previous_error:
            prompt += (
                f"Previous execution failed with the following error. "
                f"Fix it in the new version:\n{previous_error}\n\n"
            )
        prompt += (
            "Write a complete, runnable Python script that tests the hypothesis. "
            "Use only the standard library and common scientific packages "
            "(numpy, scipy, sklearn). "
            "Print all key numerical results with labels, e.g.: "
            "'print(f\"accuracy: {acc:.4f}\")'.\n"
            "Output ONLY the Python code — no markdown fences."
        )
        raw = llm.complete(prompt)
        return self._strip_code_fences(raw)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences if the LLM included them."""
        text = re.sub(r"^```(?:python)?\n?", "", text.strip())
        text = re.sub(r"\n?```$", "", text.strip())
        return text.strip()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def extract_metrics(self, output: str) -> Dict[str, float]:
        """
        Parse key=value pairs from execution output.

        Supports lines like: "accuracy: 0.9523" or "loss=0.0341"
        """
        metrics: Dict[str, float] = {}
        for line in output.splitlines():
            m = re.search(r"([\w_]+)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)", line)
            if m:
                try:
                    metrics[m.group(1).lower()] = float(m.group(2))
                except ValueError:
                    pass
        return metrics
