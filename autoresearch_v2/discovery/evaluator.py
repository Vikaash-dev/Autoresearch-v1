"""
discovery/evaluator.py — Epistemic Anchor (Zero-Trust Verification)

Responsibilities:
  - Score branches for "Novelty" and "Result Significance."
  - Verify every number in the final LaTeX paper is traceable to a
    specific line in the sandbox execution logs (Zero-Trust).
  - Flag potential hallucinations and anomalous statistical claims.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from autoresearch_v2.discovery.tree_search import ResearchBranch

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Zero-Trust Epistemic Anchor.

    Scores branches and verifies manuscript claims against execution logs.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._verification_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Branch scoring
    # ------------------------------------------------------------------

    def score(self, branch: "ResearchBranch") -> Dict[str, float]:
        """
        Return novelty and significance scores for a branch.

        Scores are in [0.0, 1.0].  When no LLM is available, heuristics
        derived from execution output are used instead.
        """
        result = branch.result or {}
        output = result.get("output", "")
        success = result.get("success", False)

        if not success:
            return {"novelty": 0.0, "significance": 0.0}

        significance = self._heuristic_significance(output)
        novelty = self._heuristic_novelty(branch.hypothesis)

        # Optionally enhance scores with LLM
        try:
            from autoresearch_v2.env.tools import LLMTool

            llm = LLMTool(self.config)
            scores = self._llm_score(llm, branch.hypothesis, output)
            novelty = scores.get("novelty", novelty)
            significance = scores.get("significance", significance)
        except Exception:
            pass  # Gracefully fall back to heuristics

        return {"novelty": novelty, "significance": significance}

    def _heuristic_significance(self, output: str) -> float:
        """Derive significance from keywords and numeric patterns in output."""
        score = 0.3  # baseline
        if re.search(r"p[\s_-]?value\s*[<=>]\s*0\.0[0-4]", output, re.I):
            score += 0.3
        if re.search(r"accuracy[:=\s]+0\.[89]\d", output, re.I):
            score += 0.2
        if re.search(r"(improvement|reduction|gain)\s+of\s+\d+", output, re.I):
            score += 0.1
        if "error" in output.lower() or "exception" in output.lower():
            score -= 0.3
        return min(max(score, 0.0), 1.0)

    def _heuristic_novelty(self, hypothesis: str) -> float:
        """Crude novelty estimate based on keyword rarity."""
        novel_keywords = [
            "novel", "first", "unexplored", "anomaly", "surprising",
            "unprecedented", "paradigm", "gap", "crossover",
        ]
        text_lower = hypothesis.lower()
        matches = sum(1 for kw in novel_keywords if kw in text_lower)
        return min(0.3 + matches * 0.1, 1.0)

    def _llm_score(self, llm: Any, hypothesis: str, output: str) -> Dict[str, float]:
        prompt = (
            "You are a rigorous scientific evaluator. "
            "Given a hypothesis and experiment output, score:\n"
            "1. Novelty (0.0–1.0): How original is the hypothesis relative to mainstream ML?\n"
            "2. Significance (0.0–1.0): How strong and reliable are the results?\n\n"
            f"Hypothesis: {hypothesis}\n\n"
            f"Experiment output (excerpt):\n{output[:500]}\n\n"
            "Respond ONLY with two lines:\n"
            "novelty: <float>\nsignificance: <float>"
        )
        raw = llm.complete(prompt)
        scores: Dict[str, float] = {}
        for line in raw.splitlines():
            m = re.match(r"(novelty|significance)\s*:\s*([01]?\.\d+)", line, re.I)
            if m:
                try:
                    scores[m.group(1).lower()] = float(m.group(2))
                except ValueError:
                    pass
        return scores

    # ------------------------------------------------------------------
    # Zero-Trust verification
    # ------------------------------------------------------------------

    def verify_manuscript(
        self,
        latex_source: str,
        execution_logs: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Verify that every number in the LaTeX manuscript is traceable
        to a line in one of the execution logs.

        Parameters
        ----------
        latex_source:
            Full LaTeX source of the paper.
        execution_logs:
            Mapping of branch_id -> execution output string.

        Returns
        -------
        A dict with:
          - verified: List[str]    — claims verified OK
          - unverified: List[str]  — numbers not found in any log
          - pass_rate: float       — fraction verified
        """
        numbers = self._extract_numbers_from_latex(latex_source)
        log_text = "\n".join(execution_logs.values())

        verified: List[str] = []
        unverified: List[str] = []

        for num_str in numbers:
            if self._number_in_logs(num_str, log_text):
                verified.append(num_str)
            else:
                unverified.append(num_str)
                logger.warning(
                    "[Evaluator] Unverified claim: %s not found in execution logs.",
                    num_str,
                )

        total = len(numbers)
        pass_rate = len(verified) / total if total > 0 else 1.0

        report = {
            "verified": verified,
            "unverified": unverified,
            "pass_rate": round(pass_rate, 3),
            "total_claims": total,
        }
        self._verification_log.append(report)
        logger.info(
            "[Evaluator] Verification complete: %.1f%% pass rate (%d/%d).",
            pass_rate * 100,
            len(verified),
            total,
        )
        return report

    @staticmethod
    def _extract_numbers_from_latex(latex: str) -> List[str]:
        """
        Extract numerical claims (decimals / percentages) from LaTeX text.

        Ignores numbers inside \\usepackage, \\documentclass, etc.
        """
        # Strip LaTeX commands to reduce false positives
        body = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", latex)
        body = re.sub(r"\\[a-zA-Z]+", " ", body)
        # Find decimal numbers and percentages
        numbers = re.findall(r"\b\d+\.\d+(?:%|\\%)?|\b\d{2,}(?:%|\\%)\b", body)
        return list(set(numbers))

    @staticmethod
    def _number_in_logs(num_str: str, log_text: str) -> bool:
        """Check if *num_str* (or its numeric value) appears in *log_text*."""
        clean = num_str.replace("\\%", "").replace("%", "")
        return clean in log_text

    # ------------------------------------------------------------------
    # Statistical significance gate
    # ------------------------------------------------------------------

    def is_statistically_significant(
        self,
        p_value: Optional[float],
        effect_size: Optional[float] = None,
        threshold: float = 0.05,
    ) -> bool:
        """
        Return True when the result meets the statistical significance bar.

        Also warns when effect size is tiny despite low p-value (p-hacking).
        """
        if p_value is None:
            return False
        if p_value >= threshold:
            return False
        if effect_size is not None and abs(effect_size) < 0.1:
            logger.warning(
                "[Evaluator] Small effect size (%.3f) despite p=%.4f — "
                "possible p-hacking.",
                effect_size,
                p_value,
            )
        return True
