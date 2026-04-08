"""
discovery/evaluator.py — Epistemic Anchor (Zero-Trust Verification)

Responsibilities:
  - Score branches on four HRDE dimensions:
      * Novelty       — is the approach unusual?
      * Significance  — do results show meaningful improvement?
      * Surprise      — are the findings unexpected? (ToM "Surprise" metric)
      * Rigorous      — is the methodology sound?
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

    Scores branches on Novelty, Significance, Surprise, and Rigorous;
    then verifies manuscript claims against execution logs.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._verification_log: List[Dict[str, Any]] = []
        self._use_llm: bool = config.get("llm", {}).get("scoring_enabled", True)

    # ------------------------------------------------------------------
    # Branch scoring
    # ------------------------------------------------------------------

    def score(self, branch: "ResearchBranch") -> Dict[str, float]:
        """
        Score a branch on four HRDE dimensions.

        Dimensions
        ----------
        novelty      (0–1): is the approach unusual relative to mainstream?
        significance (0–1): do the results show meaningful improvement?
        surprise     (0–1): are the findings unexpected? (proposal "Surprise" metric)
        rigorous     (0–1): is the methodology sound? (proposal "Rigorous" metric)

        Falls back to heuristics when LLM is unavailable.
        """
        result = branch.result or {}
        output = result.get("output", "")
        success = result.get("success", False)

        if not success:
            return {
                "novelty": 0.0, "significance": 0.0,
                "surprise": 0.0, "rigorous": 0.0,
            }

        heuristics = {
            "novelty":      self._heuristic_novelty(branch.hypothesis),
            "significance": self._heuristic_significance(output),
            "surprise":     self._heuristic_surprise(branch.hypothesis, output),
            "rigorous":     self._heuristic_rigorous(output),
        }

        if self._use_llm:
            try:
                from autoresearch_v2.env.tools import LLMTool

                llm = LLMTool(self.config)
                llm_scores = self._llm_score(llm, branch.hypothesis, output)
                # Merge: LLM scores override heuristics where present
                return {k: llm_scores.get(k, heuristics[k]) for k in heuristics}
            except Exception as exc:
                logger.warning("[Evaluator] LLM scoring failed (%s) — using heuristics.", exc)

        return heuristics

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

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
        """Crude novelty estimate based on keyword rarity in hypothesis text."""
        novel_keywords = [
            "novel", "first", "unexplored", "anomaly", "surprising",
            "unprecedented", "paradigm", "gap", "crossover", "reinvent",
        ]
        text_lower = hypothesis.lower()
        matches = sum(1 for kw in novel_keywords if kw in text_lower)
        return min(0.3 + matches * 0.1, 1.0)

    @staticmethod
    def _heuristic_surprise(hypothesis: str, output: str) -> float:
        """
        Estimate the ToM 'Surprise' metric.

        A result is surprising when the outcome contradicts the usual expectation:
        - Standard techniques unexpectedly fail (negative result with explanation).
        - An unusual combination produces outsized gains.
        """
        score = 0.4
        surprise_signals = [
            "unexpected", "surprising", "counter-intuitive", "contrary",
            "against", "outperform", "exceed", "anomaly", "unusual",
        ]
        text = (hypothesis + " " + output).lower()
        matches = sum(1 for kw in surprise_signals if kw in text)
        score += matches * 0.1
        # A very high accuracy score from an unconventional approach is surprising
        if re.search(r"accuracy[:=\s]+0\.9[5-9]", output, re.I):
            score += 0.1
        return min(score, 1.0)

    @staticmethod
    def _heuristic_rigorous(output: str) -> float:
        """
        Estimate the ToM 'Rigorous' metric.

        Rigour signals: reproducible seeds, multiple runs, confidence intervals,
        ablations, baselines, and absence of runtime errors.
        """
        score = 0.4
        rigour_signals = [
            "seed", "reproducible", "ablation", "baseline", "confidence",
            "interval", "standard deviation", "std", "n_trials", "trials",
        ]
        text_lower = output.lower()
        matches = sum(1 for kw in rigour_signals if kw in text_lower)
        score += matches * 0.08
        if "error" in text_lower or "traceback" in text_lower:
            score -= 0.3
        return min(max(score, 0.0), 1.0)

    def _llm_score(self, llm: Any, hypothesis: str, output: str) -> Dict[str, float]:
        prompt = (
            "You are a rigorous scientific evaluator. "
            "Score the following research branch on four dimensions (0.0–1.0 each):\n\n"
            f"Hypothesis: {hypothesis}\n"
            f"Experiment output (excerpt):\n{output[:500]}\n\n"
            "Respond ONLY with four lines:\n"
            "novelty: <float>\n"
            "significance: <float>\n"
            "surprise: <float>\n"
            "rigorous: <float>"
        )
        raw = llm.complete(prompt)
        scores: Dict[str, float] = {}
        for line in raw.splitlines():
            m = re.match(
                r"(novelty|significance|surprise|rigorous)\s*:\s*([01]?\.\d+)",
                line, re.I,
            )
            if m:
                try:
                    scores[m.group(1).lower()] = float(m.group(2))
                except ValueError:
                    pass
        return scores

    @staticmethod
    def _parse_score_response(raw: str) -> Dict[str, float]:
        """Parse a multi-line score response into a float dict (values clamped to [0,1])."""
        scores: Dict[str, float] = {}
        for line in raw.splitlines():
            m = re.match(
                r"(novelty|significance|surprise|rigorous)\s*:\s*"
                r"([01](?:\.\d+)?|\.\d+)",   # 0–1 inclusive, e.g. 0.72, 1, .5
                line, re.I,
            )
            if m:
                try:
                    scores[m.group(1).lower()] = min(1.0, max(0.0, float(m.group(2))))
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

        Returns
        -------
        A dict with:
          - verified: List[str]    — claims verified OK
          - unverified: List[str]  — numbers not found in any log
          - pass_rate: float       — fraction verified
          - total_claims: int
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
            pass_rate * 100, len(verified), total,
        )
        return report

    @staticmethod
    def _extract_numbers_from_latex(latex: str) -> List[str]:
        """Extract numerical claims (decimals / percentages) from LaTeX body text."""
        body = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", latex)
        body = re.sub(r"\\[a-zA-Z]+", " ", body)
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
        Return True when the result meets the significance bar.

        Also warns when effect size is tiny despite low p-value (p-hacking guard).
        """
        if p_value is None:
            return False
        if p_value >= threshold:
            return False
        if effect_size is not None and abs(effect_size) < 0.1:
            logger.warning(
                "[Evaluator] Small effect size (%.3f) despite p=%.4f — "
                "possible p-hacking.",
                effect_size, p_value,
            )
        return True

