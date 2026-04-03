"""
agents/reviewer.py — ToM-based Adversarial Reviewer Council

Implements a 3-persona review system:
  - The Skeptic    — finds methodological flaws and p-hacking.
  - The Engineer   — focuses on code reproducibility and efficiency.
  - The Visionary  — evaluates long-term impact and novelty.

Each persona provides a structured critique.  The WriterAgent must
respond to all three before the paper can be "Accepted."
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from autoresearch_v2.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Persona definitions
# ------------------------------------------------------------------

PERSONAS: Dict[str, str] = {
    "skeptic": (
        "You are The Skeptic — a rigorous frequentist statistician and methodologist. "
        "Your goal is to find flaws in methodology, p-hacking, over-claiming, and "
        "insufficient baselines. Be precise, cite specific sections, and demand "
        "concrete improvements. Do NOT approve anything you have doubts about."
    ),
    "engineer": (
        "You are The Engineer — a pragmatic software and systems researcher. "
        "Your goal is to ensure code reproducibility, computational efficiency, and "
        "that every experimental result is traceable to exact code lines. "
        "Flag missing hyperparameters, non-deterministic seeds, and hardware assumptions."
    ),
    "visionary": (
        "You are The Visionary — a cross-disciplinary research strategist. "
        "Your goal is to evaluate the long-term scientific impact, novelty relative to "
        "the state-of-the-art, and whether the contribution opens new research directions. "
        "Identify incremental vs. paradigm-shifting aspects of the work."
    ),
}

VERDICT_ACCEPT = "ACCEPT"
VERDICT_REJECT = "REJECT"
VERDICT_MAJOR_REVISION = "MAJOR_REVISION"
VERDICT_MINOR_REVISION = "MINOR_REVISION"


class ReviewerCouncil(BaseAgent):
    """
    Multi-persona adversarial review council.

    Usage::

        council = ReviewerCouncil(config)
        result = await council.run(manuscript=latex_src, round_num=1)
        # result["accepted"] is True only when >= acceptance_threshold personas approve.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are coordinating a multi-persona peer review. "
        "Each reviewer has a distinct perspective and scoring rubric."
    )

    def __init__(
        self,
        config: dict,
        personas: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, name="ReviewerCouncil", **kwargs)
        raw = config.get("review_council", "skeptic,engineer,visionary")
        self._active_personas: List[str] = personas or [
            p.strip().lower() for p in raw.split(",") if p.strip()
        ]
        self._acceptance_threshold: int = config.get("acceptance_threshold", 2)
        self._review_rounds: int = config.get("review_rounds", 3)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def _execute(
        self,
        manuscript: str,
        round_num: int = 1,
    ) -> Dict[str, Any]:
        """
        Run one review round.

        Returns a dict with:
          - reviews: Dict[persona -> review_text]
          - verdicts: Dict[persona -> verdict]
          - accepted: bool
          - round_num: int
          - consolidated_feedback: List[str]
        """
        from autoresearch_v2.env.tools import LLMTool

        llm = LLMTool(self.config)

        reviews: Dict[str, str] = {}
        verdicts: Dict[str, str] = {}

        for persona_key in self._active_personas:
            persona_prompt = PERSONAS.get(persona_key, PERSONAS["skeptic"])
            review_text = self._review_with_persona(
                llm, persona_prompt, manuscript, round_num
            )
            verdict = self._extract_verdict(review_text)
            reviews[persona_key] = review_text
            verdicts[persona_key] = verdict
            self.log(
                f"[Round {round_num}] {persona_key.title()} verdict: {verdict}"
            )

        approvals = sum(
            1 for v in verdicts.values()
            if v in (VERDICT_ACCEPT, VERDICT_MINOR_REVISION)
        )
        accepted = approvals >= self._acceptance_threshold

        feedback = self._consolidate_feedback(reviews)

        self.log(
            f"[Round {round_num}] Council result: {approvals}/{len(self._active_personas)} "
            f"approvals → {'ACCEPTED' if accepted else 'REJECTED/REVISION'}"
        )

        return {
            "reviews": reviews,
            "verdicts": verdicts,
            "accepted": accepted,
            "approvals": approvals,
            "round_num": round_num,
            "consolidated_feedback": feedback,
        }

    # ------------------------------------------------------------------
    # Review generation
    # ------------------------------------------------------------------

    def _review_with_persona(
        self,
        llm: Any,
        persona_prompt: str,
        manuscript: str,
        round_num: int,
    ) -> str:
        prompt = (
            persona_prompt + "\n\n"
            f"This is review round {round_num}.\n\n"
            "Review the following research manuscript excerpt:\n\n"
            f"{manuscript[:3000]}\n\n"  # Limit to avoid token overflow
            "Provide:\n"
            "1. A concise summary of your major concerns (≤5 bullet points).\n"
            "2. Specific suggestions for improvement.\n"
            f"3. A final verdict: one of {VERDICT_ACCEPT}, {VERDICT_MINOR_REVISION}, "
            f"{VERDICT_MAJOR_REVISION}, or {VERDICT_REJECT}.\n"
            f"End your review with exactly: VERDICT: <{VERDICT_ACCEPT}|"
            f"{VERDICT_MINOR_REVISION}|{VERDICT_MAJOR_REVISION}|{VERDICT_REJECT}>"
        )
        try:
            return llm.complete(prompt)
        except Exception as exc:
            self.log(f"LLM review failed: {exc}", level="error")
            return (
                f"[Review unavailable due to LLM error: {exc}]\n"
                f"VERDICT: {VERDICT_MAJOR_REVISION}"
            )

    @staticmethod
    def _extract_verdict(review_text: str) -> str:
        import re

        m = re.search(
            r"VERDICT:\s*("
            + "|".join(
                [VERDICT_ACCEPT, VERDICT_MINOR_REVISION, VERDICT_MAJOR_REVISION, VERDICT_REJECT]
            )
            + r")",
            review_text,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()
        # Fallback heuristic
        text_lower = review_text.lower()
        if "reject" in text_lower:
            return VERDICT_REJECT
        if "accept" in text_lower:
            return VERDICT_ACCEPT
        return VERDICT_MAJOR_REVISION

    @staticmethod
    def _consolidate_feedback(reviews: Dict[str, str]) -> List[str]:
        """Extract the most actionable bullet points from all reviews."""
        import re

        feedback: List[str] = []
        for persona, review in reviews.items():
            # Extract bullet points
            bullets = re.findall(r"[-•*]\s+(.+)", review)
            for b in bullets[:3]:  # Max 3 bullets per reviewer
                feedback.append(f"[{persona.title()}] {b.strip()}")
        return feedback

    # ------------------------------------------------------------------
    # Debate loop
    # ------------------------------------------------------------------

    async def debate_loop(
        self,
        writer_agent: Any,
        initial_manuscript: str,
        topic: str,
        hypotheses: List[str],
        experiment_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run the full multi-round adversarial debate.

        The WriterAgent revises the manuscript until the Council accepts it
        or the maximum number of rounds is exhausted.

        Returns the final review result dict plus the final manuscript.
        """
        manuscript = initial_manuscript
        result: Dict[str, Any] = {}

        for round_num in range(1, self._review_rounds + 1):
            self.log(f"Starting debate round {round_num}/{self._review_rounds}")
            result = await self.run(manuscript=manuscript, round_num=round_num)

            if result.get("accepted"):
                self.log(f"Paper ACCEPTED at round {round_num}.")
                break

            if round_num < self._review_rounds:
                # Writer revises based on feedback
                revision = await writer_agent.revise(
                    previous_source=manuscript,
                    feedback=result.get("consolidated_feedback", []),
                    revision_round=round_num,
                    topic=topic,
                    hypotheses=hypotheses,
                    experiment_results=experiment_results,
                )
                manuscript = revision.get("latex_source", manuscript)
        else:
            self.log(
                f"Max rounds ({self._review_rounds}) reached — "
                f"final verdict: {'ACCEPTED' if result.get('accepted') else 'NOT ACCEPTED'}."
            )

        result["final_manuscript"] = manuscript
        return result
