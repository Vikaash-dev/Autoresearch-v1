"""
agents/reviewer.py — ToM-based Adversarial Reviewer Council

Implements a 5-persona review system matching the HRDE proposal:
  - The Skeptic     — finds methodological flaws and p-hacking.
  - The Engineer    — focuses on code reproducibility and efficiency.
  - The Visionary   — evaluates long-term impact and novelty.
  - The Statistician — scrutinises statistical rigour and sample sizes.
  - The Pragmatist  — evaluates practical deployability and real-world impact.

The paper is accepted only when ≥ ``acceptance_threshold`` (default 4/5)
personas approve after up to ``review_rounds`` rounds of adversarial dialogue.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from autoresearch_v2.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Persona definitions  (5 personas — proposal: "4/5 consensus")
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
    "statistician": (
        "You are The Statistician — a biostatistics and experimental design expert. "
        "Your goal is to verify statistical rigour: correct test selection, adequate "
        "sample sizes, confidence intervals, effect sizes, and multiple-comparison "
        "corrections. Flag any result that is statistically underpowered or misleading."
    ),
    "pragmatist": (
        "You are The Pragmatist — an industry ML engineer and deployment specialist. "
        "Your goal is to evaluate whether this research is practically deployable: "
        "inference cost, latency, hardware requirements, and real-world failure modes. "
        "A beautiful paper that cannot run in production is not acceptable."
    ),
}

VERDICT_ACCEPT = "ACCEPT"
VERDICT_REJECT = "REJECT"
VERDICT_MAJOR_REVISION = "MAJOR_REVISION"
VERDICT_MINOR_REVISION = "MINOR_REVISION"

# Surprise and Rigorous are the two primary quality dimensions from the proposal.
QUALITY_DIMENSIONS = ("surprise", "rigorous")


class ReviewerCouncil(BaseAgent):
    """
    Multi-persona adversarial review council (5 personas, 4/5 consensus).

    Usage::

        council = ReviewerCouncil(config)
        result = await council.run(manuscript=latex_src, round_num=1)
        # result["accepted"] is True when >= acceptance_threshold personas approve.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are coordinating a multi-persona peer review. "
        "Each reviewer has a distinct cognitive perspective and scoring rubric. "
        "The Council's goal is to maximise both the Surprise (novelty) and "
        "Rigorous (methodological soundness) dimensions of the paper."
    )

    def __init__(
        self,
        config: dict,
        personas: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, name="ReviewerCouncil", **kwargs)
        raw = config.get("review_council", "skeptic,engineer,visionary,statistician,pragmatist")
        self._active_personas: List[str] = personas or [
            p.strip().lower() for p in raw.split(",") if p.strip()
        ]
        # Default 4/5 matching proposal's "4 out of 5 consensus"
        self._acceptance_threshold: int = config.get(
            "acceptance_threshold", max(1, len(self._active_personas) - 1)
        )
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
          - quality_scores: Dict[persona -> {surprise, rigorous}]
          - accepted: bool
          - round_num: int
          - consolidated_feedback: List[str]
          - mean_surprise: float
          - mean_rigorous: float
        """
        from autoresearch_v2.env.tools import LLMTool

        llm = LLMTool(self.config)

        reviews: Dict[str, str] = {}
        verdicts: Dict[str, str] = {}
        quality_scores: Dict[str, Dict[str, float]] = {}

        for persona_key in self._active_personas:
            persona_prompt = PERSONAS.get(persona_key, PERSONAS["skeptic"])
            review_text = self._review_with_persona(
                llm, persona_prompt, manuscript, round_num
            )
            verdict = self._extract_verdict(review_text)
            scores = self._extract_quality_scores(review_text)
            reviews[persona_key] = review_text
            verdicts[persona_key] = verdict
            quality_scores[persona_key] = scores
            self.log(
                f"[Round {round_num}] {persona_key.title()} verdict={verdict} "
                f"surprise={scores['surprise']:.2f} rigorous={scores['rigorous']:.2f}"
            )

        approvals = sum(
            1 for v in verdicts.values()
            if v in (VERDICT_ACCEPT, VERDICT_MINOR_REVISION)
        )
        accepted = approvals >= self._acceptance_threshold

        all_surprise = [s["surprise"] for s in quality_scores.values()]
        all_rigorous = [s["rigorous"] for s in quality_scores.values()]
        mean_surprise = sum(all_surprise) / len(all_surprise) if all_surprise else 0.0
        mean_rigorous = sum(all_rigorous) / len(all_rigorous) if all_rigorous else 0.0

        feedback = self._consolidate_feedback(reviews)

        self.log(
            f"[Round {round_num}] Council: {approvals}/{len(self._active_personas)} approve "
            f"| surprise={mean_surprise:.2f} rigorous={mean_rigorous:.2f} "
            f"→ {'ACCEPTED' if accepted else 'REVISION'}"
        )

        return {
            "reviews": reviews,
            "verdicts": verdicts,
            "quality_scores": quality_scores,
            "accepted": accepted,
            "approvals": approvals,
            "round_num": round_num,
            "consolidated_feedback": feedback,
            "mean_surprise": mean_surprise,
            "mean_rigorous": mean_rigorous,
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
            "Review the following research manuscript excerpt and evaluate it on "
            "two dimensions:\n"
            "  Surprise (0.0–1.0): How novel and unexpected are the findings?\n"
            "  Rigorous (0.0–1.0): How methodologically sound is the work?\n\n"
            f"{manuscript[:3000]}\n\n"
            "Provide:\n"
            "1. Up to 5 bullet-point concerns.\n"
            "2. Specific improvement suggestions.\n"
            f"3. VERDICT: one of {VERDICT_ACCEPT}, {VERDICT_MINOR_REVISION}, "
            f"{VERDICT_MAJOR_REVISION}, {VERDICT_REJECT}\n"
            "4. SURPRISE: <float 0.0–1.0>\n"
            "5. RIGOROUS: <float 0.0–1.0>"
        )
        try:
            return llm.complete(prompt)
        except Exception as exc:
            self.log(f"LLM review failed: {exc}", level="error")
            return (
                f"[Review unavailable: {exc}]\n"
                f"VERDICT: {VERDICT_MAJOR_REVISION}\nSURPRISE: 0.50\nRIGOROUS: 0.50"
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
        text_lower = review_text.lower()
        if "reject" in text_lower:
            return VERDICT_REJECT
        if "accept" in text_lower:
            return VERDICT_ACCEPT
        return VERDICT_MAJOR_REVISION

    @staticmethod
    def _extract_quality_scores(review_text: str) -> Dict[str, float]:
        """Parse SURPRISE and RIGOROUS float scores from review text (clamped to [0,1])."""
        import re

        scores = {"surprise": 0.5, "rigorous": 0.5}
        for dim in ("surprise", "rigorous"):
            m = re.search(
                rf"{dim}\s*:\s*([01](?:\.\d+)?|\.\d+)",  # 0–1 only
                review_text, re.IGNORECASE,
            )
            if m:
                try:
                    scores[dim] = min(1.0, max(0.0, float(m.group(1))))
                except ValueError:
                    pass
        return scores

    @staticmethod
    def _consolidate_feedback(reviews: Dict[str, str]) -> List[str]:
        """Extract the most actionable bullet points from all reviews."""
        import re

        feedback: List[str] = []
        for persona, review in reviews.items():
            bullets = re.findall(r"[-•*]\s+(.+)", review)
            for b in bullets[:3]:
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
        Run the full multi-round adversarial debate (The Review Loop).

        The WriterAgent revises the manuscript until the Council accepts it
        (≥ acceptance_threshold approvals) or max rounds are exhausted.
        Self-correction targets maximising both Surprise and Rigorous metrics.
        """
        manuscript = initial_manuscript
        result: Dict[str, Any] = {}

        for round_num in range(1, self._review_rounds + 1):
            self.log(
                f"[Review Loop] Round {round_num}/{self._review_rounds} "
                f"(need {self._acceptance_threshold}/{len(self._active_personas)} approvals)"
            )
            result = await self.run(manuscript=manuscript, round_num=round_num)

            if result.get("accepted"):
                self.log(
                    f"[Review Loop] Paper ACCEPTED at round {round_num} "
                    f"(surprise={result['mean_surprise']:.2f}, "
                    f"rigorous={result['mean_rigorous']:.2f})."
                )
                break

            if round_num < self._review_rounds:
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
                f"[Review Loop] Max rounds ({self._review_rounds}) reached — "
                f"final: {'ACCEPTED' if result.get('accepted') else 'NOT ACCEPTED'}."
            )

        result["final_manuscript"] = manuscript
        return result

