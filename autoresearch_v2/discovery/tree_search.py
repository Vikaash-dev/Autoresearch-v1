"""
discovery/tree_search.py — Branching Experiment Manager (Sakana v2 style)

Implements parallel exploration of N hypothesis branches with:
  - Multiprocessing-based concurrent execution.
  - Critic-Agent pruning after 20% of the compute budget.
  - "Grafting": merging code/framework from the best branches.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResearchBranch:
    """Represents a single hypothesis branch."""

    def __init__(self, branch_id: str, hypothesis: str) -> None:
        self.branch_id = branch_id
        self.hypothesis = hypothesis
        self.status: str = "pending"  # pending | running | done | pruned | failed
        self.result: Optional[Dict[str, Any]] = None
        self.novelty_score: float = 0.0
        self.significance_score: float = 0.0
        self.combined_score: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Branch(id={self.branch_id!r}, "
            f"status={self.status}, score={self.combined_score:.3f})"
        )


class TreeSearch:
    """
    Branching experiment manager with parallel execution and critic-based pruning.

    Usage::

        ts = TreeSearch(config, coder_agent, evaluator_agent)
        results = await ts.run(hypotheses)
    """

    def __init__(self, config: dict, coder_agent: Any, evaluator_agent: Any) -> None:
        self.config = config
        self.coder = coder_agent
        self.evaluator = evaluator_agent
        compute = config.get("compute", {})
        self._max_workers: int = compute.get("max_branch_workers", 4)
        self._prune_after_pct: float = compute.get("prune_after_pct", 0.20)
        self._prune_bottom_pct: float = compute.get("prune_bottom_pct", 0.50)

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    async def run(self, hypotheses: List[str]) -> Dict[str, Any]:
        """
        Explore all hypotheses in parallel, prune weak branches mid-way.

        Two-phase strategy:
          Phase 1 — run the first ``prune_after_pct`` fraction of branches,
                    score them, then kill the bottom ``prune_bottom_pct``.
          Phase 2 — run the remaining (unstarted) branches; already-alive
                    branches from Phase 1 are kept as-is (not re-run).

        Returns a dict with:
          - branches: List[ResearchBranch]
          - best_branch: ResearchBranch | None
          - grafted_result: Dict | None  (merged best code + framework)
        """
        branches = [
            ResearchBranch(branch_id=f"B{i+1}", hypothesis=h)
            for i, h in enumerate(hypotheses)
        ]
        logger.info("[TreeSearch] Starting %d branches.", len(branches))

        # Phase 1: run first prune_after_pct fraction of branches
        prune_at = max(1, math.ceil(len(branches) * self._prune_after_pct))
        early_branches = branches[:prune_at]
        remaining_branches = branches[prune_at:]

        await self._run_parallel(early_branches)
        self._score_branches(early_branches)

        # Prune bottom performers; returns only the surviving subset
        alive = self._prune(early_branches)
        logger.info(
            "[TreeSearch] After early pruning: %d/%d early branches alive.",
            len(alive),
            len(early_branches),
        )

        # Phase 2: run only the unstarted remaining branches
        if remaining_branches:
            await self._run_parallel(remaining_branches)
            self._score_branches(remaining_branches)

        # Combine: alive early branches + all remaining branches
        all_branches = alive + remaining_branches
        best = max(
            (b for b in all_branches if b.status == "done"),
            key=lambda b: b.combined_score,
            default=None,
        )
        grafted = self._graft(all_branches) if best else None

        return {
            "branches": all_branches,
            "best_branch": best,
            "grafted_result": grafted,
        }

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    async def _run_parallel(self, branches: List[ResearchBranch]) -> None:
        """Run branches concurrently, up to _max_workers at a time (in-place)."""
        semaphore = asyncio.Semaphore(self._max_workers)

        async def run_one(branch: ResearchBranch) -> None:
            async with semaphore:
                await self._execute_branch(branch)

        await asyncio.gather(*[run_one(b) for b in branches])

    async def _execute_branch(self, branch: ResearchBranch) -> None:
        """Execute the coder agent for one branch."""
        branch.status = "running"
        logger.info("[TreeSearch] Running branch %s.", branch.branch_id)
        try:
            result = await self.coder.run(
                hypothesis=branch.hypothesis,
                branch_id=branch.branch_id,
            )
            branch.result = result
            branch.status = "done" if result.get("success") else "failed"
        except Exception as exc:
            logger.error(
                "[TreeSearch] Branch %s failed: %s", branch.branch_id, exc
            )
            branch.result = {"success": False, "error": str(exc)}
            branch.status = "failed"

    # ------------------------------------------------------------------
    # Scoring & pruning
    # ------------------------------------------------------------------

    def _score_branches(self, branches: List[ResearchBranch]) -> None:
        """Assign novelty/significance scores via the evaluator."""
        for branch in branches:
            if branch.result is None:
                continue
            scores = self.evaluator.score(branch)
            branch.novelty_score = scores.get("novelty", 0.0)
            branch.significance_score = scores.get("significance", 0.0)
            branch.combined_score = (
                0.5 * branch.novelty_score + 0.5 * branch.significance_score
            )

    def _prune(self, branches: List[ResearchBranch]) -> List[ResearchBranch]:
        """
        Kill the bottom prune_bottom_pct branches.

        Only prune "done" branches with scores; failed ones are already dead.
        """
        scored = sorted(
            [b for b in branches if b.status in ("done", "failed")],
            key=lambda b: b.combined_score,
        )
        n_kill = max(0, math.floor(len(scored) * self._prune_bottom_pct))
        for branch in scored[:n_kill]:
            branch.status = "pruned"
            logger.info("[TreeSearch] Pruned branch %s (score=%.3f).", branch.branch_id, branch.combined_score)
        return [b for b in branches if b.status not in ("pruned", "failed")]

    # ------------------------------------------------------------------
    # Grafting
    # ------------------------------------------------------------------

    def _graft(self, branches: List[ResearchBranch]) -> Optional[Dict[str, Any]]:
        """
        Merge the best code from the top two scored branches.

        Returns a dict describing the grafted result.
        """
        done = sorted(
            [b for b in branches if b.status == "done"],
            key=lambda b: b.combined_score,
            reverse=True,
        )
        if len(done) < 2:
            return None

        a, b = done[0], done[1]
        logger.info(
            "[TreeSearch] Grafting branches %s (score=%.3f) + %s (score=%.3f).",
            a.branch_id, a.combined_score, b.branch_id, b.combined_score,
        )
        code_a = (a.result or {}).get("code", "")
        code_b = (b.result or {}).get("code", "")
        grafted_code = (
            f"# === Grafted from branch {a.branch_id} ===\n{code_a}\n\n"
            f"# === Framework from branch {b.branch_id} ===\n{code_b}"
        )
        return {
            "source_branches": [a.branch_id, b.branch_id],
            "code": grafted_code,
            "combined_score": (a.combined_score + b.combined_score) / 2,
        }
