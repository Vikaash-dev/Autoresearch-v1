"""
discovery/tree_search.py — Branching Experiment Manager (Sakana v2 style)

Implements parallel exploration of N hypothesis branches with:
  - Asyncio-based concurrent execution.
  - Two-stage pruning:
      * Count-based: after prune_after_pct of branches complete.
      * Time-based:  Critic Agent evaluates every prune_interval_secs seconds
                     (proposal: "every 30 minutes") and kills underperformers.
  - "Grafting": merging code/framework from the best surviving branches.
  - Compute reallocation: pruned branch slots are given to the best branch.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResearchBranch:
    """Represents a single hypothesis branch."""

    def __init__(self, branch_id: str, hypothesis: str) -> None:
        self.branch_id = branch_id
        self.hypothesis = hypothesis
        self.status: str = "pending"   # pending | running | done | pruned | failed
        self.result: Optional[Dict[str, Any]] = None
        self.novelty_score: float = 0.0
        self.significance_score: float = 0.0
        self.combined_score: float = 0.0
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    @property
    def elapsed_secs(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.monotonic()
        return end - self.started_at

    def __repr__(self) -> str:
        return (
            f"Branch(id={self.branch_id!r}, "
            f"status={self.status}, score={self.combined_score:.3f})"
        )


class TreeSearch:
    """
    Branching experiment manager with parallel execution, count-based
    and time-based pruning, and branch grafting.

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
        # Time-based pruning: default 1800s (30 min) — matches proposal
        self._prune_interval_secs: float = compute.get("prune_interval_secs", 1800)
        self._last_timed_prune: float = 0.0

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    async def run(self, hypotheses: List[str]) -> Dict[str, Any]:
        """
        Explore all hypotheses in parallel with two-stage pruning.

        Two-phase strategy
        ------------------
        Phase 1 — run the first ``prune_after_pct`` fraction of branches,
                  score them, kill the bottom ``prune_bottom_pct``.
        Phase 2 — run remaining (unstarted) branches; alive Phase-1 branches
                  are kept as-is (never re-executed).

        Compute reallocation: when branches are pruned, the best surviving
        branch hypothesis is extended with a "grafted focus" note.

        Returns a dict with:
          - branches: List[ResearchBranch]
          - best_branch: ResearchBranch | None
          - grafted_result: Dict | None
        """
        branches = [
            ResearchBranch(branch_id=f"B{i+1}", hypothesis=h)
            for i, h in enumerate(hypotheses)
        ]
        logger.info("[TreeSearch] Starting %d branches.", len(branches))
        self._last_timed_prune = time.monotonic()

        # Phase 1: count-based early pruning
        prune_at = max(1, math.ceil(len(branches) * self._prune_after_pct))
        early_branches = branches[:prune_at]
        remaining_branches = branches[prune_at:]

        await self._run_parallel(early_branches)
        self._score_branches(early_branches)
        alive = self._prune(early_branches, label="early")
        self._reallocate_compute(alive)

        logger.info(
            "[TreeSearch] After count-based pruning: %d/%d early branches alive.",
            len(alive), len(early_branches),
        )

        # Phase 2: run remaining branches with periodic time-based pruning
        if remaining_branches:
            await self._run_parallel_with_time_pruning(remaining_branches)
            self._score_branches(remaining_branches)

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

    async def _run_parallel_with_time_pruning(
        self, branches: List[ResearchBranch]
    ) -> None:
        """
        Run branches with periodic time-based critic pruning.

        Every ``_prune_interval_secs`` seconds, the Critic Agent evaluates
        completed branches and kills underperformers, then reallocates their
        compute budget to the best survivor.

        An asyncio lock prevents concurrent pruning when multiple branches
        complete around the same time.
        """
        semaphore = asyncio.Semaphore(self._max_workers)
        prune_lock = asyncio.Lock()

        async def run_one(branch: ResearchBranch) -> None:
            async with semaphore:
                await self._execute_branch(branch)
            # After each branch completes, check if it's time for timed pruning.
            # Use a lock so only one concurrent completion triggers the prune.
            now = time.monotonic()
            if now - self._last_timed_prune >= self._prune_interval_secs:
                async with prune_lock:
                    # Re-check after acquiring the lock (another branch may have pruned)
                    if time.monotonic() - self._last_timed_prune >= self._prune_interval_secs:
                        self._last_timed_prune = time.monotonic()
                        done_so_far = [b for b in branches if b.status == "done"]
                        self._score_branches(done_so_far)
                        self._prune(branches, label="timed")
                        self._reallocate_compute(
                            [b for b in branches if b.status not in ("pruned", "failed")]
                        )

        await asyncio.gather(*[run_one(b) for b in branches])

    async def _execute_branch(self, branch: ResearchBranch) -> None:
        """Execute the coder agent for one branch."""
        branch.status = "running"
        branch.started_at = time.monotonic()
        logger.info("[TreeSearch] Running branch %s: %s", branch.branch_id, branch.hypothesis[:60])
        try:
            result = await self.coder.run(
                hypothesis=branch.hypothesis,
                branch_id=branch.branch_id,
            )
            branch.result = result
            branch.status = "done" if result.get("success") else "failed"
        except Exception as exc:
            logger.error("[TreeSearch] Branch %s failed: %s", branch.branch_id, exc)
            branch.result = {"success": False, "error": str(exc)}
            branch.status = "failed"
        finally:
            branch.finished_at = time.monotonic()

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

    def _prune(
        self, branches: List[ResearchBranch], label: str = ""
    ) -> List[ResearchBranch]:
        """
        Kill the bottom prune_bottom_pct scored branches.
        Returns the list of surviving branches.
        """
        scored = sorted(
            [b for b in branches if b.status in ("done", "failed")],
            key=lambda b: b.combined_score,
        )
        n_kill = max(0, math.floor(len(scored) * self._prune_bottom_pct))
        for branch in scored[:n_kill]:
            branch.status = "pruned"
            logger.info(
                "[TreeSearch][%s] Pruned branch %s (score=%.3f).",
                label, branch.branch_id, branch.combined_score,
            )
        return [b for b in branches if b.status not in ("pruned", "failed")]

    def _reallocate_compute(self, alive: List[ResearchBranch]) -> None:
        """
        Compute reallocation: annotate the best surviving branch to signal
        that it should receive any freed resources (extra retries / larger budget).
        """
        if not alive:
            return
        best = max(alive, key=lambda b: b.combined_score)
        if best.result and isinstance(best.result, dict):
            best.result["compute_reallocated"] = True
        logger.info(
            "[TreeSearch] Compute reallocated to best branch %s (score=%.3f).",
            best.branch_id, best.combined_score,
        )

    # ------------------------------------------------------------------
    # Grafting
    # ------------------------------------------------------------------

    def _graft(self, branches: List[ResearchBranch]) -> Optional[Dict[str, Any]]:
        """
        Merge the best code from the top two scored branches.

        Proposal: "takes the successful code from Branch A and the successful
        theoretical framework from Branch B to create a Hybrid Branch C."
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
            "[TreeSearch] Grafting branches %s (score=%.3f) + %s (score=%.3f) → Hybrid C.",
            a.branch_id, a.combined_score, b.branch_id, b.combined_score,
        )
        code_a = (a.result or {}).get("code", "")
        code_b = (b.result or {}).get("code", "")
        grafted_code = (
            f"# === Grafted: code from branch {a.branch_id} "
            f"(score={a.combined_score:.3f}) ===\n{code_a}\n\n"
            f"# === Grafted: framework from branch {b.branch_id} "
            f"(score={b.combined_score:.3f}) ===\n{code_b}"
        )
        return {
            "source_branches": [a.branch_id, b.branch_id],
            "code": grafted_code,
            "combined_score": (a.combined_score + b.combined_score) / 2,
            "hybrid_label": "C",
        }

