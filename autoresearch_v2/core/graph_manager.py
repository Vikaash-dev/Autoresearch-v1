"""
core/graph_manager.py — Dynamic Objective Graph (DOG) Orchestrator

Replaces linear pipelines with an asynchronous task graph backed by
networkx.  Failed tasks can dynamically spawn recovery tasks.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import networkx as nx

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    DONE = auto()
    FAILED = auto()
    SPAWNED = auto()


class ObjectiveTask:
    """A node in the Dynamic Objective Graph."""

    def __init__(
        self,
        name: str,
        fn: Callable[..., Coroutine],
        kwargs: Optional[Dict[str, Any]] = None,
        recovery_task: Optional[str] = None,
    ) -> None:
        self.name = name
        self.fn = fn
        self.kwargs: Dict[str, Any] = kwargs or {}
        self.recovery_task = recovery_task  # task to spawn on failure
        self.status = TaskStatus.PENDING
        self.result: Any = None
        self.error: Optional[Exception] = None

    def __repr__(self) -> str:
        return f"ObjectiveTask(name={self.name!r}, status={self.status.name})"


class GraphManager:
    """
    Dynamic Objective Graph (DOG) Orchestrator.

    Usage::

        gm = GraphManager(config)
        gm.add_task("ideation", ideation_fn)
        gm.add_task("coding", coding_fn, depends_on=["ideation"],
                    recovery_task="library_research")
        gm.add_task("library_research", research_fn)
        gm.add_task("review", review_fn, depends_on=["coding"])
        await gm.run()
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._graph: nx.DiGraph = nx.DiGraph()
        self._tasks: Dict[str, ObjectiveTask] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_task(
        self,
        name: str,
        fn: Callable[..., Coroutine],
        kwargs: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        recovery_task: Optional[str] = None,
    ) -> None:
        """Register a task node in the DOG."""
        task = ObjectiveTask(name=name, fn=fn, kwargs=kwargs, recovery_task=recovery_task)
        self._tasks[name] = task
        self._graph.add_node(name)
        for dep in depends_on or []:
            self._graph.add_edge(dep, name)
        logger.debug("[DOG] Added task '%s' (deps=%s)", name, depends_on)

    def _validate_graph(self) -> None:
        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValueError("Task graph contains a cycle — cannot execute.")

    # ------------------------------------------------------------------
    # Execution engine
    # ------------------------------------------------------------------

    async def _execute_task(self, name: str) -> None:
        task = self._tasks[name]
        task.status = TaskStatus.RUNNING
        logger.info("[DOG] Executing task '%s'", name)
        try:
            task.result = await task.fn(**task.kwargs)
            task.status = TaskStatus.DONE
            logger.info("[DOG] Task '%s' completed successfully.", name)
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = exc
            logger.error("[DOG] Task '%s' FAILED: %s", name, exc)
            if task.recovery_task:
                self._spawn_recovery(name, task.recovery_task)

    def _spawn_recovery(self, failed_task: str, recovery_name: str) -> None:
        """Dynamically insert a recovery task into the graph."""
        if recovery_name in self._tasks:
            logger.info(
                "[DOG] Spawning recovery task '%s' for failed '%s'.",
                recovery_name,
                failed_task,
            )
            # The recovery task does not add edges; it runs independently.
            self._tasks[recovery_name].status = TaskStatus.SPAWNED
        else:
            logger.warning(
                "[DOG] Recovery task '%s' not registered.", recovery_name
            )

    async def run(self) -> Dict[str, Any]:
        """
        Execute the task graph respecting dependency order.

        Returns a mapping of task name -> result.
        """
        self._validate_graph()
        completed: Set[str] = set()
        results: Dict[str, Any] = {}

        # Topological execution order
        order = list(nx.topological_sort(self._graph))

        for name in order:
            task = self._tasks.get(name)
            if task is None:
                continue

            # Wait until all predecessors have finished (done or failed)
            preds = list(self._graph.predecessors(name))
            blocking = [
                p for p in preds
                if self._tasks.get(p) and self._tasks[p].status not in (
                    TaskStatus.DONE, TaskStatus.FAILED
                )
            ]
            if blocking:
                logger.warning(
                    "[DOG] Task '%s' blocked by: %s — skipping.", name, blocking
                )
                continue

            # Skip if all dependencies failed
            failed_preds = [
                p for p in preds
                if self._tasks.get(p) and self._tasks[p].status == TaskStatus.FAILED
            ]
            if failed_preds and len(failed_preds) == len(preds) and preds:
                logger.warning(
                    "[DOG] All predecessors of '%s' failed — skipping.", name
                )
                task.status = TaskStatus.FAILED
                continue

            await self._execute_task(name)
            completed.add(name)
            results[name] = task.result

        # Run any dynamically spawned recovery tasks
        for name, task in self._tasks.items():
            if task.status == TaskStatus.SPAWNED:
                await self._execute_task(name)
                results[name] = task.result

        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status_report(self) -> Dict[str, str]:
        return {name: task.status.name for name, task in self._tasks.items()}

    def failed_tasks(self) -> List[str]:
        return [n for n, t in self._tasks.items() if t.status == TaskStatus.FAILED]

    def visualize(self, path: str = "./logs/dog_graph.png") -> None:
        """Save a simple PNG of the task graph (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            pos = nx.spring_layout(self._graph, seed=42)
            color_map = {
                TaskStatus.PENDING: "lightgrey",
                TaskStatus.RUNNING: "yellow",
                TaskStatus.DONE: "lightgreen",
                TaskStatus.FAILED: "tomato",
                TaskStatus.SPAWNED: "lightblue",
            }
            colors = [
                color_map.get(self._tasks[n].status, "white")
                for n in self._graph.nodes
            ]
            plt.figure(figsize=(10, 6))
            nx.draw(
                self._graph,
                pos,
                with_labels=True,
                node_color=colors,
                node_size=2000,
                font_size=10,
            )
            plt.tight_layout()
            import pathlib

            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path)
            plt.close()
            logger.info("[DOG] Graph visualized → %s", path)
        except ImportError:
            logger.warning("[DOG] matplotlib not installed; skipping graph visualization.")
