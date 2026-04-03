"""AutoResearch v2 (HRDE) — package root."""

from autoresearch_v2.core.kernel import HyperKernel
from autoresearch_v2.core.graph_manager import GraphManager
from autoresearch_v2.core.memory import ResearchMemory

__all__ = ["HyperKernel", "GraphManager", "ResearchMemory"]
__version__ = "2.0.0"
