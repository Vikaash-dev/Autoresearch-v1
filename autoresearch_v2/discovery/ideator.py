"""
discovery/ideator.py — ArXiv-grounded Evolutionary Idea Generator

Generates research hypotheses by:
  1. Fetching recent ArXiv papers on the given topic.
  2. Applying evolutionary "crossover" and "mutation" to abstracts.
  3. Using an LLM to formulate concrete, testable hypotheses.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from autoresearch_v2.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class IdeatorAgent(BaseAgent):
    """ArXiv-grounded evolutionary hypothesis generator."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a creative research ideator operating at the frontier of machine learning. "
        "Given several research paper abstracts and a topic, identify non-obvious connections "
        "and formulate novel, testable hypotheses. "
        "Each hypothesis must be: specific, falsifiable, and achievable with Python code. "
        "Focus on anomalies and gaps in the literature, not incremental improvements."
    )

    def __init__(self, config: dict, **kwargs: Any) -> None:
        super().__init__(config, name="IdeatorAgent", **kwargs)
        idea_cfg = config.get("ideation", {})
        self._arxiv_max = idea_cfg.get("arxiv_max_results", 20)
        self._hypothesis_count = idea_cfg.get("hypothesis_count", 10)
        self._crossover_rate = idea_cfg.get("crossover_rate", 0.5)
        self._mutation_rate = idea_cfg.get("mutation_rate", 0.2)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def _execute(
        self,
        topic: str,
        n_hypotheses: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate hypotheses for *topic*.

        Returns a dict with:
          - hypotheses: List[str]
          - arxiv_papers: List[Dict]  — titles + abstracts used as seed
          - topic: str
        """
        from autoresearch_v2.env.tools import ArXivTool, LLMTool

        n = n_hypotheses or self._hypothesis_count
        llm = LLMTool(self.config)
        arxiv = ArXivTool(self.config)

        self.log(f"Fetching ArXiv papers for topic: {topic!r}")
        papers = arxiv.search(topic, max_results=self._arxiv_max)

        self.log(f"Fetched {len(papers)} papers. Generating {n} hypotheses...")
        hypotheses = self._generate_hypotheses(llm, topic, papers, n)

        self.log(f"Generated {len(hypotheses)} hypotheses.")
        return {"hypotheses": hypotheses, "arxiv_papers": papers, "topic": topic}

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------

    def _generate_hypotheses(
        self,
        llm: Any,
        topic: str,
        papers: List[Dict[str, str]],
        n: int,
    ) -> List[str]:
        """Apply evolutionary crossover/mutation on abstracts, then LLM-synthesise."""
        abstracts = [p.get("abstract", "") for p in papers if p.get("abstract")]

        if len(abstracts) >= 2:
            seed_text = self._crossover(abstracts)
        elif len(abstracts) == 1:
            seed_text = abstracts[0]
        else:
            seed_text = f"General knowledge about {topic}."

        if random.random() < self._mutation_rate:
            seed_text = self._mutate(seed_text, topic)

        prompt = (
            self.system_prompt + "\n\n"
            f"Research topic: {topic}\n\n"
            f"Literature seed (crossover of recent abstracts):\n{seed_text[:2000]}\n\n"
            f"Generate exactly {n} distinct, numbered research hypotheses. "
            "Each hypothesis should be one concise sentence starting with 'H<N>: '. "
            "Example: 'H1: Applying spectral normalisation to attention weights reduces "
            "training variance in sparse transformers.'\n"
            "Output ONLY the numbered list."
        )
        raw = llm.complete(prompt)
        return self._parse_hypotheses(raw, n)

    def _crossover(self, abstracts: List[str]) -> str:
        """Randomly pair and splice abstract segments (simulated evolutionary crossover)."""
        n = len(abstracts)
        idx_a = random.randint(0, n - 1)
        idx_b = random.randint(0, n - 1)
        while idx_b == idx_a and n > 1:
            idx_b = random.randint(0, n - 1)
        a, b = abstracts[idx_a], abstracts[idx_b]
        mid_a = len(a) // 2
        mid_b = len(b) // 2
        return a[:mid_a] + " [CROSSOVER] " + b[mid_b:]

    def _mutate(self, text: str, topic: str) -> str:
        """Insert a topic-specific anomaly trigger into the seed text."""
        mutations = [
            f"What if the opposite were true for {topic}?",
            f"Consider an edge case where {topic} fails completely.",
            f"Apply {topic} to a domain where it has never been used before.",
        ]
        return text + "\n\n[MUTATION] " + random.choice(mutations)

    @staticmethod
    def _parse_hypotheses(raw: str, n: int) -> List[str]:
        """Extract H1..Hn from LLM output."""
        import re

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        hypotheses: List[str] = []
        for line in lines:
            m = re.match(r"H?\d+[.:)\-]\s*(.+)", line)
            if m:
                hypotheses.append(m.group(1).strip())
        # Fallback: return non-empty lines
        if not hypotheses:
            hypotheses = [l for l in lines if len(l) > 20]
        return hypotheses[:n]
