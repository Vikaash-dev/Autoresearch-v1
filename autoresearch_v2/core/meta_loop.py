"""
core/meta_loop.py — MetaLoop: Domain-Aware Team Builder

The Meta-Loop is the *first* of the 5 operational loops in AutoResearch v2.
It analyses the user's research topic and dynamically builds the optimal
agent configuration for that specific scientific domain, rather than using
a fixed, generic team.

Proposal quote:
  "The system analyzes the user's prompt and builds the optimal team of
   agents for that specific domain (e.g., a 'Bio-Informatics Team' vs.
   a 'Quantum Physics Team')."

Domain configurations include:
  - Tailored LLM model selection (fast coder vs. deep reasoner)
  - Domain-specific reviewer persona weights
  - Adjusted hypothesis count and ArXiv category filters
  - Domain-specific evaluation heuristics
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain profiles
# ---------------------------------------------------------------------------

DOMAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "bioinformatics": {
        "description": "Biology, genomics, proteomics, drug discovery, neuroscience",
        "keywords": [
            "biology", "genomic", "protein", "dna", "rna", "drug", "clinical",
            "neuroscience", "biomarker", "cell", "genome", "phenotype", "omics",
        ],
        "llm_models": {
            "writer": "gpt-4o",
            "coder": "claude-3-5-sonnet",
            "ideator": "gemini-1.5-pro",
        },
        "arxiv_categories": ["q-bio", "cs.LG", "stat.ML"],
        "hypothesis_count": 15,
        "reviewer_emphasis": ["statistician", "engineer", "skeptic"],
        "eval_keywords": ["p-value", "clinical", "reproducibility", "cohort"],
    },
    "quantum_physics": {
        "description": "Quantum computing, quantum mechanics, condensed matter",
        "keywords": [
            "quantum", "qubit", "hamiltonian", "entanglement", "superposition",
            "condensed matter", "photon", "fermion", "boson", "lattice",
        ],
        "llm_models": {
            "writer": "gpt-4o",
            "coder": "claude-3-5-sonnet",
            "ideator": "gpt-4o",
        },
        "arxiv_categories": ["quant-ph", "cond-mat", "physics"],
        "hypothesis_count": 10,
        "reviewer_emphasis": ["skeptic", "visionary", "statistician"],
        "eval_keywords": ["fidelity", "error rate", "decoherence", "circuit depth"],
    },
    "machine_learning": {
        "description": "Deep learning, NLP, computer vision, reinforcement learning",
        "keywords": [
            "neural", "transformer", "llm", "gpt", "bert", "attention", "pruning",
            "distillation", "fine-tuning", "reinforcement", "vision", "nlp",
            "language model", "diffusion", "embedding",
        ],
        "llm_models": {
            "writer": "gpt-4o",
            "coder": "claude-3-5-sonnet",
            "ideator": "gemini-1.5-pro",
        },
        "arxiv_categories": ["cs.LG", "cs.AI", "cs.CL", "cs.CV"],
        "hypothesis_count": 20,
        "reviewer_emphasis": ["engineer", "skeptic", "pragmatist"],
        "eval_keywords": ["accuracy", "perplexity", "benchmark", "latency", "flops"],
    },
    "materials_science": {
        "description": "Materials, chemistry, solid-state physics, nanotechnology",
        "keywords": [
            "material", "alloy", "crystal", "polymer", "nanoparticle", "catalyst",
            "synthesis", "semiconductor", "electrode", "battery", "composite",
        ],
        "llm_models": {
            "writer": "gpt-4o",
            "coder": "gpt-4o",
            "ideator": "gemini-1.5-pro",
        },
        "arxiv_categories": ["cond-mat.mtrl-sci", "physics.chem-ph"],
        "hypothesis_count": 12,
        "reviewer_emphasis": ["skeptic", "statistician", "engineer"],
        "eval_keywords": ["yield", "conductivity", "stability", "efficiency"],
    },
    "general": {
        "description": "General / cross-disciplinary research",
        "keywords": [],
        "llm_models": {
            "writer": "gpt-4o",
            "coder": "claude-3-5-sonnet",
            "ideator": "gpt-4o",
        },
        "arxiv_categories": ["cs.AI", "stat.ML"],
        "hypothesis_count": 10,
        "reviewer_emphasis": ["skeptic", "engineer", "visionary"],
        "eval_keywords": ["accuracy", "improvement", "baseline"],
    },
}


# ---------------------------------------------------------------------------
# MetaLoop
# ---------------------------------------------------------------------------


class MetaLoop:
    """
    The Meta-Loop: analyses the research topic and builds an optimised
    agent configuration for that scientific domain.

    Usage::

        meta = MetaLoop(config)
        team_config = meta.build_team(topic)
        # team_config contains per-agent model overrides, adjusted hyper-params,
        # and domain-specific evaluation hints.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_team(self, topic: str) -> Dict[str, Any]:
        """
        Detect the domain of *topic* and return an optimised team configuration.

        Returns a dict merging ``config`` with domain-specific overrides:
          - domain: str
          - domain_description: str
          - llm_models: Dict[role -> model_name]
          - arxiv_categories: List[str]
          - hypothesis_count: int  (effort-scaled)
          - reviewer_emphasis: List[str]
          - eval_keywords: List[str]
          - lessons_injected: List[str]  (from ResearchMemory cross-run)
        """
        domain_key = self._classify_domain(topic)
        profile = DOMAIN_PROFILES[domain_key].copy()

        logger.info("[MetaLoop] Topic %r → domain: %s", topic, domain_key)
        logger.info("[MetaLoop] Description: %s", profile["description"])

        # Scale hypothesis_count by effort level
        effort = self.config.get("effort", "standard")
        profile["hypothesis_count"] = self._scale_hypotheses(
            profile["hypothesis_count"], effort
        )

        # Inject cross-run lessons from memory
        lessons = self._inject_lessons(topic, domain_key)
        profile["lessons_injected"] = lessons
        profile["domain"] = domain_key
        profile["domain_description"] = profile["description"]

        # Log team composition
        models = profile["llm_models"]
        logger.info(
            "[MetaLoop] Team assembled — writer=%s | coder=%s | ideator=%s "
            "| hypotheses=%d | reviewers=%s",
            models["writer"], models["coder"], models["ideator"],
            profile["hypothesis_count"],
            profile["reviewer_emphasis"],
        )

        return profile

    # ------------------------------------------------------------------
    # Domain classification
    # ------------------------------------------------------------------

    def _classify_domain(self, topic: str) -> str:
        """Return the best-matching domain key for *topic*."""
        topic_lower = topic.lower()
        best_domain = "general"
        best_score = 0

        for domain_key, profile in DOMAIN_PROFILES.items():
            if domain_key == "general":
                continue
            score = sum(1 for kw in profile["keywords"] if kw in topic_lower)
            if score > best_score:
                best_score = score
                best_domain = domain_key

        # Use LLM for ambiguous cases (score == 0 and topic is multi-word)
        if best_score == 0 and len(topic.split()) >= 3:
            best_domain = self._llm_classify(topic)

        return best_domain

    def _llm_classify(self, topic: str) -> str:
        """Ask the LLM to classify the domain when keywords don't match."""
        try:
            from autoresearch_v2.env.tools import LLMTool

            llm = LLMTool(self.config)
            domains_list = ", ".join(
                k for k in DOMAIN_PROFILES if k != "general"
            )
            prompt = (
                f"Classify the following research topic into exactly one of these domains: "
                f"{domains_list}, general.\n\n"
                f"Topic: {topic}\n\n"
                "Respond with ONLY the domain name (one word, lowercase)."
            )
            response = llm.complete(prompt).strip().lower()
            # Extract first word that matches a known domain
            for word in re.split(r"\W+", response):
                if word in DOMAIN_PROFILES:
                    return word
        except Exception as exc:
            logger.warning("[MetaLoop] LLM classification failed: %s", exc)
        return "general"

    # ------------------------------------------------------------------
    # Effort scaling
    # ------------------------------------------------------------------

    # Effort-level parameters: (multiplier, minimum_count)
    _EFFORT_PARAMS: Dict[str, tuple] = {
        "minimal":  (0.5, 1),
        "standard": (1.0, 1),
        "pro":      (3.0, 10),
        "max":      (5.0, 50),   # ≥50 — matches proposal "50+ hypotheses"
    }

    @classmethod
    def _scale_hypotheses(cls, base: int, effort: str) -> int:
        """
        Scale hypothesis count by effort level.

        ``_EFFORT_PARAMS`` is the single source of truth; multiplier and
        minimum are co-located to ensure they stay in sync.
        """
        multiplier, minimum = cls._EFFORT_PARAMS.get(effort, (1.0, 1))
        return max(int(base * multiplier), minimum)

    # ------------------------------------------------------------------
    # Cross-run skill injection
    # ------------------------------------------------------------------

    def _inject_lessons(self, topic: str, domain: str) -> List[str]:
        """
        Retrieve relevant skills/lessons from previous runs via ResearchMemory
        and return them as a list of strings to be injected into agent prompts.
        """
        try:
            from autoresearch_v2.core.memory import ResearchMemory

            memory = ResearchMemory(self.config)
            skills = memory.get_skills(topic)
            lessons = [s["text"] for s in skills[:5]]
            if lessons:
                logger.info(
                    "[MetaLoop] Injecting %d lessons from previous runs.", len(lessons)
                )
            return lessons
        except Exception as exc:
            logger.warning("[MetaLoop] Could not load lessons: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Apply team config to agent prompts
    # ------------------------------------------------------------------

    def apply_to_agents(
        self,
        team_config: Dict[str, Any],
        agents: Dict[str, Any],
    ) -> None:
        """
        Inject lessons and domain context into each agent's system prompt.

        ``agents`` is a dict of {role: agent_instance}.
        """
        lessons = team_config.get("lessons_injected", [])
        domain_desc = team_config.get("domain_description", "")

        for role, agent in agents.items():
            if not hasattr(agent, "system_prompt"):
                continue
            injections = []
            if domain_desc:
                injections.append(f"[Domain context: {domain_desc}]")
            if lessons:
                lesson_str = " | ".join(lessons[:3])
                injections.append(f"[Lessons from previous runs: {lesson_str}]")
            if injections:
                agent.system_prompt = agent.system_prompt + "\n\n" + "\n".join(injections)
                logger.debug(
                    "[MetaLoop] Injected %d items into %s prompt.", len(injections), role
                )
