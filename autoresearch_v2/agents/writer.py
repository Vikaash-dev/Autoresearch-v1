"""
agents/writer.py — LaTeX/TikZ Manuscript Generator Agent

Responsibilities:
  - Compose a full research paper in LaTeX given:
      * topic, hypotheses, experiment results, and reviewer feedback.
  - Insert TikZ diagrams and numerical tables from execution logs.
  - Revise the manuscript in response to Reviewer Council feedback.
  - (Optionally) compile to PDF via pdflatex.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoresearch_v2.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

NEURIPS_TEMPLATE = r"""
\documentclass{article}
\usepackage[preprint]{neurips_2024}
\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref,tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric}

\title{<<TITLE>>}
\author{AutoResearch~v2 (HRDE)\\Generated Manuscript}

\begin{document}
\maketitle

\begin{abstract}
<<ABSTRACT>>
\end{abstract}

<<BODY>>

\bibliographystyle{plain}
\end{document}
""".strip()


class WriterAgent(BaseAgent):
    """LaTeX / TikZ manuscript generator with iterative revision support."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a scientific writer producing NeurIPS-quality research papers. "
        "Structure your work with Abstract, Introduction, Related Work, Methodology, "
        "Experiments, Results, Discussion, and Conclusion sections. "
        "Include TikZ diagrams where appropriate to visualise architecture or workflow. "
        "All numerical claims must cite the exact experiment log line they originate from. "
        "Use formal, precise academic prose."
    )

    def __init__(self, config: dict, **kwargs: Any) -> None:
        super().__init__(config, name="WriterAgent", **kwargs)
        self._template = NEURIPS_TEMPLATE

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def _execute(
        self,
        topic: str,
        hypotheses: List[str],
        experiment_results: List[Dict[str, Any]],
        reviewer_feedback: Optional[List[str]] = None,
        revision_round: int = 0,
    ) -> Dict[str, Any]:
        """
        Draft (or revise) a LaTeX manuscript.

        Returns a dict with:
          - latex_source: str
          - pdf_path: str | None  (path if PDF compiled successfully)
          - revision_round: int
        """
        from autoresearch_v2.env.tools import LLMTool

        llm = LLMTool(self.config)

        latex_body = self._generate_body(
            llm, topic, hypotheses, experiment_results, reviewer_feedback, revision_round
        )
        title = self._extract_title(latex_body, topic)
        abstract = self._extract_abstract(latex_body)

        latex_source = (
            self._template
            .replace("<<TITLE>>", title)
            .replace("<<ABSTRACT>>", abstract)
            .replace("<<BODY>>", latex_body)
        )

        pdf_path: Optional[str] = None
        if self.config.get("output", {}).get("generate_pdf", False):
            pdf_path = self._compile_pdf(latex_source, topic)

        self.log(
            f"Manuscript draft ready (round={revision_round}, "
            f"len={len(latex_source)} chars)."
        )
        return {
            "latex_source": latex_source,
            "pdf_path": pdf_path,
            "revision_round": revision_round,
            "title": title,
        }

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _generate_body(
        self,
        llm: Any,
        topic: str,
        hypotheses: List[str],
        results: List[Dict[str, Any]],
        feedback: Optional[List[str]],
        revision_round: int,
    ) -> str:
        hyp_text = "\n".join(f"H{i+1}: {h}" for i, h in enumerate(hypotheses))
        results_text = "\n".join(
            f"Branch {r.get('branch_id','?')}: "
            f"success={r.get('success')}, "
            f"output_excerpt={str(r.get('output',''))[:300]}"
            for r in results
        )
        feedback_text = ""
        if feedback:
            feedback_text = "\n\nReviewer feedback to address in this revision:\n" + "\n".join(
                f"- {f}" for f in feedback
            )

        prompt = (
            self.system_prompt + "\n\n"
            f"Research topic: {topic}\n\n"
            f"Hypotheses tested:\n{hyp_text}\n\n"
            f"Experiment results summary:\n{results_text}\n"
            + feedback_text
            + "\n\nWrite the full LaTeX paper body (no \\documentclass preamble — "
            "just sections from \\section{{Abstract}} onward). "
            "Include a TikZ figure showing the system architecture. "
            "All numbers must appear exactly as they appear in the results summary."
            f"\nThis is revision round {revision_round}."
        )
        return llm.complete(prompt)

    @staticmethod
    def _extract_title(body: str, fallback: str) -> str:
        m = re.search(r"\\title\{([^}]+)\}", body)
        if m:
            return m.group(1)
        # Use first \section content as title
        m2 = re.search(r"\\section\{([^}]+)\}", body)
        if m2:
            return m2.group(1)
        return fallback.title()

    @staticmethod
    def _extract_abstract(body: str) -> str:
        m = re.search(
            r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.DOTALL
        )
        if m:
            return m.group(1).strip()
        return "This paper was auto-generated by AutoResearch v2 (HRDE)."

    # ------------------------------------------------------------------
    # PDF compilation
    # ------------------------------------------------------------------

    def _compile_pdf(self, latex_source: str, topic: str) -> Optional[str]:
        """Compile LaTeX to PDF using pdflatex. Returns output path or None."""
        out_dir = Path(self.config.get("output", {}).get("base_dir", "./output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^\w]+", "_", topic)[:50]
        tex_path = out_dir / f"{safe_name}.tex"

        with open(tex_path, "w") as f:
            f.write(latex_source)

        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", str(tex_path)],
                cwd=str(out_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                pdf_path = str(tex_path.with_suffix(".pdf"))
                self.log(f"PDF compiled → {pdf_path}")
                return pdf_path
            else:
                self.log(
                    f"pdflatex failed: {result.stderr[:300]}", level="warning"
                )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            self.log(f"PDF compilation unavailable: {exc}", level="warning")
        return None

    # ------------------------------------------------------------------
    # Revision entry-point (convenience wrapper)
    # ------------------------------------------------------------------

    async def revise(
        self,
        previous_source: str,
        feedback: List[str],
        revision_round: int,
        topic: str,
        hypotheses: List[str],
        experiment_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Helper that re-runs _execute with feedback for iterative revision."""
        return await self.run(
            topic=topic,
            hypotheses=hypotheses,
            experiment_results=experiment_results,
            reviewer_feedback=feedback,
            revision_round=revision_round,
        )
