"""
main.py — AutoResearch v2 (HRDE) CLI Entry Point

Usage::

    python main.py --topic "Novel LLM pruning techniques" --branches 3

    # Full options:
    python main.py \\
        --topic "Self-correcting LLM architectures using ToM" \\
        --effort pro \\
        --branches 5 \\
        --review_council "skeptic,engineer,visionary" \\
        --config autoresearch_v2/config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    level_name: str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = Path(log_cfg.get("log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "autoresearch_v2.log", mode="a"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        # Try relative to this file's directory
        p = Path(__file__).parent / "autoresearch_v2" / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Override config keys from CLI arguments."""
    if args.branches:
        config.setdefault("compute", {})["branches"] = args.branches
    if args.effort:
        config["effort"] = args.effort
    if args.review_council:
        config["review_council"] = args.review_council
    if args.output_dir:
        config.setdefault("output", {})["base_dir"] = args.output_dir
    return config


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(topic: str, config: dict) -> dict:
    """Execute the full AutoResearch v2 pipeline."""
    from autoresearch_v2.agents.coder import CoderAgent
    from autoresearch_v2.agents.reviewer import ReviewerCouncil
    from autoresearch_v2.agents.writer import WriterAgent
    from autoresearch_v2.core.graph_manager import GraphManager
    from autoresearch_v2.core.kernel import HyperKernel
    from autoresearch_v2.core.memory import ResearchMemory
    from autoresearch_v2.discovery.evaluator import Evaluator
    from autoresearch_v2.discovery.ideator import IdeatorAgent
    from autoresearch_v2.discovery.tree_search import TreeSearch
    from autoresearch_v2.env.tools import PlottingTool

    log_dir = config.get("logging", {}).get("log_dir", "./logs")
    logger = logging.getLogger("main")

    logger.info("=" * 60)
    logger.info("AutoResearch v2 (HRDE) — Starting run")
    logger.info("Topic: %s", topic)
    logger.info("Effort: %s | Branches: %d", config.get("effort"), config["compute"]["branches"])
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Initialise components
    # ------------------------------------------------------------------
    kernel = HyperKernel(config, log_dir=log_dir)
    memory = ResearchMemory(config)
    evaluator = Evaluator(config)
    coder = CoderAgent(config, log_dir=log_dir)
    writer = WriterAgent(config, log_dir=log_dir)
    reviewer = ReviewerCouncil(config, log_dir=log_dir)
    ideator = IdeatorAgent(config, log_dir=log_dir)
    tree_search = TreeSearch(config, coder_agent=coder, evaluator_agent=evaluator)
    plotter = PlottingTool(config)

    # Register agents with kernel for monitoring/patching
    for agent in [coder, writer, reviewer, ideator]:
        kernel.register_agent(agent.name, agent)

    # ------------------------------------------------------------------
    # Phase 1: Ideation (DOG task: "ideation")
    # ------------------------------------------------------------------
    logger.info("[Phase 1] Ideation — generating hypotheses from ArXiv…")
    n_branches: int = config["compute"]["branches"]
    ideation_result = await ideator.run(topic=topic, n_hypotheses=n_branches)
    hypotheses: list[str] = ideation_result.get("hypotheses", [])[:n_branches]

    if not hypotheses:
        logger.error("Ideation produced no hypotheses. Aborting.")
        sys.exit(1)

    logger.info("[Phase 1] Generated %d hypotheses:", len(hypotheses))
    for i, h in enumerate(hypotheses, 1):
        logger.info("  H%d: %s", i, h)
        memory.store(h, metadata={"type": "hypothesis", "topic": topic})

    # ------------------------------------------------------------------
    # Phase 2: Agentic Tree Search (parallel experiment branches)
    # ------------------------------------------------------------------
    logger.info("[Phase 2] Agentic Tree Search — running %d branches…", len(hypotheses))
    tree_result = await tree_search.run(hypotheses)

    branches = tree_result.get("branches", [])
    best_branch = tree_result.get("best_branch")
    grafted = tree_result.get("grafted_result")

    experiment_results = [
        b.result for b in branches if b.result is not None
    ]

    logger.info("[Phase 2] Branch summary:")
    for b in branches:
        logger.info(
            "  %s status=%-7s score=%.3f hypothesis=%s",
            b.branch_id, b.status, b.combined_score, b.hypothesis[:60],
        )

    if best_branch:
        logger.info("[Phase 2] Best branch: %s (score=%.3f)", best_branch.branch_id, best_branch.combined_score)

    # Plot branch scores
    scored_branches = [b for b in branches if b.combined_score > 0]
    if scored_branches:
        plotter.bar_chart(
            labels=[b.branch_id for b in scored_branches],
            values=[b.combined_score for b in scored_branches],
            title=f"Branch Scores — {topic[:40]}",
            filename="branch_scores.png",
        )

    # ------------------------------------------------------------------
    # Phase 3: Manuscript Writing
    # ------------------------------------------------------------------
    logger.info("[Phase 3] Writing manuscript…")
    writer_result = await writer.run(
        topic=topic,
        hypotheses=hypotheses,
        experiment_results=experiment_results,
        revision_round=0,
    )
    latex_source: str = writer_result.get("latex_source", "")
    logger.info("[Phase 3] Manuscript draft: %d chars.", len(latex_source))

    # ------------------------------------------------------------------
    # Phase 4: ToM Adversarial Review (debate loop)
    # ------------------------------------------------------------------
    logger.info("[Phase 4] ToM Reviewer Council — adversarial review…")
    review_result = await reviewer.debate_loop(
        writer_agent=writer,
        initial_manuscript=latex_source,
        topic=topic,
        hypotheses=hypotheses,
        experiment_results=experiment_results,
    )
    final_manuscript: str = review_result.get("final_manuscript", latex_source)
    accepted: bool = review_result.get("accepted", False)
    logger.info(
        "[Phase 4] Paper %s after %d review round(s).",
        "ACCEPTED" if accepted else "NOT ACCEPTED (max rounds reached)",
        review_result.get("round_num", 1),
    )

    # ------------------------------------------------------------------
    # Phase 5: Epistemic Verification (Zero-Trust)
    # ------------------------------------------------------------------
    logger.info("[Phase 5] Zero-Trust Epistemic Verification…")
    execution_logs = {
        b.branch_id: (b.result or {}).get("output", "")
        for b in branches if b.result
    }
    verification = evaluator.verify_manuscript(final_manuscript, execution_logs)
    logger.info(
        "[Phase 5] Verification: %.1f%% claims traceable (%d/%d).",
        verification["pass_rate"] * 100,
        len(verification["verified"]),
        verification["total_claims"],
    )
    if verification["unverified"]:
        logger.warning("[Phase 5] Unverified claims: %s", verification["unverified"])

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    out_dir = Path(config.get("output", {}).get("base_dir", "./output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save LaTeX
    safe_topic = "".join(c if c.isalnum() else "_" for c in topic)[:50]
    timestamp = int(time.time())
    tex_path = out_dir / f"{safe_topic}_{timestamp}.tex"
    tex_path.write_text(final_manuscript)
    logger.info("[Output] LaTeX saved → %s", tex_path)

    # Save metadata
    meta = {
        "topic": topic,
        "hypotheses": hypotheses,
        "branches": [
            {
                "id": b.branch_id,
                "hypothesis": b.hypothesis,
                "status": b.status,
                "novelty": b.novelty_score,
                "significance": b.significance_score,
                "score": b.combined_score,
            }
            for b in branches
        ],
        "accepted": accepted,
        "verification_pass_rate": verification["pass_rate"],
        "kernel_patches": kernel.patch_count,
    }
    meta_path = out_dir / f"{safe_topic}_{timestamp}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("[Output] Metadata saved → %s", meta_path)

    # Zip source code if requested
    zip_path: str | None = None
    if config.get("output", {}).get("zip_source", True):
        zip_path = str(out_dir / f"{safe_topic}_{timestamp}_source.zip")
        _create_source_zip(zip_path, out_dir, tex_path)
        logger.info("[Output] Source zip → %s", zip_path)

    # Save patch history
    kernel.save_patch_history(str(Path(log_dir) / "patch_history.yaml"))

    logger.info("=" * 60)
    logger.info("AutoResearch v2 (HRDE) — Run complete!")
    logger.info("Output directory: %s", out_dir)
    logger.info("=" * 60)

    return {
        "topic": topic,
        "hypotheses": hypotheses,
        "branches": branches,
        "accepted": accepted,
        "tex_path": str(tex_path),
        "meta_path": str(meta_path),
        "zip_path": zip_path,
        "verification": verification,
    }


def _create_source_zip(zip_path: str, out_dir: Path, tex_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include the LaTeX source
        zf.write(tex_path, tex_path.name)
        # Include all Python files from autoresearch_v2/
        pkg_root = Path(__file__).parent / "autoresearch_v2"
        for py_file in pkg_root.rglob("*.py"):
            arcname = py_file.relative_to(pkg_root.parent)
            zf.write(py_file, str(arcname))
        # Include config
        cfg_file = pkg_root / "config.yaml"
        if cfg_file.exists():
            zf.write(cfg_file, "autoresearch_v2/config.yaml")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="autoresearch_v2",
        description="AutoResearch v2 (HRDE) — Self-evolving autonomous research framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --topic "Novel LLM pruning techniques" --branches 3
  python main.py --topic "Transformers for fluid dynamics" --effort pro --branches 5
        """,
    )
    parser.add_argument(
        "--topic",
        required=True,
        type=str,
        help="Research topic for hypothesis generation and experimentation.",
    )
    parser.add_argument(
        "--branches",
        type=int,
        default=None,
        help="Number of parallel hypothesis branches (overrides config).",
    )
    parser.add_argument(
        "--effort",
        choices=["minimal", "standard", "pro", "max"],
        default=None,
        help="Effort level controlling compute budget (overrides config).",
    )
    parser.add_argument(
        "--review_council",
        type=str,
        default=None,
        help='Comma-separated reviewer personas, e.g. "skeptic,engineer,visionary".',
    )
    parser.add_argument(
        "--config",
        type=str,
        default="autoresearch_v2/config.yaml",
        help="Path to config.yaml (default: autoresearch_v2/config.yaml).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    config = _merge_cli_overrides(config, args)
    _setup_logging(config)

    # Ensure branches key exists
    config.setdefault("compute", {}).setdefault("branches", 3)

    asyncio.run(run_pipeline(topic=args.topic, config=config))


if __name__ == "__main__":
    main()
