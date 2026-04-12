"""
main.py — AutoResearch v2 (HRDE) CLI Entry Point

5-Loop Operational Workflow (proposal-aligned):
  Loop 1 · Meta-Loop        — domain detection + optimal team building
  Loop 2 · Ideation Loop    — evolutionary hypothesis generation (ArXiv + OpenAlex)
  Loop 3 · Lab Loop         — parallel sandbox experiments + self-healing code
  Loop 4 · Synthesis Loop   — LaTeX / TikZ manuscript generation
  Loop 5 · Review Loop      — ToM adversarial council + self-correction

Usage
-----
  python main.py --topic "Novel LLM pruning techniques" --branches 3
  python main.py --topic "..." --effort pro --branches 5 \
                 --review_council "skeptic,engineer,visionary,statistician,pragmatist" \
                 --data_source "arxiv,openalex"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import zipfile
from pathlib import Path

import yaml


def _setup_logging(level: str = "INFO", log_dir: str = "./logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "autoresearch_v2.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )


def _load_config(config_path: str = "autoresearch_v2/config.yaml") -> dict:
    try:
        with open(config_path) as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logging.warning("Config file %r not found — using defaults.", config_path)
        return {}


def _merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.effort:
        config["effort"] = args.effort
    if args.branches:
        config.setdefault("compute", {})["branches"] = args.branches
    if args.review_council:
        config["review_council"] = args.review_council
    if args.output_dir:
        config.setdefault("output", {})["base_dir"] = args.output_dir
    if args.data_source:
        config["data_source"] = args.data_source
    return config


async def run_pipeline(config: dict, topic: str) -> dict:
    from autoresearch_v2.agents.coder import CoderAgent
    from autoresearch_v2.agents.reviewer import ReviewerCouncil
    from autoresearch_v2.agents.writer import WriterAgent
    from autoresearch_v2.core.graph_manager import GraphManager
    from autoresearch_v2.core.kernel import HyperKernel
    from autoresearch_v2.core.memory import ResearchMemory
    from autoresearch_v2.core.meta_loop import MetaLoop
    from autoresearch_v2.discovery.evaluator import Evaluator
    from autoresearch_v2.discovery.ideator import IdeatorAgent
    from autoresearch_v2.discovery.tree_search import TreeSearch

    log_dir = config.get("logging", {}).get("log_dir", "./logs")
    out_dir = config.get("output", {}).get("base_dir", "./output")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_id = f"{topic[:30].replace(' ', '_')}_{int(time.time())}"
    logger = logging.getLogger("pipeline")

    # ── Loop 1 · Meta-Loop ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loop 1 · Meta-Loop: building optimal research team")
    logger.info("=" * 60)
    meta = MetaLoop(config)
    team_config = meta.build_team(topic)
    n_branches: int = config.get("compute", {}).get("branches", 3)
    n_hypotheses: int = team_config.get("hypothesis_count", n_branches * 3)

    kernel = HyperKernel(config, log_dir=log_dir)
    memory = ResearchMemory(config)
    coder    = CoderAgent(config, log_dir=log_dir)
    writer   = WriterAgent(config, log_dir=log_dir)
    reviewer = ReviewerCouncil(config, log_dir=log_dir)
    ideator  = IdeatorAgent(config, log_dir=log_dir)
    evaluator = Evaluator(config)
    tree = TreeSearch(config, coder_agent=coder, evaluator_agent=evaluator)

    for name, agent in [("coder", coder), ("writer", writer),
                         ("reviewer", reviewer), ("ideator", ideator)]:
        kernel.register_agent(name, agent)
    meta.apply_to_agents(team_config, {
        "coder": coder, "writer": writer,
        "reviewer": reviewer, "ideator": ideator,
    })
    kernel.start()

    gm = GraphManager(config)
    results: dict = {}

    # ── Loop 2 · Ideation Loop ─────────────────────────────────────────────
    async def ideation_loop() -> None:
        logger.info("=" * 60)
        logger.info("Loop 2 · Ideation Loop: generating hypotheses from literature")
        logger.info("=" * 60)
        data_source = config.get("data_source", "arxiv")
        idea_result = await ideator.run(
            topic=topic, n_hypotheses=n_hypotheses, data_source=data_source
        )
        results["hypotheses"]   = idea_result.get("hypotheses", [topic])
        results["arxiv_papers"] = idea_result.get("arxiv_papers", [])
        memory.store(
            "Hypotheses for '{}': ".format(topic) + "; ".join(results["hypotheses"][:3]),
            {"type": "hypothesis", "topic": topic},
        )
        logger.info("Loop 2 complete: %d hypotheses generated.", len(results["hypotheses"]))

    # ── Loop 3 · Lab Loop ──────────────────────────────────────────────────
    async def lab_loop() -> None:
        logger.info("=" * 60)
        logger.info("Loop 3 · Lab Loop: parallel experiments + self-healing code")
        logger.info("=" * 60)
        hyps = results.get("hypotheses", [topic])[:n_branches]
        tree_result = await tree.run(hyps)
        results["tree"] = tree_result
        for branch in tree_result.get("branches", []):
            if branch.result:
                (kernel.record_success("coder") if branch.result.get("success")
                 else kernel.record_failure("coder"))
                if not branch.result.get("success"):
                    memory.store_failure("coder", branch.hypothesis,
                                         branch.result.get("output", "")[:300])
        n_done = sum(1 for b in tree_result.get("branches", []) if b.status == "done")
        logger.info("Loop 3 complete: %d/%d branches succeeded.",
                    n_done, len(tree_result.get("branches", [])))

    # ── Loop 4 · Synthesis Loop ────────────────────────────────────────────
    async def synthesis_loop() -> None:
        logger.info("=" * 60)
        logger.info("Loop 4 · Synthesis Loop: LaTeX + TikZ manuscript generation")
        logger.info("=" * 60)
        branches = results.get("tree", {}).get("branches", [])
        exp_results = []
        for b in branches:
            sc = evaluator.score(b) if b.result else {}
            exp_results.append({
                "branch_id": b.branch_id, "hypothesis": b.hypothesis,
                "success": b.status == "done",
                "output": (b.result or {}).get("output", ""),
                "code":   (b.result or {}).get("code", ""),
                "novelty_score":      sc.get("novelty", 0.0),
                "significance_score": sc.get("significance", 0.0),
                "surprise":           sc.get("surprise", 0.0),
                "rigorous":           sc.get("rigorous", 0.0),
            })
        write_result = await writer.run(
            topic=topic, hypotheses=results.get("hypotheses", []),
            experiment_results=[{"branch_id": b.branch_id, "hypothesis": b.hypothesis, "success": b.status == "done", "output": (b.result or {}).get("output", ""), "code": (b.result or {}).get("code", "")} for b in results.get("tree", {}).get("branches", [])],
        )
        results["manuscript"] = write_result
        logger.info("Loop 4 complete: manuscript drafted.")

    # ── Loop 5 · Review Loop ───────────────────────────────────────────────
    async def review_loop() -> None:
        council = config.get("review_council",
                             "skeptic,engineer,visionary,statistician,pragmatist")
        n_council = len(council.split(","))
        threshold = config.get("acceptance_threshold", max(1, n_council - 1))
        logger.info("=" * 60)
        logger.info("Loop 5 · Review Loop: ToM adversarial council "
                    "(need %d/%d approvals)", threshold, n_council)
        logger.info("=" * 60)
        initial_src = results.get("manuscript", {}).get("latex_source", "")
        review_result = await reviewer.debate_loop(
            writer_agent=writer,
            initial_manuscript=initial_src,
            topic=topic,
            hypotheses=results.get("hypotheses", []),
            experiment_results=[{"branch_id": b.branch_id, "hypothesis": b.hypothesis, "success": b.status == "done", "output": (b.result or {}).get("output", ""), "code": (b.result or {}).get("code", "")} for b in results.get("tree", {}).get("branches", [])],
        )
        results["review"] = review_result
        accepted = review_result.get("accepted", False)
        logger.info("Loop 5 complete: paper %s (surprise=%.2f, rigorous=%.2f).",
                    "ACCEPTED" if accepted else "NOT ACCEPTED",
                    review_result.get("mean_surprise", 0.0),
                    review_result.get("mean_rigorous", 0.0))
        for fb in review_result.get("consolidated_feedback", [])[:5]:
            memory.store(fb, {"type": "lesson", "topic": topic, "phase": "review"})

    gm.add_task("ideation_loop", ideation_loop)
    gm.add_task("lab_loop",      lab_loop,      depends_on=["ideation_loop"])
    gm.add_task("synthesis_loop", synthesis_loop, depends_on=["lab_loop"])
    gm.add_task("review_loop",   review_loop,   depends_on=["synthesis_loop"])
    await gm.run()
    kernel.stop()

    # ── Zero-Trust Verification ────────────────────────────────────────────
    branches = results.get("tree", {}).get("branches", [])
    execution_logs = {b.branch_id: (b.result or {}).get("output", "") for b in branches}
    final_manuscript = (
        results.get("review", {}).get("final_manuscript")
        or results.get("manuscript", {}).get("latex_source", "")
    )
    verification = evaluator.verify_manuscript(final_manuscript, execution_logs)
    logger.info("Zero-Trust verification: %.1f%% of numeric claims traceable.",
                verification["pass_rate"] * 100)

    # ── Write outputs ──────────────────────────────────────────────────────
    safe_topic = topic[:40].replace(" ", "_").replace("/", "-")
    tex_path  = Path(out_dir) / f"{safe_topic}.tex"
    meta_path = Path(out_dir) / f"{safe_topic}_meta.json"
    tex_path.write_text(final_manuscript, encoding="utf-8")

    council_cfg = config.get("review_council",
                              "skeptic,engineer,visionary,statistician,pragmatist")
    meta_data = {
        "run_id": run_id, "topic": topic,
        "effort": config.get("effort", "standard"),
        "domain": team_config.get("domain", "general"),
        "hypotheses": results.get("hypotheses", []),
        "n_branches": len(branches),
        "n_succeeded": sum(1 for b in branches if b.status == "done"),
        "accepted": results.get("review", {}).get("accepted", False),
        "mean_surprise": results.get("review", {}).get("mean_surprise", 0.0),
        "mean_rigorous": results.get("review", {}).get("mean_rigorous", 0.0),
        "verification": verification,
        "kernel_patches": kernel.patch_count,
        "data_source": config.get("data_source", "arxiv"),
        "review_council": council_cfg,
    }
    meta_path.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")

    if config.get("output", {}).get("generate_pdf", False):
        _compile_pdf(tex_path)
    zip_path: str | None = None
    if config.get("output", {}).get("zip_source", False):
        zip_path = _create_source_zip(out_dir, safe_topic, branches)

    logger.info("=" * 60)
    logger.info("AutoResearch v2 complete.")
    logger.info("  LaTeX : %s", tex_path)
    logger.info("  Meta  : %s", meta_path)
    if zip_path:
        logger.info("  ZIP   : %s", zip_path)
    logger.info("=" * 60)
    return {"tex_path": str(tex_path), "meta_path": str(meta_path),
            "zip_path": zip_path, "meta": meta_data}


def _compile_pdf(tex_path: Path) -> None:
    import subprocess
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(tex_path.name)],
            cwd=tex_path.parent, capture_output=True, timeout=120, check=True,
        )
        logging.getLogger("pipeline").info("PDF compiled: %s.pdf", tex_path.stem)
    except Exception as exc:
        logging.getLogger("pipeline").warning("pdflatex failed: %s", exc)


def _create_source_zip(out_dir: str, safe_topic: str, branches: list) -> str:
    zip_path = Path(out_dir) / f"{safe_topic}_source.zip"
    pkg_root = Path(__file__).parent / "autoresearch_v2"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if pkg_root.exists():
            for py_file in pkg_root.rglob("*.py"):
                zf.write(py_file, py_file.relative_to(pkg_root.parent))
        for branch in branches:
            if branch.result and branch.result.get("code"):
                zf.writestr(f"experiments/{branch.branch_id}_experiment.py",
                            branch.result["code"])
    return str(zip_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autoresearch_v2",
        description="AutoResearch v2 (HRDE) — Self-Evolving Autonomous Research Framework",
    )
    p.add_argument("--topic", required=True, help="Research topic or question")
    p.add_argument("--effort", choices=["minimal", "standard", "pro", "max"],
                   default="standard", help="Compute budget level (default: standard)")
    p.add_argument("--branches", type=int, default=3,
                   help="Number of parallel hypothesis branches (default: 3)")
    p.add_argument("--review_council",
                   default="skeptic,engineer,visionary,statistician,pragmatist",
                   help="Comma-separated reviewer personas (default: all 5)")
    p.add_argument("--data_source", default="arxiv",
                   help="Comma-separated literature sources: arxiv,openalex (default: arxiv)")
    p.add_argument("--output_dir", default="./output",
                   help="Output directory (default: ./output)")
    p.add_argument("--config", default="autoresearch_v2/config.yaml",
                   help="Path to config.yaml")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _load_config(args.config)
    config = _merge_cli_overrides(config, args)
    log_level = config.get("logging", {}).get("level", "INFO")
    log_dir   = config.get("logging", {}).get("log_dir", "./logs")
    _setup_logging(log_level, log_dir)
    logger = logging.getLogger("main")
    logger.info("AutoResearch v2 (HRDE) starting.")
    logger.info("  topic          = %s", args.topic)
    logger.info("  effort         = %s", config.get("effort"))
    logger.info("  branches       = %s", config.get("compute", {}).get("branches", 3))
    logger.info("  review_council = %s", config.get("review_council"))
    logger.info("  data_source    = %s", config.get("data_source", "arxiv"))
    summary = asyncio.run(run_pipeline(config, args.topic))
    logger.info("Done. Output: %s", summary.get("tex_path"))


if __name__ == "__main__":
    main()
