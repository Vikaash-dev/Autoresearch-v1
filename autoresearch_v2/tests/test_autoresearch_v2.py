"""
tests/test_autoresearch_v2.py — Unit tests for AutoResearch v2 (HRDE)

Tests cover the core components without requiring real LLM API keys,
Docker, or ChromaDB installations — all external I/O uses the built-in
fallback / stub paths.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Shared minimal config
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "llm": {"default_model": "gpt-4o", "temperature": 0.7, "max_tokens": 512},
    "compute": {
        "branches": 2,
        "max_branch_workers": 2,
        "prune_after_pct": 0.5,
        "prune_bottom_pct": 0.5,
        # max_debug_retries=3: loop fires only after >3 failures (i.e. 4th failure)
        "max_debug_retries": 3,
        "kernel_check_interval": 1,
        "sandbox_timeout": 10,
    },
    "memory": {
        "persist_directory": "/tmp/test_research_memory",
        "collection_name": "test_col",
        "embedding_model": "all-MiniLM-L6-v2",
    },
    "output": {
        "base_dir": "/tmp/ar_test_output",
        "generate_pdf": False,
        "zip_source": False,
    },
    "logging": {"level": "WARNING", "log_dir": "/tmp/ar_test_logs"},
    "sandbox": {"enabled": False},
    "review_council": "skeptic,engineer,visionary",
    "acceptance_threshold": 2,
    "review_rounds": 1,
    "ideation": {
        "arxiv_max_results": 2,
        "hypothesis_count": 2,
        "crossover_rate": 0.5,
        "mutation_rate": 0.0,
    },
}


# ===========================================================================
# HyperKernel
# ===========================================================================

class TestHyperKernel:
    def _make_kernel(self) -> Any:
        from autoresearch_v2.core.kernel import HyperKernel
        return HyperKernel(BASE_CONFIG, log_dir="/tmp/ar_test_logs")

    def test_register_and_read_agent(self) -> None:
        kernel = self._make_kernel()
        mock_agent = MagicMock()
        mock_agent.system_prompt = "Original prompt"
        kernel.register_agent("test_agent", mock_agent)
        assert kernel.read_agent_prompt("test_agent") == "Original prompt"

    def test_write_agent_prompt(self) -> None:
        kernel = self._make_kernel()
        mock_agent = MagicMock()
        mock_agent.system_prompt = "Original prompt"
        kernel.register_agent("test_agent", mock_agent)
        result = kernel.write_agent_prompt("test_agent", "New prompt")
        assert result is True
        assert mock_agent.system_prompt == "New prompt"
        assert kernel.patch_count == 1

    def test_failure_tracking_below_threshold(self) -> None:
        """3 failures should NOT trigger the loop (threshold > 3)."""
        kernel = self._make_kernel()
        mock_agent = MagicMock()
        mock_agent.system_prompt = "prompt"
        kernel.register_agent("coder", mock_agent)
        for _ in range(3):
            kernel.record_failure("coder")
        assert not kernel._detect_reasoning_loop("coder")

    def test_failure_tracking_at_threshold(self) -> None:
        """4 failures SHOULD trigger the loop (> max_retries=3)."""
        kernel = self._make_kernel()
        mock_agent = MagicMock()
        mock_agent.system_prompt = "prompt"
        kernel.register_agent("coder", mock_agent)
        for _ in range(4):
            kernel.record_failure("coder")
        assert kernel._detect_reasoning_loop("coder")

    def test_success_resets_failure_count(self) -> None:
        kernel = self._make_kernel()
        for _ in range(4):
            kernel.record_failure("coder")
        kernel.record_success("coder")
        assert kernel._failure_counts.get("coder", 0) == 0

    def test_read_unknown_agent_returns_none(self) -> None:
        kernel = self._make_kernel()
        assert kernel.read_agent_prompt("nonexistent") is None

    def test_write_unknown_agent_returns_false(self) -> None:
        kernel = self._make_kernel()
        assert kernel.write_agent_prompt("ghost", "prompt") is False

    def test_log_scanning_does_not_double_count(self, tmp_path: Any) -> None:
        """monitor_once() must not re-count lines already processed."""
        from autoresearch_v2.core.kernel import HyperKernel

        cfg = {**BASE_CONFIG, "logging": {"log_dir": str(tmp_path)}}
        cfg["compute"] = {**BASE_CONFIG["compute"], "kernel_check_interval": 1}
        kernel = HyperKernel(cfg, log_dir=str(tmp_path))

        log_file = tmp_path / "coder.log"
        log_file.write_text("[ERROR] [coder] execution fail\n")

        kernel.monitor_once()  # first scan — counts 1 error
        count_after_first = kernel._failure_counts.get("coder", 0)

        kernel.monitor_once()  # second scan — same file, no new lines
        count_after_second = kernel._failure_counts.get("coder", 0)

        assert count_after_first == count_after_second, (
            "monitor_once() must not re-count already-read lines"
        )

    def test_start_and_stop_thread(self) -> None:
        kernel = self._make_kernel()
        kernel.start()
        assert kernel._monitor_thread is not None
        assert kernel._monitor_thread.is_alive()
        kernel.stop()
        assert not kernel._running


# ===========================================================================
# GraphManager
# ===========================================================================

class TestGraphManager:
    def _make_gm(self) -> Any:
        from autoresearch_v2.core.graph_manager import GraphManager
        return GraphManager(BASE_CONFIG)

    def test_simple_sequential_run(self) -> None:
        gm = self._make_gm()
        results = []

        async def task_a() -> str:
            results.append("a")
            return "result_a"

        async def task_b() -> str:
            results.append("b")
            return "result_b"

        gm.add_task("a", task_a)
        gm.add_task("b", task_b, depends_on=["a"])

        output = asyncio.run(gm.run())
        assert results == ["a", "b"]
        assert output["a"] == "result_a"
        assert output["b"] == "result_b"

    def test_failed_task_spawns_recovery(self) -> None:
        gm = self._make_gm()
        recovery_ran = []

        async def failing_task() -> None:
            raise RuntimeError("simulated failure")

        async def recovery_task() -> str:
            recovery_ran.append(True)
            return "recovered"

        gm.add_task("main_task", failing_task, recovery_task="recovery")
        gm.add_task("recovery", recovery_task)

        asyncio.run(gm.run())
        assert recovery_ran, "Recovery task should have been spawned."

    def test_status_report(self) -> None:
        gm = self._make_gm()

        async def noop() -> None:
            pass

        gm.add_task("x", noop)
        asyncio.run(gm.run())
        report = gm.status_report()
        assert report["x"] == "DONE"

    def test_cycle_detection(self) -> None:
        from autoresearch_v2.core.graph_manager import GraphManager
        gm = GraphManager(BASE_CONFIG)

        async def dummy() -> None:
            pass

        gm.add_task("a", dummy)
        gm.add_task("b", dummy, depends_on=["a"])
        gm.add_task("c", dummy, depends_on=["b"])
        gm._graph.add_edge("c", "a")  # force cycle

        with pytest.raises(ValueError, match="cycle"):
            asyncio.run(gm.run())


# ===========================================================================
# ResearchMemory
# ===========================================================================

class TestResearchMemory:
    def _make_memory(self) -> Any:
        from autoresearch_v2.core.memory import ResearchMemory
        mem = ResearchMemory(BASE_CONFIG)
        mem._use_chroma = False  # always use fallback
        return mem

    def test_store_and_query_fallback(self) -> None:
        mem = self._make_memory()
        mem.store("LLM pruning reduces model size by 50%", {"type": "result"})
        results = mem.query("LLM pruning", n_results=3)
        assert any("pruning" in r["text"].lower() for r in results)

    def test_get_skills_respects_where_filter(self) -> None:
        """get_skills must return only 'skill' type entries, not other types."""
        mem = self._make_memory()
        mem.store("sparse data technique", {"type": "skill", "name": "handle_sparse"})
        mem.store("sparse neural network architecture", {"type": "result"})

        skills = mem.get_skills("sparse")
        assert all(s["metadata"].get("type") == "skill" for s in skills), (
            "get_skills must filter by type=skill"
        )

    def test_fallback_where_eq_filter(self) -> None:
        from autoresearch_v2.core.memory import ResearchMemory
        mem = ResearchMemory(BASE_CONFIG)
        mem._use_chroma = False
        mem.store("alpha", {"category": "A"})
        mem.store("alpha beta", {"category": "B"})

        results = mem.query("alpha", where={"category": {"$eq": "A"}})
        assert all(r["metadata"]["category"] == "A" for r in results)

    def test_fallback_where_ne_filter(self) -> None:
        from autoresearch_v2.core.memory import ResearchMemory
        mem = ResearchMemory(BASE_CONFIG)
        mem._use_chroma = False
        mem.store("alpha", {"category": "A"})
        mem.store("alpha beta", {"category": "B"})

        results = mem.query("alpha", where={"category": {"$ne": "A"}})
        assert all(r["metadata"]["category"] != "A" for r in results)

    def test_store_failure_and_count(self) -> None:
        mem = self._make_memory()
        initial = mem.count()
        mem.store_failure("coder", "context", "error msg")
        assert mem.count() == initial + 1


# ===========================================================================
# CoderAgent
# ===========================================================================

class TestCoderAgent:
    def _make_coder(self) -> Any:
        from autoresearch_v2.agents.coder import CoderAgent
        return CoderAgent(BASE_CONFIG, log_dir="/tmp/ar_test_logs")

    def test_strip_code_fences(self) -> None:
        from autoresearch_v2.agents.coder import CoderAgent
        raw = "```python\nprint('hello')\n```"
        assert CoderAgent._strip_code_fences(raw) == "print('hello')"

    def test_strip_no_fences(self) -> None:
        from autoresearch_v2.agents.coder import CoderAgent
        raw = "print('hello')"
        assert CoderAgent._strip_code_fences(raw) == "print('hello')"

    def test_extract_metrics(self) -> None:
        coder = self._make_coder()
        output = "accuracy: 0.9523\nloss=0.0341\nf1_score: 0.88"
        metrics = coder.extract_metrics(output)
        assert abs(metrics.get("accuracy", 0) - 0.9523) < 1e-4
        assert abs(metrics.get("loss", 0) - 0.0341) < 1e-4

    def test_execute_success(self) -> None:
        coder = self._make_coder()
        result = asyncio.run(
            coder.run(hypothesis="Test hypothesis", branch_id="B1")
        )
        assert "success" in result
        assert "code" in result


# ===========================================================================
# WriterAgent
# ===========================================================================

class TestWriterAgent:
    def _make_writer(self) -> Any:
        from autoresearch_v2.agents.writer import WriterAgent
        return WriterAgent(BASE_CONFIG, log_dir="/tmp/ar_test_logs")

    def test_execute_produces_latex(self) -> None:
        writer = self._make_writer()
        result = asyncio.run(writer.run(
            topic="LLM pruning",
            hypotheses=["H1: Pruning reduces latency."],
            experiment_results=[{
                "branch_id": "B1", "success": True, "output": "accuracy: 0.92",
            }],
        ))
        assert "latex_source" in result
        assert "\\documentclass" in result["latex_source"]

    def test_extract_abstract(self) -> None:
        from autoresearch_v2.agents.writer import WriterAgent
        body = "text \\begin{abstract}The abstract.\\end{abstract} more"
        assert WriterAgent._extract_abstract(body) == "The abstract."

    def test_extract_abstract_missing(self) -> None:
        from autoresearch_v2.agents.writer import WriterAgent
        body = "no abstract here"
        abstract = WriterAgent._extract_abstract(body)
        assert "AutoResearch" in abstract  # falls back to default string

    def test_extract_title_from_section(self) -> None:
        from autoresearch_v2.agents.writer import WriterAgent
        body = "\\section{My Great Paper}"
        assert WriterAgent._extract_title(body, "fallback") == "My Great Paper"

    def test_extract_title_fallback(self) -> None:
        from autoresearch_v2.agents.writer import WriterAgent
        assert WriterAgent._extract_title("no title here", "My Fallback Topic") == "My Fallback Topic"


# ===========================================================================
# ReviewerCouncil
# ===========================================================================

class TestReviewerCouncil:
    def _make_reviewer(self) -> Any:
        from autoresearch_v2.agents.reviewer import ReviewerCouncil
        return ReviewerCouncil(BASE_CONFIG, log_dir="/tmp/ar_test_logs")

    def test_extract_verdict_accept(self) -> None:
        from autoresearch_v2.agents.reviewer import ReviewerCouncil
        assert ReviewerCouncil._extract_verdict("Good paper.\nVERDICT: ACCEPT") == "ACCEPT"

    def test_extract_verdict_reject(self) -> None:
        from autoresearch_v2.agents.reviewer import ReviewerCouncil
        assert ReviewerCouncil._extract_verdict("Major issues.\nVERDICT: REJECT") == "REJECT"

    def test_extract_verdict_minor(self) -> None:
        from autoresearch_v2.agents.reviewer import ReviewerCouncil
        assert ReviewerCouncil._extract_verdict("VERDICT: MINOR_REVISION") == "MINOR_REVISION"

    def test_extract_verdict_fallback_reject(self) -> None:
        from autoresearch_v2.agents.reviewer import ReviewerCouncil
        assert ReviewerCouncil._extract_verdict(
            "I strongly recommend to reject this submission."
        ) == "REJECT"

    def test_extract_verdict_fallback_accept(self) -> None:
        from autoresearch_v2.agents.reviewer import ReviewerCouncil
        assert ReviewerCouncil._extract_verdict(
            "Overall I think we should accept this work."
        ) == "ACCEPT"

    def test_execute_returns_structured_result(self) -> None:
        reviewer = self._make_reviewer()
        result = asyncio.run(
            reviewer.run(manuscript="\\section{Test} Some content.", round_num=1)
        )
        assert "reviews" in result
        assert "verdicts" in result
        assert "accepted" in result
        assert "consolidated_feedback" in result
        assert isinstance(result["consolidated_feedback"], list)

    def test_consolidate_feedback(self) -> None:
        from autoresearch_v2.agents.reviewer import ReviewerCouncil
        reviews = {
            "skeptic": "- Missing ablation\n- P-values unclear",
            "engineer": "- Seed not set\n- Code not reproducible",
        }
        feedback = ReviewerCouncil._consolidate_feedback(reviews)
        assert any("Skeptic" in f for f in feedback)
        assert any("Engineer" in f for f in feedback)

    def test_active_personas_from_config(self) -> None:
        from autoresearch_v2.agents.reviewer import ReviewerCouncil
        cfg = {**BASE_CONFIG, "review_council": "skeptic,visionary"}
        r = ReviewerCouncil(cfg)
        assert r._active_personas == ["skeptic", "visionary"]


# ===========================================================================
# IdeatorAgent
# ===========================================================================

class TestIdeatorAgent:
    def _make_ideator(self) -> Any:
        from autoresearch_v2.discovery.ideator import IdeatorAgent
        return IdeatorAgent(BASE_CONFIG, log_dir="/tmp/ar_test_logs")

    def test_parse_hypotheses(self) -> None:
        from autoresearch_v2.discovery.ideator import IdeatorAgent
        raw = "H1: First hypothesis.\nH2: Second hypothesis.\nH3: Third one."
        parsed = IdeatorAgent._parse_hypotheses(raw, 3)
        assert len(parsed) == 3
        assert "First hypothesis" in parsed[0]

    def test_parse_hypotheses_respects_limit(self) -> None:
        from autoresearch_v2.discovery.ideator import IdeatorAgent
        raw = "\n".join(f"H{i}: Hypothesis {i}." for i in range(1, 10))
        parsed = IdeatorAgent._parse_hypotheses(raw, 3)
        assert len(parsed) == 3

    def test_crossover_produces_marker(self) -> None:
        ideator = self._make_ideator()
        result = ideator._crossover(["Abstract A about X.", "Abstract B about Y."])
        assert "[CROSSOVER]" in result

    def test_execute_returns_hypotheses(self) -> None:
        ideator = self._make_ideator()
        result = asyncio.run(ideator.run(topic="LLM pruning", n_hypotheses=2))
        assert "hypotheses" in result
        assert len(result["hypotheses"]) > 0
        assert "arxiv_papers" in result


# ===========================================================================
# TreeSearch
# ===========================================================================

class TestTreeSearch:
    def _make_ts(self) -> Any:
        from autoresearch_v2.agents.coder import CoderAgent
        from autoresearch_v2.discovery.evaluator import Evaluator
        from autoresearch_v2.discovery.tree_search import TreeSearch
        coder = CoderAgent(BASE_CONFIG, log_dir="/tmp/ar_test_logs")
        evaluator = Evaluator(BASE_CONFIG)
        return TreeSearch(BASE_CONFIG, coder_agent=coder, evaluator_agent=evaluator)

    def test_run_returns_correct_branch_count(self) -> None:
        ts = self._make_ts()
        result = asyncio.run(
            ts.run(["H1: Prune attention heads.", "H2: Distil to small model."])
        )
        assert "branches" in result
        # With prune_after_pct=0.5 on 2 branches: 1 in Phase1, 1 in Phase2.
        # After pruning bottom 50% of early (1 branch), 0 or 1 survive + 1 remaining.
        assert len(result["branches"]) >= 1

    def test_alive_branches_not_re_executed(self) -> None:
        """Branches that survive Phase 1 pruning must not be re-run in Phase 2."""
        from autoresearch_v2.agents.coder import CoderAgent
        from autoresearch_v2.discovery.evaluator import Evaluator
        from autoresearch_v2.discovery.tree_search import TreeSearch

        run_counts: Dict[str, int] = {}

        class CountingCoder(CoderAgent):
            async def _execute(self, hypothesis: str, branch_id: str = "", **kw: Any) -> Dict[str, Any]:  # type: ignore[override]
                run_counts[branch_id] = run_counts.get(branch_id, 0) + 1
                return {"success": True, "code": "pass", "output": "accuracy: 0.9", "retries": 0}

        coder = CountingCoder(BASE_CONFIG, log_dir="/tmp/ar_test_logs")
        evaluator = Evaluator(BASE_CONFIG)
        ts = TreeSearch(BASE_CONFIG, coder_agent=coder, evaluator_agent=evaluator)
        asyncio.run(ts.run(["H1", "H2", "H3", "H4"]))

        for branch_id, count in run_counts.items():
            assert count == 1, f"{branch_id} was executed {count} times (expected 1)"

    def test_prune_removes_low_scorers(self) -> None:
        from autoresearch_v2.discovery.tree_search import ResearchBranch, TreeSearch
        from autoresearch_v2.discovery.evaluator import Evaluator
        from autoresearch_v2.agents.coder import CoderAgent

        ts = self._make_ts()
        branches = [ResearchBranch(f"B{i}", f"H{i}") for i in range(4)]
        for i, b in enumerate(branches):
            b.status = "done"
            b.combined_score = float(i) * 0.25  # 0.0, 0.25, 0.5, 0.75

        alive = ts._prune(branches)
        assert len(alive) == 2
        for b in alive:
            assert b.status != "pruned"


# ===========================================================================
# Evaluator
# ===========================================================================

class TestEvaluator:
    def _make_evaluator(self) -> Any:
        from autoresearch_v2.discovery.evaluator import Evaluator
        return Evaluator(BASE_CONFIG)

    def test_heuristic_significance_success(self) -> None:
        evaluator = self._make_evaluator()
        output = "accuracy: 0.9523\np_value < 0.001\nimprovement of 15 points"
        score = evaluator._heuristic_significance(output)
        assert score > 0.5

    def test_heuristic_significance_error_penalised(self) -> None:
        evaluator = self._make_evaluator()
        output = "Traceback (most recent call last):\nAttributeError: 'NoneType'"
        score = evaluator._heuristic_significance(output)
        assert score < 0.3

    def test_verify_manuscript_all_traceable(self) -> None:
        evaluator = self._make_evaluator()
        latex = r"\section{Results} The accuracy is 0.9523 and the loss is 0.034."
        logs = {"B1": "accuracy: 0.9523\nloss: 0.034"}
        report = evaluator.verify_manuscript(latex, logs)
        assert report["pass_rate"] == 1.0

    def test_verify_manuscript_flags_missing(self) -> None:
        evaluator = self._make_evaluator()
        latex = r"\section{Results} Accuracy: 0.9999"
        logs = {"B1": "accuracy: 0.1234"}
        report = evaluator.verify_manuscript(latex, logs)
        assert "0.9999" in report["unverified"]

    def test_statistical_significance_below_threshold(self) -> None:
        evaluator = self._make_evaluator()
        assert evaluator.is_statistically_significant(0.01)

    def test_statistical_significance_above_threshold(self) -> None:
        evaluator = self._make_evaluator()
        assert not evaluator.is_statistically_significant(0.1)

    def test_statistical_significance_none(self) -> None:
        evaluator = self._make_evaluator()
        assert not evaluator.is_statistically_significant(None)

    def test_number_in_logs_found(self) -> None:
        from autoresearch_v2.discovery.evaluator import Evaluator
        assert Evaluator._number_in_logs("0.9523", "Result: accuracy 0.9523")

    def test_number_in_logs_not_found(self) -> None:
        from autoresearch_v2.discovery.evaluator import Evaluator
        assert not Evaluator._number_in_logs("0.0001", "Result: accuracy 0.9523")

    def test_score_failed_branch_zero(self) -> None:
        from autoresearch_v2.discovery.tree_search import ResearchBranch
        evaluator = self._make_evaluator()
        branch = ResearchBranch("B1", "hypothesis")
        branch.result = {"success": False}
        scores = evaluator.score(branch)
        assert scores["novelty"] == 0.0
        assert scores["significance"] == 0.0

    def test_score_successful_branch_positive(self) -> None:
        from autoresearch_v2.discovery.tree_search import ResearchBranch
        evaluator = self._make_evaluator()
        branch = ResearchBranch("B1", "novel hypothesis")
        branch.result = {"success": True, "output": "accuracy: 0.9523\np_value < 0.001"}
        scores = evaluator.score(branch)
        assert scores["novelty"] > 0
        assert scores["significance"] > 0


# ===========================================================================
# DockerSandbox
# ===========================================================================

class TestDockerSandbox:
    def _make_sandbox(self) -> Any:
        from autoresearch_v2.env.sandbox import DockerSandbox
        return DockerSandbox(BASE_CONFIG)

    def test_simple_code_execution(self) -> None:
        sandbox = self._make_sandbox()
        result = sandbox.run("print('accuracy: 0.95')", timeout=10)
        assert result["success"] is True
        assert "0.95" in result["output"]

    def test_failing_code(self) -> None:
        sandbox = self._make_sandbox()
        result = sandbox.run("raise ValueError('intentional')", timeout=10)
        assert result["success"] is False

    def test_syntax_error(self) -> None:
        sandbox = self._make_sandbox()
        result = sandbox.run("def broken(\n    pass", timeout=10)
        assert result["success"] is False

    def test_output_captured(self) -> None:
        sandbox = self._make_sandbox()
        result = sandbox.run("for i in range(3): print(f'step {i}')", timeout=10)
        assert result["success"] is True
        assert "step 0" in result["output"]


# ===========================================================================
# LLMTool
# ===========================================================================

class TestLLMTool:
    def _make_llm(self) -> Any:
        from autoresearch_v2.env.tools import LLMTool
        return LLMTool(BASE_CONFIG)

    def test_stub_hypothesis_output(self) -> None:
        llm = self._make_llm()
        response = llm.complete("Generate hypotheses for LLM pruning techniques")
        assert len(response) > 0

    def test_stub_code_output_contains_print(self) -> None:
        llm = self._make_llm()
        response = llm.complete("Write Python code to test this hypothesis")
        assert "print" in response or "def" in response

    def test_stub_review_contains_verdict(self) -> None:
        llm = self._make_llm()
        response = llm.complete("Review this paper and provide a verdict")
        assert "VERDICT" in response

    def test_stub_score_output(self) -> None:
        llm = self._make_llm()
        response = llm.complete("Score the novelty of this hypothesis")
        assert "novelty" in response.lower()


# ===========================================================================
# ArXivTool
# ===========================================================================

class TestArXivTool:
    def test_stub_papers_have_required_keys(self) -> None:
        from autoresearch_v2.env.tools import ArXivTool
        tool = ArXivTool(BASE_CONFIG)
        papers = tool._stub_papers("LLM pruning")
        assert len(papers) >= 1
        for p in papers:
            assert "abstract" in p
            assert "title" in p
            assert "id" in p
            assert "url" in p

