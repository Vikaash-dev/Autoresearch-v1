"""
conftest.py — Shared pytest fixtures for AutoResearch v2 tests.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Event-loop fixture (Python 3.10+ compatible)
# ---------------------------------------------------------------------------

@pytest.fixture
def event_loop():
    """Provide a fresh event loop for each test function."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Shared config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config() -> Dict[str, Any]:
    return {
        "llm": {"default_model": "gpt-4o", "temperature": 0.7, "max_tokens": 512},
        "compute": {
            "branches": 2,
            "max_branch_workers": 2,
            "prune_after_pct": 0.5,
            "prune_bottom_pct": 0.5,
            "max_debug_retries": 3,
            "kernel_check_interval": 1,
            "sandbox_timeout": 10,
        },
        "memory": {
            "persist_directory": "/tmp/test_research_memory",
            "collection_name": "test_col",
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
