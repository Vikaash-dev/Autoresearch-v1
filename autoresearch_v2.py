"""
autoresearch_v2.py — Top-level entry point (proposal CLI compatibility).

The proposal specifies:
  python autoresearch_v2.py --topic "..." --effort pro --branches 5

This module is a thin shim that delegates to ``main.py``.
"""
from main import main

if __name__ == "__main__":
    main()
