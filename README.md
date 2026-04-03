# AutoResearch v2 (HRDE)
**A Self-Evolving, Theory-of-Mind-Enhanced Autonomous Research Framework**

> *"We are no longer just optimising for Accuracy (the relationship between the AI and the Data).  
> We are optimising for Epistemic Impact (the relationship between the AI, the Data, and the Human Mind)."*

---

## What Is This?

AutoResearch v2 (HRDE — **H**yper-agentic **R**ecursive **D**iscovery **E**ngine) is a self-improving multi-agent system that treats the **entire research process** — ideation, experimentation, writing, and peer review — as a single, continuously self-optimising loop rather than a fixed pipeline.

It supersedes linear automation frameworks (Sakana v1, AutoResearchClaw) by introducing:

| Capability | v1 / Sakana | AutoResearch v2 (HRDE) |
|---|---|---|
| **Structure** | Static pipeline (Stage 1 → Stage N) | Dynamic Objective Graph (DOG) |
| **Self-improvement** | Human-coded rule updates | Self-Modifying Hyper-Kernel (DGM-H) |
| **Reviewing** | Single generic AI reviewer | Multi-Persona ToM Council |
| **Verification** | Plausibility checks | Zero-Trust Epistemic Anchoring |
| **Novelty search** | Combinatorial (A + B) | Anomaly-driven gap detection |

---

## The Five Core Pillars

### 1 · Hyper-Kernel (DGM-H) — `core/kernel.py`
A **self-referential meta-agent** with read/write access to every sub-agent's system prompt.

- Monitors agent log files in real time (byte-offset tracking avoids double-counting).
- When a **Reasoning Loop** is detected (agent fails > `max_debug_retries` times), it calls an LLM to **rewrite the offending agent's prompt** and resets the failure counter.
- Runs in a daemon background thread via `kernel.start()` / `kernel.stop()`.

```
[CoderAgent fails 4×]
        ↓
HyperKernel detects loop
        ↓
LLM generates improved system prompt
        ↓
kernel.write_agent_prompt("CoderAgent", new_prompt)
        ↓
CoderAgent retries with patched instructions
```

### 2 · Dynamic Objective Graph (DOG) — `core/graph_manager.py`
A **networkx-backed async task graph** that replaces linear pipelines.

- Tasks declare dependencies; the engine runs them in topological order.
- If `coding` fails, it dynamically **spawns** the registered `library_research` recovery task.
- Cycle detection raises `ValueError` before execution begins.

```python
gm.add_task("ideation",  ideation_fn)
gm.add_task("coding",    coding_fn,   depends_on=["ideation"], recovery_task="research")
gm.add_task("research",  research_fn)
gm.add_task("review",    review_fn,   depends_on=["coding"])
await gm.run()
```

### 3 · ToM Reviewer Council — `agents/reviewer.py`
A **3-persona adversarial review system** modelling Theory of Mind.

Each persona has a distinct cognitive bias baked into its system prompt:

| Persona | Focus |
|---|---|
| **The Skeptic** | Methodological flaws, p-hacking, over-claiming |
| **The Engineer** | Code reproducibility, deterministic seeds, hardware assumptions |
| **The Visionary** | Long-term novelty, paradigm impact, cross-disciplinary bridges |

The `debate_loop()` runs up to `review_rounds` rounds. After each round, the WriterAgent **revises the manuscript** based on consolidated feedback. The paper is accepted only when ≥ `acceptance_threshold` personas approve.

### 4 · Agentic Tree Search — `discovery/tree_search.py`
**Parallel hypothesis exploration** with mid-run pruning and branch grafting (Sakana v2 style).

- **Phase 1**: Run the first `prune_after_pct` (default 20%) of branches concurrently.
- **Pruning**: Kill the bottom `prune_bottom_pct` (default 50%) by combined novelty + significance score. Already-surviving branches are **never re-executed**.
- **Phase 2**: Run remaining unstarted branches.
- **Grafting**: Merge the best code from Branch A with the best theoretical framework from Branch B into a hybrid Branch C.

### 5 · Epistemic Anchor — `discovery/evaluator.py`
**Zero-Trust verification**: every numerical claim in the final LaTeX paper must be traceable back to a line in the sandbox execution logs.

- `verify_manuscript(latex_source, execution_logs)` → `{verified, unverified, pass_rate}`.
- `is_statistically_significant(p_value, effect_size)` — warns on small effect sizes despite low p-values (p-hacking guard).
- Branch scoring uses heuristics + optional LLM scoring for novelty and significance.

---

## Directory Structure

```
autoresearch_v2/
├── core/
│   ├── kernel.py          # DGM-H Hyper-Kernel (self-modification + monitoring)
│   ├── graph_manager.py   # Dynamic Objective Graph (DOG) orchestrator
│   └── memory.py          # Vectorized Research Graph (ChromaDB / fallback)
├── agents/
│   ├── base_agent.py      # Abstract ToM-enabled agent base class
│   ├── coder.py           # Self-healing Python coder (Docker-integrated)
│   ├── writer.py          # LaTeX / TikZ manuscript generator
│   └── reviewer.py        # Multi-persona adversarial reviewer council
├── discovery/
│   ├── ideator.py         # ArXiv-grounded evolutionary hypothesis generator
│   ├── tree_search.py     # Branching experiment manager (prune + graft)
│   └── evaluator.py       # Zero-trust epistemic anchor + branch scorer
├── env/
│   ├── sandbox.py         # Docker / subprocess execution environment
│   └── tools.py           # LLMTool · ArXivTool · WebSearchTool · PlottingTool
├── tests/
│   └── test_autoresearch_v2.py   # 62 unit tests (no API keys or Docker required)
├── config.yaml            # Hyper-parameters (effort, compute, LLM, output)
└── __init__.py
main.py                    # CLI entry point
conftest.py                # Pytest fixtures
requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set LLM API keys (optional — stub fallback works offline)
```bash
export OPENAI_API_KEY="sk-..."        # for GPT-4o (writer)
export ANTHROPIC_API_KEY="sk-ant-..."  # for Claude (coder)
export GOOGLE_API_KEY="..."            # for Gemini (ideator)
```

### 3. Run
```bash
# Minimal (3 parallel branches, standard effort)
python main.py --topic "Novel LLM pruning techniques" --branches 3

# Full options
python main.py \
  --topic "Self-correcting LLM architectures using ToM" \
  --effort pro \
  --branches 5 \
  --review_council "skeptic,engineer,visionary" \
  --output_dir ./my_output
```

### What happens
1. **Ideation** — IdeatorAgent fetches ArXiv papers, applies evolutionary crossover/mutation, generates `N` hypotheses.
2. **Tree Search** — CoderAgent runs each hypothesis as a Python experiment in an isolated sandbox (Docker if available, subprocess fallback otherwise). Weak branches are pruned at 20% of the budget.
3. **Writing** — WriterAgent drafts a NeurIPS-template LaTeX paper from the results.
4. **Review** — ReviewerCouncil runs a multi-round adversarial debate; WriterAgent revises until accepted.
5. **Verification** — Evaluator checks every number in the paper against execution logs.
6. **Output** — `output/<topic>_<timestamp>.tex` + metadata JSON + optional `source_code.zip`.

---

## Running Tests
```bash
# All 62 unit tests — no API keys, no Docker required
python -m pytest autoresearch_v2/tests/ -v
```

Tests cover every component in isolation using the built-in stub LLM and subprocess sandbox.

---

## Configuration (`autoresearch_v2/config.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `effort` | `standard` | Compute budget level: `minimal` / `standard` / `pro` / `max` |
| `compute.branches` | `3` | Parallel hypothesis branches |
| `compute.max_debug_retries` | `3` | Failures before HyperKernel patches an agent |
| `compute.prune_after_pct` | `0.20` | Prune after this fraction of branches complete |
| `compute.prune_bottom_pct` | `0.50` | Kill this fraction of low-scoring branches |
| `review_council` | `skeptic,engineer,visionary` | Active reviewer personas |
| `acceptance_threshold` | `2` | Approvals needed (out of 3) to accept the paper |
| `sandbox.enabled` | `false` | Set `true` when Docker daemon is available |
| `output.generate_pdf` | `false` | Set `true` when `pdflatex` is installed |
| `llm.default_model` | `gpt-4o` | LiteLLM-compatible model name |

---

## Infrastructure Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.12 |
| LLM | None (stub) | GPT-4o + Claude 3.5 + Gemini 1.5 |
| Vector DB | None (in-memory fallback) | ChromaDB |
| Execution | subprocess | Docker |
| PDF output | None | pdflatex (TeX Live) |

---

## Architecture Diagram

```
User CLI (main.py)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│                   HyperKernel (DGM-H)                │
│  Monitors logs · Patches agent prompts · Reflective  │
└───────────────────┬──────────────────────────────────┘
                    │ registers / patches
        ┌───────────┼───────────┬──────────────┐
        ▼           ▼           ▼              ▼
  IdeatorAgent  CoderAgent  WriterAgent  ReviewerCouncil
  (ArXiv seed)  (sandbox)   (LaTeX/TikZ) (Skeptic|Engineer|Visionary)
        │           │           │              │
        └─────────┬─┘           └──────┬───────┘
                  ▼                    ▼
           TreeSearch (DOG)      debate_loop()
           prune · graft         multi-round revision
                  │
                  ▼
            Evaluator
         Zero-Trust Verify
                  │
                  ▼
         output/<topic>.tex
         output/<topic>_meta.json
         output/<topic>_source.zip
```

---

## Design Philosophy

> **From Automation to Autopoiesis.**  
> A system that does not just *perform* research but *builds its own research methodology* and improves it with every run.

- **Compute-optimal**: 90% of thinking time on *why*, 10% on *how*. The Ideator runs thought-experiments (hypothesis simulations) before writing a single line of code.
- **Architecture-heterogeneous verification**: Author (Transformer-based LLM) must convince reviewers with different inductive biases. Truth must transcend any single architecture.
- **No echo chamber**: The Skeptic, Engineer, and Visionary reviewers independently model different cognitive communities, preventing the "Superhuman Sycophancy" failure mode.
- **Physicality anchor**: All claims traced to execution logs; optional integration with real-world datasets (OpenAQ, CERN, NASA) via `env/tools.py`.

