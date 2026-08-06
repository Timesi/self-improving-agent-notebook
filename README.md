# Self-Improving Agent Notebook

<p align="center">
  <strong>Build AI agents from scratch through 17 runnable Jupyter Notebooks.</strong>
</p>

<p align="center">
  <a href="README.md"><strong>English</strong></a>
  ·
  <a href="README-CN.md"><strong>中文文档</strong></a>
  ·
  <a href="https://walkinglabs.github.io/self-improving-agent-notebook/"><strong>Read Online</strong></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#what-you-will-build">What You Will Build</a> ·
  <a href="#why-this-project">Why</a> ·
  <a href="#what-is-included">What Is Included</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#project-status">Status</a> ·
  <a href="#curriculum">Curriculum</a> ·
  <a href="#notebook-index">Notebook Index</a> ·
  <a href="#quality-bar">Quality Bar</a> ·
  <a href="#papers-and-systems">Papers</a> ·
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C">
  <img alt="Notebooks" src="https://img.shields.io/badge/Notebooks-17-orange">
  <img alt="Language" src="https://img.shields.io/badge/Language-Chinese-2ea44f">
  <img alt="LLM" src="https://img.shields.io/badge/LLM-OpenAI%20Compatible%20%2B%20Mock-4b32c3">
</p>

---

<p align="center">
  <strong>Paper-based agent learning: paper -> study notes -> runnable notebook -> experiment.</strong>
</p>

<p align="center">
  <em>Each lecture starts from the assigned papers, builds intuition with hand calculations, then
  implements the core agent algorithm from scratch in a notebook you can run offline.</em>
</p>

## Overview

Self-Improving Agent Notebook is a hands-on course for building AI agent systems from the ground up.
Instead of treating agents as a black box, you implement the pieces yourself: the agent loop,
test-time compute scaling, verifiers, tool use and code feedback, planning and tree search,
reinforcement learning for reasoning, open-ended evolution, search agents, memory, evaluation, and
more.

Every notebook follows the same learning contract:

```text
intuition -> hand calculation -> implementation -> experiment
```

That contract matters. A reader should not only know that repeated sampling improves accuracy, or
that a process reward model is harder to train than an outcome one. They should be able to trace the
numbers, write the minimal loop, and explain why the behavior appears.

The course is designed as an **educational reference implementation**. It is not an agent framework,
not a wrapper around a hosted agent product, and not a model zoo. Its purpose is to make the
internal machinery of AI agents legible to engineers who want to reason from first principles.

The repository ships with a unified LLM client (`llm_client.py`) that talks to any OpenAI-compatible
endpoint when a key is present, and falls back to a deterministic mock otherwise. This means every
notebook runs to completion offline, and lights up with real model behavior when you add an API key.

## What You Will Build

By the end, you will have implemented a compact version of the systems behind modern agents:

| Stage | You build | Why it matters |
|:---|:---|:---|
| Agent loop | ReAct-style thought/action/observation loop, action parser, tool registry | See how one LLM call becomes a multi-step agent |
| Test-time compute | Repeated sampling, self-consistency, best-of-n, compute-optimal scaling | Understand where inference-time gains actually come from |
| Verification | Outcome and process reward models (ORM/PRM), verifier training | Learn why "generate" needs a reliable "verify" |
| Tool & code feedback | Tool calling, code execution as reward, critique-revision pipelines | Ground reasoning in environment signals |
| Planning | Task decomposition, UCT/tree search (LATS-style), interleaved plan-execute | See how agents search over actions, not just tokens |
| Training for reasoning | STaR bootstrapping, GRPO advantage and loss from scratch | Connect RL objectives to agent capability |
| Evolution & search | Agent design search, sampling-filter-cluster code generation | Program the meta-loop that improves agents |
| Memory | Hierarchical context management, eviction, KV cache reuse | Handle windows that are smaller than the world |
| Evaluation | Agent eval harness, win-rate, time-horizon fitting, judge bias | Measure what an agent can actually finish |
| Frontiers | CoT, formal proof search with verifiers, VLA action tokens, autonomy | Turn recent papers into small runnable examples |

```text
LLM call -> agent loop -> tools -> verification -> planning -> RL -> memory -> evaluation
```

## Why This Project

Agent education often falls into two extremes.

Some resources are mathematically precise but difficult to enter: they introduce formulas before the
reader understands the problem being solved. Other resources are easy to run but heavily abstracted:
the important ideas disappear behind a framework call.

Self-Improving Agent Notebook takes the middle path. It treats agents as systems that can be
decomposed, tested, and rebuilt piece by piece. The goal is not to replace papers or agent
frameworks. The goal is to give you the mental model needed to read those papers and use those
frameworks with judgment.

Use this project if you want to:

- Understand the full loop that turns a language model into an agent.
- Build a ReAct loop, a tree search, a verifier, or a GRPO update without treating them as black boxes.
- See how inference-time compute, verification, and training-time RL interact.
- Learn why memory, evaluation, and reliable execution are systems problems, not prompts.
- Connect recent research such as verifiers, test-time compute scaling, and agentic RL back to small
  runnable examples.

## What Is Included

| Area | Topics | Reference implementations |
|:---|:---|:---|
| Agent core | ReAct loop, action parsing, tool registry | `AgentLoop`, `parse_actions`, `MiniWiki` |
| Test-time compute | Repeated sampling, self-consistency, best-of-n, coverage/precision | `pass@k` estimator, majority vote, best-of-n |
| Verification | ORM/PRM, step reward, verifier training | `VerifierMLP`, step-parsing judge, PRM product/min |
| Planning | UCT selection, backprop, task decomposition | `Node`/`UCT`, recursive decomposer, stage numbers |
| RL for reasoning | STaR loop, GRPO advantage and loss, PPO vs GRPO | `grpo_loss`, group advantage, Clip-Higher |
| Evolution & search | Meta-agent design loop, sampling-filter-cluster | `compile_agent`, behavior signatures, clustering |
| Memory | Hierarchical contexts, eviction policies, KV fusion | `ToyAgent` memory, FIFO/LRU/importance, CacheBlend-style fusion |
| Evaluation | Eval harness, win-rate, time-horizon logistic fit | `run_eval`, win/tie/loss, `p = σ((log h − log t)·β)` |
| Frontiers | CoT templates, proof search + verifier, action tokenization | few-shot CoT, horn-clause engine, `discretize`/`detokenize` |

## What This Project Is Not

This repository intentionally avoids several things so the learning path stays clear:

- It is not a production agent framework.
- It is not a wrapper around a hosted agent product or a particular model vendor.
- It does not require an API key to run (mock mode covers every LLM call).
- It does not use agent frameworks (LangChain, AutoGen, ...) as a shortcut for core implementations.
- It does not assume the reader already knows the terminology.

Agent frameworks may appear for comparison in the text, but the teaching path keeps every core
algorithm explicit. LLM calls go through one thin client (`llm_client.py`) and nothing else.

## Quick Start

### Python notebooks

```bash
git clone https://github.com/walkinglabs/self-improving-agent-notebook.git
cd self-improving-agent-notebook

# Create an isolated Python environment instead of installing into the system Python.
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user \
  --name self-improving-agent \
  --display-name "Python (self-improving-agent)"

jupyter notebook notebooks/part1-foundation/01-course-overview.ipynb
```

If `jupyter: command not found` appears, the virtual environment is probably not active. Run:

```bash
source .venv/bin/activate
```

Or call Jupyter directly from the environment:

```bash
.venv/bin/jupyter notebook notebooks/part1-foundation/01-course-overview.ipynb
```

Recommended environment:

- Python 3.10+
- PyTorch 2.0+
- NumPy, Matplotlib, Jupyter
- 16GB RAM

Most notebooks run on CPU. The reinforcement-learning demos use tiny models and finish quickly;
larger training experiments are easier with a GPU.

### Mock mode (no API key needed)

Every notebook is written so that all LLM-dependent cells work with a deterministic mock. To force
mock mode:

```bash
export LLM_MOCK=1
jupyter nbconvert --to notebook --execute \
  notebooks/part1-foundation/02-test-time-compute.ipynb --output /tmp/out.ipynb
```

### Real LLM (optional)

To see real model behavior, point the client at any OpenAI-compatible endpoint:

```bash
export AGENT_LLM_BASE_URL="https://api.deepseek.com"
export AGENT_LLM_API_KEY="sk-..."
export AGENT_LLM_MODEL="deepseek-v4-flash"
```

Without a key, `get_llm()` in `llm_client.py` returns the mock automatically. All agent demos parse
tolerant of mock output, so the same cell works in both modes.

## Project Status

| Area | Status |
|:---|:---|
| Notebooks | Complete 17/17 across 4 parts |
| Study notes | 17/17 lectures, one `papers/lecture-XX/NOTES.md` each |
| Papers | 39 papers downloaded and read; reproducible via `scripts/download_papers.py` |
| Format | All notebooks pass `nbformat.validate`; metadata normalized |
| Execution | 17/17 execute with zero errors in mock mode; real-API spot checks pass |
| Language | Chinese notebooks, English README |

### Near-Term Roadmap

1. Polish the flow of the existing notebooks so explanations read more naturally from intuition to
   code.
2. Widen real-API verification across all notebooks that exercise the LLM client.
3. Add deeper material on agent reliability and scalable oversight.
4. Consider an English mirror of the notebooks.

## Curriculum

The curriculum is organized as four parts and 17 self-contained notebooks.

```text
Self-Improving Agent Notebook
│
├── Part 1: Foundation
│   ├── 01 Course Overview
│   ├── 02 Test-time Compute Scaling
│   ├── 03 Robust Verification
│   ├── 04 Tool Use & Code Feedback
│   └── 05 Multi-step Reasoning & Planning
│
├── Part 2: Training & Evolution
│   ├── 06 Train-time Scaling & RL
│   ├── 07 Open-Ended Evolution of Self-Improving Agents
│   ├── 08 Search & Deep Research Agents
│   └── 09 Post-training: from Chatbots to Agents
│
├── Part 3: Agent Engineering
│   ├── 13 Agentic Frameworks for Software Engineering
│   ├── 14 Augmenting Agents with Memory
│   └── 17 Agentic Evaluations & Long-Horizon Tasks
│
└── Part 4: Frontiers
    ├── 15 LLM Reasoning
    ├── 16 Mathematical Reasoning: AlphaProof & AlphaGeometry
    ├── 18 Building Agentic Systems for Autonomy
    ├── 19 Multimodal AI Agents in Robotics
    └── 20 Future Research Areas
```

Each notebook is designed to be runnable on its own. You can follow the full sequence or jump to a
topic without depending on hidden runtime state from earlier notebooks. (Lectures 10-12 in the source
course were midterm presentations and have no notebooks.)

## Notebook Index

### Part 1: Foundation

| # | Notebook | Primary question | Implementation focus |
|:---:|:---|:---|:---|
| 01 | [Course Overview](notebooks/part1-foundation/01-course-overview.ipynb) | What is an agent, and why is it the next step after a chatbot? | Agent loop skeleton, action parser |
| 02 | [Test-time Compute Scaling](notebooks/part1-foundation/02-test-time-compute.ipynb) | How does spending more inference compute help? | pass@k, self-consistency, best-of-n, scaling curves |
| 03 | [Robust Verification](notebooks/part1-foundation/03-robust-verification.ipynb) | How do we check answers the model generates? | ORM vs PRM, verifier training, step reward |
| 04 | [Tool Use & Code Feedback](notebooks/part1-foundation/04-tool-code-feedback.ipynb) | How does an agent act in an environment? | ReAct loop, tool registry, execution reward |
| 05 | [Multi-step Reasoning & Planning](notebooks/part1-foundation/05-multi-step-planning.ipynb) | How do agents plan beyond one step? | Decomposition, UCT tree search, parallel execution |

### Part 2: Training & Evolution

| # | Notebook | Primary question | Implementation focus |
|:---:|:---|:---|:---|
| 06 | [Train-time Scaling & RL](notebooks/part2-training/06-train-time-scaling-rl.ipynb) | How do we train reasoning into a model? | STaR bootstrapping, GRPO advantage and loss |
| 07 | [Open-Ended Evolution](notebooks/part2-training/07-open-ended-evolution.ipynb) | How do agents design better agents? | Meta-agent search loop, reward hacking |
| 08 | [Search & Deep Research](notebooks/part2-training/08-search-deep-research.ipynb) | How do agents search programs and knowledge? | Sampling-filter-cluster, retrieval injection |
| 09 | [Post-training Evolution](notebooks/part2-training/09-post-training-evolution.ipynb) | How did post-training evolve from chatbots to agents? | SFT vs RLHF vs RLVR losses, signal sources |

### Part 3: Agent Engineering

| # | Notebook | Primary question | Implementation focus |
|:---:|:---|:---|:---|
| 13 | [SWE Agents](notebooks/part3-engineering/13-swe-agents.ipynb) | How do agents fix real code? | Test-then-edit state machines, fast_p metric |
| 14 | [Agent Memory](notebooks/part3-engineering/14-agent-memory.ipynb) | How do agents remember what they saw? | Hierarchical contexts, eviction, KV reuse |
| 17 | [Agent Evaluation](notebooks/part3-engineering/17-agent-evaluation.ipynb) | How do we measure long-horizon agents? | Eval harness, win-rate, time-horizon fit |

### Part 4: Frontiers

| # | Notebook | Primary question | Implementation focus |
|:---:|:---|:---|:---|
| 15 | [LLM Reasoning](notebooks/part4-frontiers/15-llm-reasoning.ipynb) | Where does reasoning ability come from? | CoT templates, self-consistency, emergent curves |
| 16 | [Mathematical Reasoning](notebooks/part4-frontiers/16-alphaproof-math.ipynb) | How do verifiers + search prove theorems? | Horn-clause engine, proof verifier, search loop |
| 18 | [Autonomy](notebooks/part4-frontiers/18-autonomy-agents.ipynb) | What is missing between demo and autonomy? | Self-check, recovery, confidence-gated takeover |
| 19 | [Multimodal Robotics](notebooks/part4-frontiers/19-multimodal-robotics.ipynb) | How do vision-language-action models work? | Action tokenization, vocabulary masking, mini VLA |
| 20 | [Future Research Areas](notebooks/part4-frontiers/20-future-research.ipynb) | What is still unsolved? | Open-problem map, reward hacking, coordination |

## Quality Bar

The repository follows a small set of standards to keep the notebooks useful as learning material:

- Concepts are introduced by motivation before notation.
- New terminology is defined before it is used heavily.
- Core algorithms include at least one concrete hand calculation or toy example.
- Code cells are kept small and observable.
- Randomized experiments use fixed seeds where appropriate.
- Every notebook is self-contained and runs in mock mode with zero errors.
- Markdown explanations are written for patient beginners, while the code remains close to the real
  algorithmic structure.

## Papers and Systems

The course connects implementation details to influential papers and systems:

| Paper or system | Concepts covered |
|:---|:---|
| Large Language Monkeys / Snell et al. | Repeated sampling, compute-optimal test-time scaling |
| Cobbe verifiers / Lightman / Math-Shepherd | Outcome vs process reward models |
| ReAct / RLEF / Constitutional AI | Tool use, execution feedback, AI feedback |
| LATS / ADaPT / SPRINT | Tree search, as-needed decomposition, parallel execution |
| STaR / DeepSeekMath-GRPO / DAPO | Reasoning bootstrapping, group-relative RL |
| ADAS / AI Scientist / AlphaEvolve | Automated agent design, open-ended evolution |
| AlphaCode / Search-o1 | Program sampling-filter-cluster, agentic retrieval |
| MemGPT / Cartridges / CacheBlend | Hierarchical memory, learned compression, KV reuse |
| CodeMonkeys / KernelBench | Test-time compute for software engineering |
| METR / GDPval / DeepScholar-Bench | Long-horizon and economic-value evaluation |
| CoT / Self-Consistency / Emergent Abilities | Reasoning traces, sampling-and-marginalizing |
| AlphaGeometry / AlphaProof | Neuro-symbolic proof, formalization + RL |
| RT-2 / OpenVLA | Vision-language-action models |

## Course Origin

The curriculum mirrors the syllabus of Stanford's [CS329A: AI Agents](https://cs329a.stanford.edu/)
(Autumn 2025). The course page lists each lecture's papers without lecture details. This repository
downloads and reads those papers, writes a study note for every lecture (`papers/lecture-XX/NOTES.md`),
then turns each lecture into a runnable notebook that reconstructs how the lecturer would teach it.

Guest lectures without assigned papers (e.g., post-training evolution, LLM reasoning, AlphaProof,
autonomy, robotics) are reconstructed from the speaker's research direction and related papers. See
[papers/](papers/) for the study notes and `scripts/download_papers.py` for reproducing the paper
collection. The teaching contract and repository style are inspired by
[Modern LLM Notebook](https://github.com/walkinglabs/modern-llm-notebook).

## Repository Structure

```text
self-improving-agent-notebook/
├── notebooks/                    # Chinese notebooks (17)
│   ├── part1-foundation/         # 01-05
│   ├── part2-training/           # 06-09
│   ├── part3-engineering/        # 13, 14, 17
│   └── part4-frontiers/          # 15, 16, 18, 19, 20
├── papers/                       # Study notes (NOTES.md) per lecture; PDFs reproducible
│   ├── lecture-01/ ... lecture-20/
│   └── NOTES_TEMPLATE.md
├── scripts/
│   └── download_papers.py        # Resolve and download all lecture papers from arXiv
├── llm_client.py                 # Unified LLM client (OpenAI-compatible + deterministic mock)
├── .claude/CLAUDE.md             # Notebook writing guide
├── OUTLINE.md                    # Full course outline
├── CLAUDE.md
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome when they improve clarity, correctness, or coverage.

Good contributions include:

- Fixing incorrect explanations, broken cells, or outdated APIs.
- Improving hand-calculation sections and visualizations.
- Adding focused exercises with assertions.
- Proposing new notebooks for important agent topics.
- Helping verify notebooks against a real LLM endpoint.

## Citation

If Self-Improving Agent Notebook helps your research or work, please cite:

```bibtex
@misc{self-improving-agent-notebook,
  title   = {Self-Improving Agent Notebook: Build AI Agents from Scratch},
  author  = {WalkingLabs},
  year    = {2026},
  url     = {https://github.com/walkinglabs/self-improving-agent-notebook},
  note    = {GitHub repository, accessed 2026}
}
```

## License

License to be determined. The project follows the spirit of the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/),
same as the reference project Modern LLM Notebook.

---

<p align="center">
  <sub>
    Built for engineers who want to understand agent systems from the inside.
    <br>
    Maintained by <a href="https://github.com/walkinglabs">walkinglabs</a>.
  </sub>
</p>
