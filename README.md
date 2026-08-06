# Self-Improving Agent Notebook

<p align="center">
  <strong>From a single LLM call to a self-improving agent, through 17 runnable Jupyter Notebooks.</strong>
</p>

<p align="center">
  <a href="README.md"><strong>English</strong></a>
  ·
  <a href="README-CN.md"><strong>中文文档</strong></a>
  ·
  <a href="https://walkinglabs.github.io/self-improving-agent-notebook/"><strong>Read Online</strong></a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C">
  <img alt="Notebooks" src="https://img.shields.io/badge/Notebooks-17-orange">
  <img alt="Language" src="https://img.shields.io/badge/Language-Chinese-2ea44f">
  <img alt="LLM" src="https://img.shields.io/badge/LLM-OpenAI%20Compatible%20%2B%20Mock-4b32c3">
</p>

<p align="center">
  <a href="#this-is-a-course">This Is a Course</a> ·
  <a href="#learning-path">Learning Path</a> ·
  <a href="#what-you-will-build">What You Will Build</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#project-status">Status</a> ·
  <a href="#curriculum">Curriculum</a> ·
  <a href="#notebook-index">Notebook Index</a> ·
  <a href="#quality-bar">Quality Bar</a> ·
  <a href="#course-origin">Course Origin</a>
</p>

---

## This Is a Course

Self-Improving Agent Notebook is a hands-on course on AI agents. It is not a framework tutorial —
every lecture starts from the assigned papers and rebuilds the core agent algorithm by hand.

One sentence captures the whole arc:

> **A single LLM call can only "talk", not "do". We turn it, step by step, into an agent that can
> think, act, verify, be trained, and eventually improve itself.**

Every notebook follows the same learning contract:

```text
intuition -> hand calculation -> implementation -> experiment
```

You will not just run code. You will trace the numbers, implement the algorithm yourself, and explain
why the behavior appears.

## Learning Path

The course is ordered by how hard each idea is to grasp, and every step builds on the one before it.
**At the end of each step you can do something you could not do before.** You are welcome to jump
around, but read in order and each notebook picks up the question the previous one left open.

### Step 1 · What one call can and cannot do (L1)

Before writing any code, look at the four hard limits of a plain LLM app: it can only "talk" not
"do"; its context window is finite; it cannot correct its own mistakes; its knowledge stops at the
training cutoff. This lecture builds the **first minimal agent loop** — the skeleton the whole course
returns to.

### Step 2 · Does spending more inference compute help? (L2)

The model is not good enough, but training is expensive. Different approach: **ask the same question
several times, then vote.** This lecture implements repeated sampling, self-consistency, and best-of-n,
and hand-calculates the "more samples, better coverage" curve. The key lesson — **generating is the
easy half; picking the good answers is the hard part** — leads directly to the next step.

### Step 3 · Who picks? Let the machine check the machine (L3)

Picking cannot rely on guessing. This lecture trains a **verifier**: an outcome reward model (ORM)
scores only the final answer; a process reward model (PRM) checks every step. You will see the same
wrong answer scored very differently by the two. Takeaway: **generation + verification is a pair**,
and verifiers keep returning throughout the course.

### Step 4 · Let the agent act: tools and environment (L4)

Thinking without acting is meaningless. This lecture implements the **ReAct loop**: the model emits
"thought + action" each round, the action runs in a tool, and the observation is fed back. You will
wire up search, calculator, and other tools and watch the agent "think while doing".

### Step 5 · Let the agent plan: beyond one step (L5)

What if a task is too long for one step? This lecture is about **planning**: splitting the task into
subgoals (ADaPT), doing tree search over actions (LATS-style UCT), and packing parallelizable steps
(SPRINT). You will see that "keep several candidate paths" beats "greedily take one step" — the same
idea as the voting in Step 2.

### Step 6 · Train the ability into the model: reinforcement learning (L6)

Inference-time compute is expensive. Can we **put the reasoning ability into the model**? This lecture
implements STaR bootstrapping (the model keeps learning from problems it solved) and GRPO
(group-relative RL). You will hand-calculate the advantage of a batch of samples and see why we
subtract the mean and divide by the standard deviation. **The verifier from Step 3 becomes the reward
signal for training.**

### Step 7 · Let the agent improve the agent: open-ended evolution (L7)

If agents can be improved, why can't the improver be an agent too? This lecture implements an
**evolution loop**: a "design agent" writes agent code, scores it on a task, stores it, and improves
in the next round. It also exposes a dangerous failure mode — **reward hacking**: an agent learns to
"game the score" instead of "doing the job".

### Step 8 · Let the agent search and do research itself (L8)

Amplify the "sample more" idea from Step 2 by ten thousand and you get program synthesis and deep
research. This lecture implements the **sample-filter-cluster** pipeline (AlphaCode), on-demand
retrieval in reasoning (Search-o1), and a mini deep-research workflow.

### Step 9 · The whole picture: one model's life from chatbot to agent (L9)

Tie the first eight steps together with the evolution of **post-training**: SFT (imitate
demonstrations) → RLHF (learn preferences) → RLVR (learn verifiable correctness) → agentic training
(learn from environment feedback). The through-line is the **migration of the reward signal**.

### Step 10 · Engineering: make it actually useful (L13 / L14 / L17)

So far we built *abilities*; now we build *delivery*: make agents **fix code** (software-engineering
agents), **remember** (memory systems), and **be judged fairly** (evaluation harnesses and
long-horizon tasks). Evaluation introduces the time horizon — how long a task takes a human versus an
agent, and how the capability curve is fitted.

### Step 11 · Frontiers: where the boundary is (L15 / L16 / L18-L20)

Finally, walk to the frontier of agents: **the limits of reasoning** (CoT and self-consistency),
**mathematical proof** (the neuro-symbolic blend of AlphaGeometry and AlphaProof), **autonomy and
oversight**, **robotics** (vision-language-action models), and the **open problems** that are not
solved yet.

### A relay of ideas

The lectures are not just a list — many concepts are handed off from one lecture to the next:

```text
L2 voting  → L3 verifiers pick for you → L5 tree search scores nodes the same way
L3 verifiers → L6 GRPO uses them as training reward → L7 evolution uses them as fitness
L2 more sampling → L8 AlphaCode scales it to millions of candidates
L4 ReAct loop → L5 embedded in a search tree → L13 called repeatedly on code tasks
L7 reward hacking → L17 evaluation guards against it → L20 Goodhart's law
```

Read in order and each notebook catches the question the last one left; jump around and the relay
diagram still tells you where you are.

## What You Will Build

| You will build | What it means (no jargon) |
|:---|:---|
| An agent loop | The model emits "thought + action" each round, the action runs in a tool, the result feeds back, until the task is done |
| Repeated sampling and voting | Ask the same question N times, majority vote wins; see why "generating is easy, picking is hard" |
| A verifier | Train a small model to score answers: outcome-level checks only the final answer, process-level checks every step |
| A ReAct agent | Wire tools like search and calculator to the model and watch it "think while doing" |
| A search tree | Treat the task as a tree: select → expand → evaluate → back up the result (hand-calculate UCT and backup) |
| A GRPO update | Hand-calculate the advantage of a batch, then run one group-relative RL update |
| An evolution loop | A "design agent" writes agent code, scores it, archives it, improves it; reproduce reward hacking |
| A hierarchical memory | Evict old context when the window overflows, recall it when needed; understand KV cache reuse |
| An eval harness | Define tasks, run the agent, aggregate pass rate; fit a time-horizon capability curve |
| A proof searcher | The model proposes proof steps and a verifier checks each one — the AlphaProof recipe |

## Quick Start

### Python notebooks

```bash
git clone https://github.com/walkinglabs/self-improving-agent-notebook.git
cd self-improving-agent-notebook

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user \
  --name self-improving-agent \
  --display-name "Python (self-improving-agent)"

jupyter notebook notebooks/part1-foundation/01-course-overview.ipynb
```

If `jupyter: command not found` appears, run `source .venv/bin/activate` or call
`.venv/bin/jupyter notebook ...` directly.

Recommended environment: Python 3.10+, PyTorch 2.0+, NumPy, Matplotlib, Jupyter, 16GB RAM. Most
notebooks run on CPU.

### Mock mode (no API key needed)

Every notebook guarantees that all LLM-dependent cells work with a deterministic mock:

```bash
export LLM_MOCK=1
jupyter nbconvert --to notebook --execute \
  notebooks/part1-foundation/02-test-time-compute.ipynb --output /tmp/out.ipynb
```

### Real LLM (optional)

Point the client at any OpenAI-compatible endpoint to see real model behavior:

```bash
export AGENT_LLM_BASE_URL="https://api.deepseek.com"
export AGENT_LLM_API_KEY="sk-..."
export AGENT_LLM_MODEL="deepseek-v4-flash"
```

Without a key, `get_llm()` in `llm_client.py` returns the mock automatically.

## Project Status

| Area | Status |
|:---|:---|
| Notebooks | Complete 17/17 across 4 parts |
| Study notes | 17/17 lectures, one `papers/lecture-XX/NOTES.md` each |
| Papers | 39 downloaded and read; reproducible via `scripts/download_papers.py` |
| Teaching depth | Every notebook adds "intuition + concrete example + hand calculation + why" |
| Format | All pass `nbformat.validate`; cell ids normalized |
| Execution | 17/17 run with zero errors in mock mode; real-API spot checks pass |
| Language | Chinese notebooks, bilingual README |

### Near-Term Roadmap

1. Polish the writing so explanations flow naturally from intuition to code.
2. Widen real-API verification across all notebooks.
3. Add deeper material on reliability and scalable oversight.
4. Consider an English mirror of the notebooks.

## Curriculum

```text
Self-Improving Agent Notebook
│
├── Part 1 · Foundation (L1-L5)          # loop, test-time compute, verification, tools, planning
│   ├── 01 Course Overview
│   ├── 02 Test-time Compute Scaling
│   ├── 03 Robust Verification
│   ├── 04 Tool Use & Code Feedback
│   └── 05 Multi-step Reasoning & Planning
│
├── Part 2 · Training & Evolution (L6-L9) # train into the model, evolve, search, post-training
│   ├── 06 Train-time Scaling & RL
│   ├── 07 Open-Ended Evolution
│   ├── 08 Search & Deep Research
│   └── 09 Post-training: from Chatbots to Agents
│
├── Part 3 · Agent Engineering (L13/14/17) # make it actually useful
│   ├── 13 SWE Agents
│   ├── 14 Agent Memory
│   └── 17 Agent Evaluation
│
└── Part 4 · Frontiers (L15-L20)         # where the boundary is
    ├── 15 LLM Reasoning
    ├── 16 Mathematical Reasoning
    ├── 18 Autonomy
    ├── 19 Multimodal Robotics
    └── 20 Future Research
```

Every notebook is self-contained. Lectures 10-12 in the source course were midterm presentations and
have no notebooks.

## Notebook Index

### Part 1 · Foundation

| # | Notebook | Primary question | Implementation focus |
|:---:|:---|:---|:---|
| 01 | [Course Overview](notebooks/part1-foundation/01-course-overview.ipynb) | What is an agent, and why is it the next step after a chatbot? | Minimal loop, action parser |
| 02 | [Test-time Compute](notebooks/part1-foundation/02-test-time-compute.ipynb) | Why does more inference compute help? | Repeated sampling, self-consistency, best-of-n |
| 03 | [Robust Verification](notebooks/part1-foundation/03-robust-verification.ipynb) | How do we check the answers the model generates? | ORM vs PRM, verifier training, step rewards |
| 04 | [Tool Use & Code Feedback](notebooks/part1-foundation/04-tool-code-feedback.ipynb) | How does an agent act in an environment? | ReAct loop, tool registry, execution reward |
| 05 | [Planning](notebooks/part1-foundation/05-multi-step-planning.ipynb) | How does an agent plan beyond one step? | Decomposition, UCT tree search, parallel execution |

### Part 2 · Training & Evolution

| # | Notebook | Primary question | Implementation focus |
|:---:|:---|:---|:---|
| 06 | [Train-time Scaling & RL](notebooks/part2-training/06-train-time-scaling-rl.ipynb) | How do we train reasoning into a model? | STaR bootstrapping, GRPO advantage and loss |
| 07 | [Open-Ended Evolution](notebooks/part2-training/07-open-ended-evolution.ipynb) | How do agents design better agents? | Evolution loop, reward hacking |
| 08 | [Search & Deep Research](notebooks/part2-training/08-search-deep-research.ipynb) | How do agents search programs and knowledge? | Sample-filter-cluster, on-demand retrieval |
| 09 | [Post-training Evolution](notebooks/part2-training/09-post-training-evolution.ipynb) | How did post-training evolve from chatbots to agents? | SFT vs RLHF vs RLVR, signal migration |

### Part 3 · Agent Engineering

| # | Notebook | Primary question | Implementation focus |
|:---:|:---|:---|:---|
| 13 | [SWE Agents](notebooks/part3-engineering/13-swe-agents.ipynb) | How do agents fix real code? | Coverage-selection, serial repair loop |
| 14 | [Agent Memory](notebooks/part3-engineering/14-agent-memory.ipynb) | How do agents remember what they saw? | Hierarchical contexts, eviction, KV reuse |
| 17 | [Agent Evaluation](notebooks/part3-engineering/17-agent-evaluation.ipynb) | How do we measure long-horizon agents? | Eval harness, win-rate, time horizon |

### Part 4 · Frontiers

| # | Notebook | Primary question | Implementation focus |
|:---:|:---|:---|:---|
| 15 | [LLM Reasoning](notebooks/part4-frontiers/15-llm-reasoning.ipynb) | Where does reasoning ability come from? | CoT, self-consistency, emergent measures |
| 16 | [Mathematical Reasoning](notebooks/part4-frontiers/16-alphaproof-math.ipynb) | How do verifiers + search prove theorems? | Symbolic engine, proof verifier, search |
| 18 | [Autonomy](notebooks/part4-frontiers/18-autonomy-agents.ipynb) | What is missing between demo and autonomy? | Reliability, self-check retry, oversight |
| 19 | [Multimodal Robotics](notebooks/part4-frontiers/19-multimodal-robotics.ipynb) | How do VLA models work? | Action tokenization, vocabulary masking |
| 20 | [Future Research](notebooks/part4-frontiers/20-future-research.ipynb) | What is still unsolved? | Open problems, Goodhart, coordination |

## Quality Bar

The repository follows a small set of standards to keep the notebooks useful as learning material:

- Concepts are introduced by motivation before notation.
- New terminology is defined before it is used heavily.
- Core algorithms include at least one concrete hand calculation or toy example.
- Code cells are kept small and observable.
- Randomized experiments use fixed seeds.
- Every notebook is self-contained and runs in mock mode with zero errors.
- Explanations are written for patient beginners, while the code stays close to the real algorithm.

## Papers and Systems

| Paper or system | Concepts covered |
|:---|:---|
| Large Language Monkeys / Snell et al. | Repeated sampling, compute-optimal test-time scaling |
| Cobbe verifiers / Lightman / Math-Shepherd | Outcome vs process reward models |
| ReAct / RLEF / Constitutional AI | Tool use, execution feedback, AI feedback |
| LATS / ADaPT / SPRINT | Tree search, as-needed decomposition, parallel execution |
| STaR / DeepSeekMath-GRPO / DAPO | Reasoning bootstrapping, group-relative RL |
| ADAS / AI Scientist / AlphaEvolve | Automated agent design, open-ended evolution |
| AlphaCode / Search-o1 | Program sample-filter-cluster, agentic retrieval |
| MemGPT / Cartridges / CacheBlend | Hierarchical memory, learned compression, KV reuse |
| CodeMonkeys / KernelBench | Test-time compute for software engineering |
| METR / GDPval / DeepScholar-Bench | Long-horizon and economic-value evaluation |
| CoT / Self-Consistency / Emergent Abilities | Reasoning traces, sampling-and-marginalizing |
| AlphaGeometry / AlphaProof | Neuro-symbolic proof, formalization + RL |
| RT-2 / OpenVLA | Vision-language-action models |

## Course Origin

The curriculum mirrors the syllabus of Stanford's [CS329A: AI Agents](https://cs329a.stanford.edu/)
(Autumn 2025). The course page lists each lecture's papers without lecture details. This repository
downloads and reads those papers, writes a study note for every lecture
(`papers/lecture-XX/NOTES.md`), then turns each lecture into a runnable notebook that reconstructs how
the lecturer would teach it.

Guest lectures without assigned papers (post-training evolution, LLM reasoning, AlphaProof, autonomy,
robotics) are reconstructed from the speaker's research direction and related papers. See
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
├── papers/                       # Study notes per lecture; PDFs reproducible
├── scripts/download_papers.py    # Resolve and download all course papers from arXiv
├── llm_client.py                 # Unified LLM client (OpenAI-compatible + mock)
├── web/                          # React/Vite reader (deployed to GitHub Pages)
├── .claude/CLAUDE.md             # Notebook writing guide
├── OUTLINE.md                    # Full course outline
└── README.md / README-CN.md
```

## Contributing

Contributions are welcome when they improve clarity, correctness, or coverage: fix incorrect
explanations, improve hand calculations and visualizations, add focused exercises with assertions,
propose new notebooks for important agent topics, or help verify notebooks against a real LLM
endpoint.

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
