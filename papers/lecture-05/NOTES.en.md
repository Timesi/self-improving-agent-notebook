# Lecture 05 — Multi-step reasoning and planning (study notes)

> This file is the paper study notes for CS329A lecture 05, and the source material for the matching notebook (`05-multi-step-planning.ipynb`).
> Source: https://cs329a.stanford.edu/ (Autumn 2025 syllabus)
>
> Papers: LATS (2310.04406), ADaPT (2311.05772), SPRINT (2506.05745),
> Wider or Deeper (2503.04412), SWiRL (2504.04736).

## Lecture theme

The core problem this lecture solves: **a single LLM call can complete only one step; how can an agent finish tasks that need many steps, have branches, can fail, and need backtracking?**

It sits last in Part 1, picking up two threads from the previous three lectures:

- Lecture 2 covered why "test-time compute" works — spend compute at inference rather than training;
- Lecture 3 covered verifiers — score a solution with a scalar; lecture 4 covered tools and execution feedback — let the model interact with the environment and get observations.

Lecture 5 **assembles these three into a "multi-step reasoning/planning" system**: verifier scores become value signals during search, tool execution becomes actions, and test-time compute is spent on "searching many paths" rather than "walking only one path". It also plants a seed: should search be trained? SWiRL's step-wise RL is a contrast that "trains planning ability directly into the model", leading into Part 2's RL-scaling lectures.

The five papers of this lecture form four complementary views of "how to plan":

1. Decomposition (ADaPT) — split the task into subtasks, only when needed;
2. Tree search (LATS) — keep many candidate paths on a state tree, use MCTS to decide where to go;
3. Parallel execution (SPRINT) — pack independent reasoning steps and run them in parallel, cutting latency;
4. Adaptive branching (Wider or Deeper) — answer "widen (sample more) or deepen (refine)";
5. Training view (SWiRL) — instead of search at inference, train "when to decompose, when to call a tool" into the model.

## Close reading

### Paper 1: Language Agent Tree Search (LATS, arxiv:2310.04406, lats.pdf)

- **Core idea**: upgrade a ReAct-style "walk forward one step" agent into a **tree search**. The authors observe that many LLM tasks satisfy the "backtrackable" property MCTS needs: to return to any historical state, copy-paste previous text as input again; no world model is required. So Monte-Carlo tree search (MCTS) is placed on a language agent, and the same LLM plays three roles — Agent (sample actions), value function (evaluate states), reflector (summarize lessons from failure). This is the first work that unifies reasoning, acting, and planning in one frame.

- **Key algorithms / formulas**:
  - Six operations in a cycle: **selection → expansion → evaluation → simulation → backpropagation → reflection**, until success or budget of k trajectories is exhausted.
  - **Selection**: from the root, pick the child with max UCT each layer until a leaf. UCT:
    $$
    UCT(s) = V(s) + w\sqrt{\frac{\ln N(p)}{N(s)}}
    $$
    where $V(s)$ is child value, $N(s)$ visit count, $N(p)$ parent visit count, $w$ exploration weight.
  - **Expansion**: sample $n$ actions from $p_\theta$ at the current state, execute each in the environment, get observations, produce $n$ new children.
  - **Evaluation**: the value function is a weighted average of two terms
    $$
    V(s) = \lambda \cdot LM(s) + (1-\lambda)\cdot SC(s)
    $$
    $LM(s)$ is an LLM score of the state (ToT-style heuristic, but after environment feedback, more accurate); $SC(s)$ is a self-consistency score — actions that appear often when sampling multiple times in the same state are more reliable.
  - **Backpropagation**: along the root-to-leaf path update each node:
    $N(s_i)=N(s_{i-1})+1$, $V(s_i)=\dfrac{V(s_{i-1})N(s_{i-1})+r}{N(s_i)}$, $r$ is the final reward (success=1, fail=0).
  - **Reflection**: after failure, let the LLM write a natural-language reflection (what went wrong, what strategy to try next), store it in memory as context for later iterations — a richer "semantic gradient" than scalar $r$, and no training required.
  - Key hyperparameters: expansion samples $n=5$, trajectory budget $k=50$ (programming tasks $k=8$).

- **Key experimental results**:
  - HotPotQA (GPT-3.5): LATS (CoT+ReAct) EM=0.71, more than twice ReAct (0.32); acting-only LATS (ReAct) 0.63 > RAP(ReAct) 0.54 > Reflexion 0.51 > ToT(ReAct) 0.39.
  - HumanEval pass@1: GPT-4 LATS=92.7 (then SOTA), above Base LM 80.1 and Reflexion 91.0; GPT-3.5 83.8 vs Reflexion 68.1. MBPP (GPT-3.5) 81.1 vs RAP 71.4.
  - WebShop (GPT-3.5): LATS score 75.9 / success 38.0, above ReAct (53.8/28.0), Reflexion (64.2/35.0), even above trained IL+RL (62.4/28.7).
  - Game of 24: LATS(CoT) 0.44 > RAP 0.40 > ToT 0.20.
  - Ablation (HotPotQA): drop LM scoring → 0.37, switch to DFS → 0.42, drop reflection → 0.58, full 0.63 — every component is needed.
  - Cost: among tree-search methods, LATS expands the fewest nodes on success (k=50: 66.65 vs RAP 70.60 vs ToT 84.05), token cost below ToT/RAP.

- **Relation to the lecture theme**: the flagship of the "tree search" idea, answering "why plan" — greedy decoding looks at one path; errors accumulate and cannot look back; tree search keeps many candidates and balances exploration vs exploitation with UCT. It also shows how a verifier (L3) becomes a search value function, and how tool feedback (L4) becomes a search observation.

- **Demo-ready code points**:
  - Implement `Node` (state/visits/value/children) + UCT selection + backprop from scratch, no agent framework.
  - Run simplified LATS on a decidable environment (e.g. Game of 24: four numbers, +−×÷ to make 24): environment feedback itself gives reward; a scripted LLM only generates candidate expressions.
  - Visualize the search tree (which node is picked each round, how reward is backed up), seeing exploration and exploitation alternate.

### Paper 2: ADaPT — As-Needed Decomposition and Planning (arxiv:2311.05772, adapt.pdf)

- **Core idea**: the most basic planning is "plan then execute" (plan-and-execute), but a fixed plan cannot handle "some subtask actually cannot be done". ADaPT proposes **as-needed decomposition**: first let the executor try the whole task; only when the executor itself reports failure does the planner split the task into 3–5 subtasks, then recursively call the same procedure on each. Decomposition depth is jointly decided by task difficulty and executor ability, not preset.

- **Key algorithms / formulas**:
  - Three modules: **executor** (LLM, interacts with the environment, emits actions, must finally emit "task completed" or "task failed" as a success heuristic), **planner** (LLM, splits the task into 3–5 abstract subtasks combined with AND / OR), **controller** (a fixed recursive LLM program that wires the first two).
  - Recursion: `ADaPT(Task, k)`: if $k > d_{max}$ run only the executor; else first `Executor(Task)`, succeed then stop; fail then `Planner(Task)` yields `step1 AND step2 ...`, recurse `ADaPT(step, k+1)` on each step, combine sub-results by the logic.
  - AND means subtasks must run in sequence; OR means exploration cases (e.g. "find a mug on the table OR find a mug in the cabinet"), any success is enough.
  - Key hyperparameters: $d_{max}=3$ (ALFWorld/WebShop); TextCraft uses $d_{max}=4$ because recipes are deeper; plans stay short abstract 3–5 steps, allowing later as-needed splits.

- **Key experimental results** (GPT-3.5, success rate):
  - ALFWorld: ADaPT=71.6%, 28.3 points above ReAct (43.3) and Plan-and-Execute (43.3), 14.1 above Reflexion (57.5). Hard subclass pick2 (two pickups combined, long action history): ADaPT=52.9%, ReAct-family baselines all below 12% (about 4×).
  - WebShop: ADaPT=44.0, above LATS (38.0), Reflexion (35.0), ReAct (32.0), Plan-and-Execute (17.0).
  - TextCraft (a Minecraft-recipe text game introduced in the paper, naturally decomposable, recipe depth 2–4): ADaPT=52.0 vs Reflexion 32.0 vs Plan-and-Execute 27.0 vs ReAct 19.0, 33 points higher.
  - Analysis: success rises monotonically with $d_{max}$; ADaPT automatically splits finer when the executor is swapped for a weaker model (LLaMA-2, Lemur) — it really adapts "as needed".

- **Relation to the lecture theme**: this is the "decomposition" idea. Complementary to LATS: LATS expands "horizontally" (keep multiple candidate actions at the same state), ADaPT expands "vertically" (cut the task finer). ADaPT's key insight is that **the failure signal comes from the executor's self-report rather than an external reward** — the same theme as LATS reflection and verifier scores: let the LLM itself judge "how far we have gone, whether to look back".

- **Demo-ready code points**:
  - A synthetic task of configurable depth (TextCraft-style recipe tree); contrast expansion of three strategies: ReAct (one long path), Plan-and-Execute (split all at once), ADaPT (split only on failure).
  - Implement the recursive controller: parse AND/OR plans from a scripted LLM, combine subtask results by the logic.
  - Draw a "call tree": show ADaPT going deep only on one subpath, while plan-and-execute uses uniform depth throughout.

### Paper 3: SPRINT — Interleaved Planning and Parallelized Execution (arxiv:2506.05745, sprint.pdf)

- **Core idea**: long-CoT reasoning models (e.g. DeepSeek-R1) are accurate, but they emit a long sequential token stream, so latency is high. The authors analyze R1 reasoning traces and find many steps (reflection, task decomposition, trial-and-error exploration, several independent subcomputations) are independent and can run in parallel. SPRINT has two stages: **at training**, a data-cleaning pipeline rearranges original sequential traces into structured "multi-round plan + parallel execute" data, and finetunes the model to discover parallelism on its own; **at inference**, the model alternates as planner and executor, the planner generates a batch of independent subtasks, the executor runs them in parallel and syncs back to the main context, forming a rolling plan→execute→sync→replan loop.

- **Key algorithms / formulas**:
  - Data pipeline in four steps: (1) **step extraction** — GPT-4o splits each trace into steps, each with a planning stage $P_i$ and an execution stage $E_i$ (plan-only steps have $E_i=\varnothing$); (2) **DAG creation** — GPT-4o-mini judges dependencies $D=\{(S_i,S_j)\mid S_j \text{ depends on } S_i\}$; (3) **packing** — compute each step's stage number from dependencies
    $$
    \sigma(S_i)=\begin{cases}1, & S_i \text{ has no parent}\\
    \max_{S_p\in Parents(S_i)}\big(\sigma(S_p)+\mathbf{1}[E_p\neq\varnothing]\big), & \text{otherwise}\end{cases}
    $$
    same-stage steps pack in parallel; optimization: if the parent is plan-only ($E_p=\varnothing$), the child can join the parent's stage; (4) **filter + SFT** — keep only traces with parallelization ratio = steps/stages ≥ 1.5 (about 6000 → 1700), rearrange into `<Plan_i>`/`<prompt_i.j>`/`<execution_i.j>` tags, finetune DeepSeek-R1-Distill-Qwen-7B (5 epochs, lr 1e-5, 8×A100).
  - Sequential-token metric at inference (approximate wall-clock latency): $\text{seq tokens}=\sum_{i}\max_k(P_{i.k}+E_{i.k})$, i.e. each stage takes the longest path's token count.

- **Key experimental results** (DeepSeek-R1-Distill-Qwen-7B, trained on MATH-500):
  - MATH-500: SPRINT accuracy 92.5%, above same-data finetuned RFT (91.0%) and the base (89.1%), and 440 fewer sequential tokens on average (about 15%). On long traces needing >8000 tokens, sequential tokens drop by up to 39%.
  - Generalization (not trained on OOD data): Countdown accuracy 85.9% (>RFT 84.9%), sequential tokens down 53.5% (2284 vs 4917), up to 65% on long traces; GPQA-Diamond accuracy 51.0% (highest), sequential tokens down 10.8%, up to 45% on long traces.
  - Higher difficulty, larger parallelism gain: short traces (<2000 tokens) instead pay about 5% sequential-token overhead (tags and extra prompts).
  - Runtime estimate: MATH-500 36.92s per problem vs RFT 40.57s (9% faster); long traces 74.47s vs 120.54s (38% faster).
  - Vs SoT (single-round plan) almost 3× more tokens — showing that multi-round interleaved plan→sync→replan is key; one round of planning cannot hold mutually dependent subtasks.

- **Relation to the lecture theme**: this is the "parallel execution" idea. The previous two (ADaPT, LATS) care about **getting the task right** (correctness); SPRINT cares about **getting it fast** (latency) — finding the parallelism that already exists in reasoning traces. It is also the only work in this lecture that combines "interleaved plan and execute" with "training": instead of hand-defining a search structure, data teaches the model to split tasks at inference. Like SWiRL, it represents the line "ability can be trained into the model".

- **Demo-ready code points**:
  - Hand-write a step-dependency table (or parse from a scripted trace), implement $\sigma(S_i)$ stage assignment, parallelize same-stage steps, compare sequential tokens vs serial.
  - Draw the dependency DAG with networkx, showing how a plan-only parent folds children into the same stage.
  - With a scripted LLM, simulate one rolling "plan→parallel execute→sync" loop, count the longest-path tokens per stage, seeing where latency comes from.

### Paper 4: Wider or Deeper? — Adaptive Branching Tree Search (AB-MCTS, arxiv:2503.04412, wider-or-deeper.pdf)

- **Core idea**: the simplest effective inference-time scaling is repeated sampling (best-of-n, majority voting) — it only "widens" (sample new answers), not using external feedback; sequential refinement only "deepens" (improve an existing answer with feedback). Standard MCTS wants both, but branch width is a fixed hyperparameter, limiting expansion. AB-MCTS's key insight: **branching should be unbounded and adaptive** — at each node dynamically decide "widen" (generate a brand-new candidate from the current node, the GEN action) or "deepen" (refine an existing answer).

- **Key algorithms / formulas**:
  - Every node $N$ has a special **GEN child**, meaning "generate another new answer from here". Picking GEN widens; picking an existing child deepens. Unlike standard MCTS, already-expanded nodes can expand again; the number of branches is theoretically unbounded.
  - Selection uses **Thompson sampling** rather than UCT: UCT assumes arms (branches) are static, but GEN dynamically creates new arms, so it does not apply. Method: for each candidate action $a_j$ compute the posterior predictive $P_N(r\mid a_j)$ of score $r$, sample one score from each distribution, take the max.
  - Two implementations:
    - **AB-MCTS-M** (mixture model): each subtree is a "group", mixed Bayesian model $r_{N_{new},a_j}=\alpha_j+\sigma_y\epsilon$ (group-level intercept + instance noise), MCMC fits shared hyperparameters; GEN's $\alpha_0$ is inferred from the shared posterior.
    - **AB-MCTS-A** (node aggregation): a CONT node aggregates all "refine" children, exponential family + conjugate prior for closed-form updates — Gaussian variant (normal-inverse-χ², unbounded scores), Beta variant ([0,1] scores). Lighter, closer to standard UCT's philosophy.
  - Generation budget = LLM-call cap, experiments use $2^7=128$.

- **Key experimental results** (GPT-4o / DeepSeek-V3, LiveCodeBench, CodeContest, ARC-AGI, MLE-Bench):
  - Average rank: AB-MCTS-M 2.3, AB-MCTS-A (Gaussian) 2.7, AB-MCTS-A (Beta) 2.7, all better than repeated sampling (3.5), standard MCTS (4.2), sequential refinement (5.5).
  - LiveCodeBench (GPT-4o): AB-MCTS-M 38.9 vs repeated sampling 37.8 vs standard MCTS 36.7; CodeContest: 40.6 vs 37.9 vs 37.5.
  - ARC-AGI: repeated sampling is strong (needs wide exploration), AB-MCTS is comparable; when budget grows to $2^9=512$, repeated sampling flattens while AB-MCTS keeps rising — at large budget, adaptive branching "spends money where it matters".
  - MLE-Bench (GPT-4o): AB-MCTS-M average rank 1.3, best baseline (sequential refinement) 2.3.
  - Tree-shape analysis: AB-MCTS trees are wider than standard MCTS (because it can widen from any node), and they also deepen on promising branches — both directions adaptive.

- **Relation to the lecture theme**: answers the lecture's key "how to choose" question — widen or deepen? The answer is not a fixed policy, but an **online decision per node** (Bayesian posterior sampling). It is LATS's "next generation": still MCTS, but LATS's fixed "expand $n$ children" hyperparameter becomes adaptive branching. For teaching, it is the step from "can search" to "can search adaptively", and it points out UCT's limit in the LLM setting (arms generated dynamically).

- **Demo-ready code points**:
  - A mini adaptive branch: two actions (GEN / refine), each maintains a Beta(α, β) posterior, Thompson sampling picks the action, run several rounds on a scripted scorer, plot the choice trajectory.
  - Contrast three strategies' tree shapes (pure widen / pure deepen / adaptive) on the same synthetic scoring environment.
  - Hand-calculate "why UCT does not apply": construct a scene that dynamically generates new arms, compare UCB vs Thompson sampling.

### Paper 5: SWiRL — Synthetic Data Generation & Multi-Step RL (arxiv:2504.04736, swirl.pdf)

- **Core idea**: the previous four papers all work at inference (test time); SWiRL takes another road — **train "how to step, when to call a tool, when to wrap up" into the model**. Traditional RLHF/RLAIF is single-step optimization (only the final answer), but in multi-step tasks one intermediate error cascades to the end. SWiRL proposes **Step-Wise RL**: cut a multi-step trajectory into many sub-trajectories (prefixes) by each action, score each step separately with a generative reward model for RL, with no golden label or human annotation throughout.

- **Key algorithms / formulas**:
  - **Stage 1 data generation**: open-source Gemma 2 plus tools (search engine / calculator) iteratively generates multi-step trajectories; each step the model may freely emit CoT, call a tool, or give a final answer (tagged `<search_query>` / `<math_exp>` / `<answer>`). Generate 50,000 HotPotQA trajectories (10k problems × 5) and 37,500 GSM8K trajectories. Each k-step trajectory is split into k sub-trajectories (prefix + current action).
  - **Filter strategies**: four contrasts — no filter, process filter (Gemini 1.5 Pro Thinking judges "whether each step is reasonable"), outcome filter (final answer matches golden), process+outcome. **Process-only filter works best**.
  - **Stage 2 step-wise RL**: objective
    $$
    J(\theta)=\mathbb{E}_{s\sim T,\,a\sim\pi_\theta(s)}\big[R(a\mid s)\big]
    $$
    where reward $R(a\mid s)$ is scored by a generative reward model (Gemini 1.5 Pro) on "action $a$ given previous context $s$", no golden label; optimizer is the policy gradient already used for Gemma 2 finetuning.
  - At inference: iteratively let the model "call a tool or give an answer", retrieval via Gecko vector nearest neighbors, math via SymPy, at most 5 queries (QA) / 10 (math).

- **Key experimental results** (Gemma-2-27b finetuned):
  - Relative accuracy lift: GSM8K +21.5%, HotPotQA +12.3%, CofCA +14.8%, MuSiQue +11.1%, BeerQA +15.3% (about 15% on average).
  - Cross-task generalization: train only on HotPotQA (multi-hop QA), zero-shot lift GSM8K (math) 16.9%; reverse (GSM8K → HotPotQA) 9.2% — what is learned is general "how to reason", not task-specific.
  - **Process filter > outcome filter**: RL can learn from trajectories whose final answer is wrong but whose steps are reasonable (opposite of SFT — SFT needs a correct result, otherwise it can even be worse than the base; this echoes "SFT memorizes, RL generalizes"). SWiRL benefits most from process-filtered data that mixes positive and negative endings.
  - Scale: 1000 trajectories already show a clear gain; Gemma-2-27b has cross-task generalization, 2b/9b only in-domain.
  - Mechanism: SWiRL raises average process label (HotPotQA 82.5%→91.0%, GSM8K 87.5%→91.6%), so final accuracy really comes from better multi-step reasoning quality, not only distilling the reward model.

- **Relation to the lecture theme**: this paper moves "planning" from inference into training, the lecture's close and turning point. LATS/ADaPT use search and decomposition to help the "inference-time LLM" plan; SWiRL uses RL so the model "learns to plan itself" — two roads to the same place (SPRINT's SFT training also belongs to the second road). Its step-wise split (cut trajectories into prefixes) continues ADaPT's as-needed decomposition and LATS's value function (LLM scoring), but used as a training signal. SWiRL connects directly to "RL scaling" and "post-training evolution" after L6, the bridge from Part 1 to Part 2.

- **Demo-ready code points**:
  - With a scripted LLM, generate several scripted multi-step trajectories, implement "sub-trajectory split" (k steps → k prefixes), wrap a scripted process scorer, reproduce "process filter vs outcome filter" differences in data mix.
  - With a toy policy (small softmax classifier), show gradient differences of "step-wise reward vs reward only at the last step" (can simplify to weighted updates).
  - Count "SWiRL on only-correct / only-wrong / mixed-ending data", seeing that RL does not reject wrong-ending samples.

## Teaching thread (how a Stanford instructor might teach this)

The instructor's narrative order (matching notebook section order):

1. **Motivation: limits of single-step reasoning.** A concrete failure: let the agent directly answer "who is older: Glenn Hughes or Ross Lynch?" or play Game of 24. Point out three problems of greedy decoding — error propagation along the path, cannot look back after a mistake, does not use environment feedback. The question then arises naturally: in multi-step tasks, "thinking clearly about the next step" is no longer enough; we need "planning".

2. **Idea A: decomposition — cut a large task into small ones (ADaPT).** Intuition: cooking a dish we have never cooked, we open the recipe only when we hit a step we cannot do, rather than memorizing ten steps at the start. The core teaching is "as needed": the executor tries first, only on failure does the planner split, then recurse. Explain AND/OR logic, and why the success heuristic (executor self-reports completed/failed) is a feasible substitute reward. Where readers stall: why not split everything at once? — because we cannot know which subtask is hard; splitting too fine introduces wrong assumptions.

3. **Idea B: tree search — explore many paths at once (LATS).** Introduce MCTS: node=state, edge=action, UCT balances exploration and exploitation. First hand-calculate UCT (a small table), then the six operations, stressing the premise "LLM tasks are backtrackable" that lets MCTS work without training. Contrast decomposition and search: ADaPT is a "vertical slice" (lower per-step difficulty), LATS is a "horizontal expansion" (keep many possibilities); one handles "the task itself is hard", the other "the path is uncertain". Where readers stall: why is $\ln N(p)$ in a log and $N(s)$ outside in UCT? — an unexplored child's visit count is small, so $\ln N(p)/N(s)$ is large and it is tried first.

4. **Idea C: parallelism — do independent steps at the same time (SPRINT).** From "do it right" to "do it fast". A concrete example: reflection, decomposition, and trial-and-error reasoning steps are actually independent. Cover the data pipeline (split steps → build DAG → pack stages → filter and finetune), then the inference-time planner/executor rolling loop. Where readers stall: why not let the model generate one parallel plan at once? — SoT does that but generates mutually dependent subtasks; multi-round sync is required.

5. **Idea D: adaptive — widen or deepen (Wider or Deeper).** Upgrade the question: repeated sampling only widens, sequential refinement only deepens, standard MCTS has fixed width. AB-MCTS lets each node decide. Cover GEN nodes, Thompson sampling (why UCT fails), and the split of variants M / A. This folds all previous "search hyperparameter" questions into one "online decision" question.

6. **Contrast and close: training view (SWiRL) + how to choose.** Note that everything so far is an inference-time method; SWiRL trains planning into the model, paving Part 2. Finally a choice table: whether feedback is available (whether we can use a verifier/search), whether errors can backtrack (whether MCTS applies), whether latency is sensitive (whether to parallelize), budget size (branch width). Mental model for the reader: **decomposition solves task difficulty, search solves path uncertainty, parallelism solves latency, training solves the cost of "searching from scratch every time".**

## Code demo ideas (4–6)

All demos follow the project contract: unified `get_llm()` from `llm_client.py`; in scripted mode LLM output is a scripted trajectory, but algorithm logic (tree, decomposition, packing, sampling) must still run fully; core algorithms implemented from scratch in numpy.

1. **Hand-calculate the UCT selection formula (numpy, no LLM)**: a handwritten small tree (root with 4 children, each with N and V), implement `uct = V + w*sqrt(ln(N_parent)/N_child)`, print each child's UCT, pick the next node to expand. Change exploration weight w and observe the choice. Expected output: a UCT number table + the chosen node.

2. **Simplified LATS from scratch (scripted-LLM compatible)**: implement `Node` (state, visits, value, children) and a minimal six operations — UCT selection, sample n candidate actions with `get_llm()` (scripted mode returns scripted candidates), environment judges reward (e.g. Game of 24 whether the expression equals 24), backprop update, store a reflection text on failure. After 20 rounds output: best path, tree size, sequence of chosen nodes. Key design: reward comes from environment judgment, not LLM scoring, so scripted mode can run; the LLM only "emits candidate actions"; parsing is tolerant of scripted output.

3. **Visual contrast of ADaPT as-needed decomposition**: a synthetic recipe environment of configurable depth (goal item→sub-items→sub-sub-items, one branch set to "executor always fails"). Scripted LLM as executor (returns completed/failed) and planner (returns AND/OR plans); run ReAct (one long path), Plan-and-Execute (split all at once), ADaPT (split on failure, dmax=3). Output: call trees of the three strategies (who called whom at each layer) + action counts, seeing ADaPT go deep only on the bad branch.

4. **SPRINT DAG packing and sequential-token computation**: a handwritten step-dependency table (e.g. "compute subexpression A", "check", "reflect", including plan-only steps), implement the $\sigma(S_i)$ stage formula, pack same-stage steps; given token counts per step, compute sequential tokens serially vs in parallel (parallel = longest path per stage). Draw the DAG with networkx, annotate stages. Expected output: stage groups + serial/parallel token bar chart, showing the plan-only-parent optimization that folds children into the same stage.

5. **Adaptive depth-width choice (mini Thompson sampling, Beta conjugate)**: a simplified AB-MCTS — each node has only two actions: GEN (generate a new answer) and Refine (refine an existing answer). Maintain a Beta(α, β) posterior per action, sample scores from each posterior, take the max and execute; a scripted scorer returns 0/1 and updates the posterior. Run 50 rounds, plot "widen count vs deepen count" and the choice trajectory, contrast "pure widen (repeated sampling)" and "pure deepen (sequential refinement)" on a synthetic environment.

6. **SWiRL-style data split and filter**: with a scripted LLM generate several scripted multi-step trajectories (with search/answer action tags), implement "k-step trajectory → k prefix sub-trajectories"; write a scripted process scorer (0/1 on "is each step reasonable") and an outcome judge (does the last step match golden); count samples kept and label composition under four filter strategies (none/process/outcome/process+outcome). Expected output: a filter comparison table, showing "process-only keeps a mix of positive and negative examples", echoing the paper "RL can learn from wrong-ending samples".

## Exercise ideas (3)

1. **Complete UCT selection (fill-in + assert)**: given a tree (children's N, V, parent N), fill `uct = ...`, `assert` the chosen node index matches a hand calculation. Hint: $N(s)$ is the child's visit count, $N(p)$ the parent's; first think which node should be tried first.

2. **Complete LATS backprop (fill-in + assert)**: given a root-to-leaf node path (each node's N_old, V_old) and terminal reward r, fill the update formulas, `assert` the updated root N and V equal the hand-calculated values. Hint: $V(s_i)=\dfrac{V(s_{i-1})N(s_{i-1})+r}{N(s_i)}$, compute $N$ first then $V$.

3. **Complete SPRINT stage-number computation (fill-in + assert)**: given a step-dependency table (including a plan-only parent), fill the $\sigma(S_i)$ implementation, `assert` each step's stage matches the hand calculation, and `assert` that the plan-only parent's children join the same stage. Hint: only $E_p\neq\varnothing$ increments the parent's stage by 1; when the parent is plan-only, children share the parent's stage.

## References

- LATS: Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models (arxiv.org/abs/2310.04406, github.com/lapisrocks/LanguageAgentTreeSearch) — main tree-search thread of this lecture; first general frame combining MCTS with LLMs.
- ADaPT: As-Needed Decomposition and Planning with Language Models (arxiv.org/abs/2311.05772, allenai.github.io/adaptllm) — decomposition idea; as-needed recursive split and AND/OR plans.
- SPRINT: Enabling Interleaved Planning and Parallelized Execution in Reasoning Models (arxiv.org/abs/2506.05745, github.com/stanford-futuredata/Sprint) — parallel execution; DAG packing + planner/executor rolling loop.
- Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search (arxiv.org/abs/2503.04412, github.com/SakanaAI/treequest) — adaptive branching; GEN nodes + Thompson sampling, answering "widen or deepen".
- SWiRL: Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use (arxiv.org/abs/2504.04736) — training view; step-wise RL, process/outcome filter contrast, cross-task generalization.
- Prior concepts (lectures 2–4, for review): Chain-of-Thought (Wei et al., 2022); Self-Consistency (Wang et al., 2022); ReAct (Yao et al., 2023b); Tree-of-Thought (Yao et al., 2023a); Reflexion (Shinn et al., 2023); RAP: Reasoning via Planning (Hao et al., 2023); UCT (Kocsis & Szepesvári, 2006); AlphaGo's MCTS application (Silver et al., 2016).
