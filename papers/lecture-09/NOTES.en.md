# Lecture 09 — Post-training evolution: from chatbot to agent (study notes)

> This file is the study notes for CS329A lecture 9 (guest Melvin Johnson, Google DeepMind) "Evolution of Post-training from Chatbots to Agents", and the source material for the matching notebook.
> Source: https://cs329a.stanford.edu/ (Autumn 2025 syllabus)
> Note: this lecture has no assigned papers. It is "theme close reading" rather than "paper close reading". Material comes from a representative set — InstructGPT, RLHF, Constitutional AI, DeepSeek-R1, RLEF, WebRL, WebShop — plus WebSearch-supplemented agent-post-training lineage.

## Lecture theme

The core question this lecture answers: **pretraining only gives a model the ability to "continue text"; how does it become "an assistant that follows people", then "an agent that uses tools to finish tasks"? What does each step on this road change?** Post-training means all extra training after pretraining ends, to make the model serve a concrete goal: supervised finetuning (SFT), reinforcement learning from human feedback (RLHF), reinforcement learning from verifiable rewards (RLVR), and reinforcement learning directly on the finest-grained agent trajectories.

Its place in the course is the **closing overview** of Part 2 (Training & Evolution, L6–L9). The previous three lectures each give one puzzle piece: L6 train-time scaling gives the RL algorithm engine (STaR bootstrap, GRPO in-group advantage, DAPO engineering); L7 open-ended evolution shows the target form "an agent improves itself"; L8 search and deep research shows a concrete agent form. This lecture folds the first eight lectures' scattered methods into one **timeline**, explaining how the "signal source" migrates from human labels all the way to environment execution results — the premise that "self-improvement" can hold: **reward no longer depends on people, but on whether the task itself is verified successful**. Later Part 3 L10 (SWE agents) and L11 (memory) are this timeline landing on the engineering side.

Inviting Melvin Johnson (Google DeepMind) as guest has dual meaning: first, Gemini is one of the most aggressive frontier models of agent post-training, and his view is "what a real industrial post-training pipeline looks like"; second, DeepMind's own agent-RL work (milestone rewards, subgoal planning) fills in the "sparse trajectory-level reward" segment.

The whole lecture's narrative is a **migration of signal source**:
- SFT uses **human-written behavior demonstrations** as signal (learn "how to do");
- RLHF uses **human preference comparisons** as signal (learn "what is good");
- RLVR uses **verifiable answer correctness** as signal (learn "what is correct");
- Agent post-training uses **environment execution results** as signal (learn "what action sequence finishes the task").

Each migration happens because the previous signal is too expensive, too sparse, or too easy to "cheat" (reward hacking). The agent era pushes this chain to the end: reward comes directly from the environment.

## Theme close reading (by evolution stage)

### Stage 0: starting point — a pretrained model "cannot talk"

A pretrained model is trained on "continuing internet text": given context, predict the next token. It has learned grammar, knowledge, and a world model, but it does not follow instructions (instructions are continued as part of the text), does not stop (does not say "I don't know"), and does not use tools. This stage has no "post-training" act, only a base model.

- **Core goal**: let the model learn the interaction format "conversation".
- **Signal source**: none (no alignment signal yet).
- **Product**: GPT-3 / PaLM / various open-source base models.
- **Why post-training is needed**: the base model's ability is strong, but the "direction" is wrong — it optimizes next-token likelihood, not user intent.

### Stage 1: the SFT era (2021–2022) — imitate human demonstrations

**Core goal**: let the model learn the behavior "output according to an instruction" itself. Collect human-written (instruction, expected reply) demonstrations and do supervised finetuning (SFT) directly. FLAN (arXiv:2109.01652) does instruction tuning on a union of text-to-text tasks, showing that "teaching the model a behavior template" generalizes to new instructions.

**Method**: SFT is standard supervised learning; the loss is output-sequence cross-entropy
$$L_{\mathrm{SFT}}(\theta) = -\mathbb{E}_{(x, y_{\mathrm{demo}})\sim D}\left[ \sum_t \log \pi_\theta(y_{\mathrm{demo},t} \mid x, y_{\mathrm{demo},<t}) \right].$$
The gradient only pushes up the probability of "behavior humans demonstrated". **InstructGPT** (arXiv:2203.02155, OpenAI) is the benchmark of this step: about 40 annotators write ~13K demonstrations, finetune 16 epochs, yielding the SFT model. **Conclusions such as RLEF table 3a "SFT only improves on valid set" show SFT's boundary: it can only learn behavior that appeared in the data, and cannot exceed the demonstrator's level.**

**Signal source**: human-written demonstration text (behavior-level imitation signal).
**Product**: InstructGPT's SFT model, the FLAN series. **Limits**: (a) human demonstrations are expensive and inconsistent; (b) the model can only imitate, "cannot think" — it degrades on cases outside the demonstrations; (c) it cannot express preference strength of "some behaviors are better".

### Stage 2: the RLHF era (2022–2023) — align with human preferences

**Core goal**: not only learn "how to do", but learn "what is good". Have the model first emit several candidates, humans rank them, train a **reward model (RM)**, then optimize that RM with RL. **RLHF's idea source is Christiano et al. (arXiv:1706.03741, 2017) "Deep Reinforcement Learning from Human Preferences"**, replacing reward engineering with "preference learning"; InstructGPT industrializes it into a three-stage pipeline, and ChatGPT (GPT-3.5) is the direct product of that pipeline.

**Method** (InstructGPT's three stages):
1. **SFT**: human-demonstration finetuning, yielding initial policy $\pi_{\mathrm{SFT}}$.
2. **RM**: generate $K=4\text{-}9$ replies to the same prompt, annotators rank, train the RM with pairwise ranking loss:
$$L_{\mathrm{RM}}(\theta) = -\frac{1}{\binom{K}{2}}\mathbb{E}\left[\log \sigma\!\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right],$$
where $y_w$ is the preferred side. InstructGPT uses a 6B RM (175B RM training is unstable). Data volume ~33K prompts.
3. **PPO**: maximize RM score, with a KL constraint not to drift too far from the SFT model; the PPO-ptx variant also mixes in a pretraining gradient $\gamma \mathbb{E}[\log \pi_\theta(x)]$, mitigating "alignment tax" (alignment causing public-benchmark regression).

**Key numbers**: 1.3B InstructGPT is preferred over 175B GPT-3 in human evaluation (**a 100× parameter gap is closed**); 175B InstructGPT vs 175B GPT-3 preference rate 85%±3%; closed-domain hallucination rate from 41% to 21%.

**Signal source**: human preference comparisons (ranked candidate pairs). **Cost**: RLHF's RM is a **proxy objective** learned from a few thousand preferences, and will Goodhart — the model can fool the RM for a high score ("reward hacking"). Constitutional AI (arXiv:2212.08073, L4 close-read) wants to compress this cost: a principle list lets AI critique, revise, and score itself, compressing human labels from tens of thousands to one review. **Limits**: RM scores are a scalar scale, human ranking of long replies is noisy, "helpful vs harmless" fight each other, and the RM only measures "is this text good", not "can this action sequence finish the task".

### Stage 3: the RLVR / reasoning era (2024–2025) — verifiable rewards

**Core goal**: drop the learned RM, switch to **rule-based, verifiable rewards** — whether the answer matches the gold answer, whether the code passes tests. RLVR = Reinforcement Learning with Verifiable Rewards. **DeepSeek-R1** (arXiv:2501.12948) is the landmark: R1-Zero starts from a base model, **no SFT at all, pure RLVR**, optimizing two rule rewards "answer correct + format valid" with GRPO (L6 close-read), and **long-chain reasoning (thinking) emerges spontaneously** — the model learns to think first then answer, because thinking more makes it easier to score. This step shows: **as long as the reward is "right" and verifiable, the model can discover smarter strategies itself, without human demonstration.**

**Method**: reward is a rule function $R(y) = \mathbb{1}[\text{answer correct}] + \mathbb{1}[\text{format valid}]$ (DAPO uses $\pm 1$ rule rewards); advantage is GRPO in-group normalization $(R_i - \mathrm{mean}(R))/\mathrm{std}(R)$; the essential difference from stage 2 is **there is no RM to cheat** (DeepSeek-R1 paper's own words: there is no intermediate layer of reward hacking).

**Key numbers**: R1 series reaches ~79% on AIME 2024 (base before training <15%); DeepSeekMath-RL 7B uses RLVR to push MATH from 46.8% to 51.7% (already in L6); DAPO with Qwen2.5-32B reaches 50 on AIME.

**Signal source**: verifiable correctness (exact answer match, tests pass, rule judgment). **Limits**: only usable on tasks where "a criterion can be written" — math, code, puzzles; open problems ("write a good email") have no verifier. That is the motivation to migrate toward the agent era: **agent tasks are open, but the environment itself can be the verifier** (task success = tests pass / goal reached / state converges).

### Stage 4: the agent era (2024–2026) — environment feedback

**Core goal**: upgrade the training unit from "one text output" to "a whole agent trajectory" — the model repeatedly calls tools, reads observations, corrects actions in a loop; the training signal comes from **success/failure the environment gives at the end of the trajectory**, plus step-level signals produced during execution. This step answers the question L4 planted: RLEF already showed "execution feedback + RL" can teach a model to repair code (70B: 27.5→40.1 on CodeContests); agent post-training pushes the same idea to **long-horizon, multi-tool, real environments**.

**Method** (three nested grades of agent RL):
1. **Tool-call level**: "which API to call / what tool arguments to write" as the action. ToolLLM (arXiv:2307.16789) uses API-RL finetuning so the model learns 16000+ real APIs; WebShop (arXiv:2207.01206) trains a shopping agent with IL+RL (but L4 recorded its lesson: IL+RL 28.7 only ties IL 29.1, showing the environment signal was not strong enough then).
2. **Environment-feedback level**: RLEF throws the whole "code-repair MDP" into PPO; **WebRL** (arXiv:2411.02337) builds WebArena-style web tasks into a self-evolving online curriculum — first let the model explore online, use an **outcome-supervised reward model (ORM)** to judge subgoal success/failure, then process failed tasks into new training data. **Key numbers**: Llama-3.1-8B on WebArena-Lite 4.8%→42.4%, GLM-4-9B 6.1%→43%, above GPT-4-Turbo's 17.6%.
3. **Full-trajectory level**: OpenAI *Optimizing Agent Trajectories* (2025-10, technical report) applies RL directly to **complete agent trajectories run in a remote sandboxed Linux environment** (terminal, files, code execution); reward is a rule-based environment signal (whether tests pass, whether a terminal task succeeds), advantage is critic-free group-relative estimate, model scale to frontier reasoning-model level. DeepMind's *A Subgoal-driven Framework for Improving Long-Horizon LLM Agents* (MiRA) addresses agent RL's **sparse delayed reward**: offline RL with milestone dense rewards, Gemma3-12B on WebArena-Lite 6.4%→43.0%.

**Signal source**: environment execution results (tests pass, tool returns, terminal state, subgoal reached) — from "humans write demonstrations" to "humans rank preferences" to "rules judge", this step maximizes the signal's **availability** and **automatability**, which is why a self-improving agent can close the loop: **reward generation needs no humans at all**. **New problems**: sparse delayed reward makes credit assignment hard (the model does not know which step went wrong); the environment must be sandboxed, replayable, parallelizable; reward criteria must resist hacking (e.g. the model learns to "spam logs pretending to work"); policy distribution drift (online exploration forgets). These engineering problems connect directly to L10 SWE-Agent engineering and L14 agent evaluation.

### One table to close: signal-source migration across four stages

| Stage | Signal source | Data form | Method | Goal | Representatives |
|---|---|---|---|---|---|
| SFT | Human demonstrations | (instruction, expected reply) | Cross-entropy finetune | Learn "how to do" | FLAN, InstructGPT-SFT |
| RLHF | Human preferences | Ranked candidate replies | RM + PPO | Learn "what is good" | InstructGPT, ChatGPT |
| RLVR | Verifiable correctness | Rule criteria | GRPO (no RM) | Learn "what is correct" | DeepSeek-R1, DAPO |
| Agent RL | Environment execution results | Whole trajectory + success/fail | Trajectory-level RL / milestone rewards | Learn "what action sequence finishes the task" | RLEF, WebRL, MiRA, OpenAI Agent RL |

## Teaching thread (how Melvin Johnson might teach this)

Melvin is from industry; the most natural method is **follow one model's life** — from a base model being born to becoming an agent, growing a segment each time a new signal is fed, each "cannot feed it in" being the next stage's motivation:

1. **Start from the base model, build the intuition that "the model cannot talk"**: a base model is a text continuer; feed it an instruction, it continues a piece that "looks like answering but did not follow the instruction". A demo: give a base model an instruction, observe how it treats the instruction as body text. **Motivation**: need to teach it the behavior "conversation".

2. **SFT: teach it "how to do"**. Use human demonstrations as supervision directly. Make clear SFT is cross-entropy; it does one thing — push up the probability of demonstrated behavior. Then immediately give its ceiling: the model only imitates seen behavior; demonstrations contain no "judge good vs bad" information. A counterexample: after SFT the model has no concept of "helpful vs harmful" and does whatever is asked. **Motivation**: need the dimension "good vs bad".

3. **RLHF: teach it "what is good"**. Introduce the three stages (SFT→RM→PPO), focusing on the RM learning a **proxy objective**, the KL constraint preventing the model from going too far, PPO-ptx fighting alignment tax. Key intuition: the RM is a scalar, an approximation, and **can be fooled**. Give a reward-hacking example (the model learns boilerplate for high scores), introduce the cost problem Constitutional AI wants to solve (L4 already close-read, one sentence here). **Motivation**: the RM can be cheated, and it only rates "text quality", not "whether the task succeeded".

4. **RLVR: replace the RM with a verifier**. The most important flip of the session: rather than learn a reward function, directly check whether the answer is correct. Cover R1-Zero's drama — **pure RL, no SFT, thinking emerges spontaneously** — showing "when the signal is right, smart strategies need not be taught". Point out RLVR's boundary: only tasks where a criterion can be written apply. **Motivation**: open tasks have no gold answer, but the environment can be the verifier.

5. **Agent post-training: reward comes from the environment**. Swap "a piece of text" for "a trajectory", swap "reward model" for "environment criteria". Use RLEF (repair code), WebRL (web shopping), MiRA (milestone rewards), OpenAI's remote-environment RL to make clear: the action space expands from "next token" to "tool-call sequences", reward from "text quality" to "task success". **Where readers most easily stall**: (a) thinking agent RL and RLVR are the same thing — RLVR's reward is a criterion written before the task starts, agent RL's reward is produced by running the environment, possibly sparse and delayed; (b) not seeing why credit assignment is harder — a trajectory has dozens of steps, only the last gives +1, which intermediate step went wrong has no direct signal; (c) thinking SFT/RLHF are obsolete — a real pipeline is **stacked**, an agent model also SFT first, then align, then RLVR, and only then trajectory-level RL.

6. **Close on the course map**: map the whole timeline back to the first eight lectures — L4's RLEF/CAI is the seed of agent feedback, L6's GRPO/DAPO is the algorithm engine of agent RL, L8's deep research is a concrete form of "the environment as verifier"; then preview that L10/L11/L14 will repeatedly use the three words "trajectory, environment feedback, evaluation". One sentence to close the whole course: "The history of post-training is the history of the reward signal moving from human hands, to a verifier's hands, and finally to the environment's hands."

## Code demo ideas (5)

1. **Hand-calculate loss forms of SFT / RLHF / RLVR**: the same toy data (one instruction, several candidate outputs, labeled: demonstration text / preference pair / answer right-wrong), implement three losses in numpy — SFT cross-entropy (learn only the demonstration), RLHF PPO clipped surrogate + KL (optimize RM score but limit drift), RLVR GRPO in-group advantage ($r$ normalized then weight each group's tokens). Print the same candidate's gradient direction and magnitude under different paradigms. Expected output: SFT only pushes up "demonstrated" probability, RLHF pushes up "preferred but not too far", RLVR pushes up "closer to correct within this group" — seeing the three eras' optimization-target differences.

2. **Numpy contrast of PPO vs GRPO on a toy bandit**: a 10-arm bandit (action = emit an answer, environment judges right/wrong), implement (a) PPO with a value network + GAE, and (b) GRPO with in-group normalized advantage, iterate several rounds and contrast sampling-distribution convergence. Then reproduce a famous RLVR failure mode: **all-correct / all-wrong group advantage all 0**, gradient zero, the batch is wasted (leads to DAPO Dynamic Sampling and RL-ZVP). Expected output: PPO needs the value network to converge, GRPO is critic-free but all-correct groups stall — contrast curves.

3. **Visualization of reward-signal sources**: draw a "signal-source migration" figure — each of the four stages' data cost (annotation cost), automatability, dense/sparse, whether hackable; bar or scatter projecting the four signals "human demonstration / human preference / rule criteria / environment result" onto a "cost vs information" plane. Expected output: a figure that shows at a glance "why we migrated all the way to environment feedback"; annotate representative work of each stage.

4. **Credit-assignment contrast of trajectory-level vs step-level reward**: a toy 3-step environment (only step 3 decides success/failure), train with REINFORCE under "trajectory-level reward" (±1 at the end) vs "step-level dense reward" (a subgoal signal each step, simulating MiRA milestone rewards), compare gradient variance and steps to converge. Expected output: sparse trajectory-level reward has huge variance and converges slowly; dense step-level reward has small variance and converges fast — directly explaining "why agent RL needs process signals / milestone rewards".

5. **Flywheel simulation of reward-signal sources (toy)**: do not train a large model; with a rule environment simulate a SWE-bench/WebArena-style loop — "model emits an action sequence → environment judges success/failure → success/failure as reward → filter successful trajectories → sample again" (STaR/WebRL skeleton). Run a multi-round loop with llm_client scripted mode, show "environment feedback raises the share of successful trajectories round by round", and demo two fates of failed tasks (discard vs rewrite into new training samples, corresponding to WebRL's self-evolving curriculum). Expected output: success-rate rising curve + data-flywheel inflation figure.

## Exercise ideas (3)

1. **Four-stage post-training classification fill-in**: given 6 training-setup descriptions (e.g. "cross-entropy finetune directly on human-written (prompt, reply)", "GRPO with a rule function judging whether the answer matches gold", "annotators rank 6 candidates then train an RM then PPO", "run a full agent trajectory in a sandbox, tests-pass as reward"), fill in which stage (SFT/RLHF/RLVR/Agent RL), assert classification is correct. Hint: first find the "signal source" — demonstration, preference, rule criteria, or environment result.

2. **RLVR advantage fill-in**: given `rewards = np.array([1.0, -1.0, 0.0, 1.0])`, fill in in-group normalized advantage `(r - mean) / std`, assert a known result; then assert "all-correct group `[1,1,1,1]` advantage is all 0, gradient is zero", and answer why this wastes compute (leads to Dynamic Sampling). Hint: np.std's ddof must match the tutorial convention; compute mean first, then subtract mean divide std.

3. **Write environment reward functions for a toy agent task**: a 3-step tool-call task (e.g. "query database → filter → return result"), each step has a subgoal (whether the subgoal is reached is decidable), fill in two reward functions — `trajectory_reward` (only +1 on final success) and `milestone_reward` (+0.3 per step subgoal, +1 on final success), assert both numbers, and print the gradient-signal difference of "sparse vs dense". Hint: a milestone reward is giving the model a "process signal", which is MiRA's motivation; first judge whether each step's subgoal is reached, then accumulate.

## References

- Training language models to follow instructions with human feedback (arXiv:2203.02155, InstructGPT) — benchmark of the SFT→RM→PPO three-stage RLHF, ChatGPT's technical predecessor; the key number that 1.3B matches 175B
- Deep Reinforcement Learning from Human Preferences (arXiv:1706.03741, Christiano et al. 2017) — RLHF's idea source: learn a reward model from human preferences, rather than handwrite rewards
- Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073) — compress human-label cost with AI feedback; already close-read in L4
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948) — RLVR landmark: R1-Zero no SFT, pure RL, thinking emerges spontaneously; original source of "no hackable RM"
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (arXiv:2402.03300) — GRPO proposed; already close-read in L6
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale (arXiv:2503.14476) — RLVR engineering (all-correct/all-wrong group zero gradient, Clip-Higher, etc.); already close-read in L6
- RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning (arXiv:2410.02089) — "execution feedback + RL" teaches a model to repair code, the smallest prototype of agent-trajectory RL; already close-read in L4
- WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning (arXiv:2411.02337) — self-evolving online-curriculum RL for web agents; Llama-3.1-8B 4.8%→42.4% @ WebArena-Lite
- WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents (arXiv:2207.01206) — early IL+RL web-shopping agent, contrast starting point of agent RL
- ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs (arXiv:2307.16789) — early representative of tool-call training
- OpenAI *Optimizing Agent Trajectories* (2025-10 technical report, published with the blog "Advancing RL for agentic systems") — frontier case of full-trajectory RL in a remote sandboxed Linux environment; reward from tests/terminal success
- DeepMind *A Subgoal-driven Framework for Improving Long-Horizon LLM Agents* (MiRA/SGO) — milestone dense rewards for agent RL's sparse delayed reward; Gemma3-12B 6.4%→43.0% @ WebArena-Lite
- FLAN: Finetuned Language Models are Zero-Shot Learners (arXiv:2109.01652) — representative of the instruction-tuning / SFT era
- CS329A syllabus (https://cs329a.stanford.edu/) — this lecture's place in the course; guest Melvin Johnson (Google DeepMind)
