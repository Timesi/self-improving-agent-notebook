# Lecture 04 — Tool use and code feedback (study notes)

> This file is the paper study notes for CS329A lecture 4, and the source material for the matching notebook.
> Source: https://cs329a.stanford.edu/ (Autumn 2025 syllabus)

## Lecture theme

The core problem this lecture solves: **a single LLM call can only emit a piece of text; how can an agent use information from the outside world (knowledge bases, code executors, users) to finish a task?**

Its place in the course is deliberate. In the first three lectures (L1–L3) the model is "static": L2 is test-time compute, via more samples and voting; L3 is a verifier, letting the model check its own output. None of those methods lets the model interact with the environment. This lecture introduces a **closed loop**: the model emits an action → the action runs in the environment → the observation is fed back → the model emits again. This loop is the foundation of every later agent system (L5 planning, L10 SWE, L15 autonomy).

The lecture unfolds along "**where does feedback come from**" in three layers:

1. **Environment feedback** (ReAct): tool-execution observations as context, i.e. "act to reason / reason to act". This is prompt-level feedback.
2. **Execution feedback + training** (RLEF): turn "did the code pass tests" into an RL reward, and train the model end-to-end to iterate and repair. This is training-level feedback.
3. **AI feedback** (Constitutional AI): no longer depend on human labels; let AI self-critique, revise, and score each other against a written set of principles (a constitution). This is "let AI supervise AI" feedback.

The three papers form one line: **first let the agent obtain feedback (tools/execution), then let feedback enter training (RL), finally let the feedback provider also be AI (the philosophical base of self-improvement)**. This lecture is therefore both the close of Part 1 Foundation and the lead-in to Part 2 Training (RL and evolution in L6–L9).

---

## Close reading

### Paper 1: ReAct: let language models reason and act together (arxiv:2210.03629, react.pdf)

- **Core idea**
  Previous two camps each miss half: Chain-of-Thought (CoT) lets the model think, but thinking is a "static black box" that never touches the outside world, so it hallucinates and errors propagate along the chain; the pure-action camp (Act) lets the model interact with the environment but does not plan goals or maintain working memory in language, so it gets lost on long tasks. ReAct's observation is simple: **put "language" into the action space as well** — between actions the model emits a free-text Thought. That text does not change the environment; it only updates context, to decompose goals, inject common sense, track progress, and handle exceptions. The result is a trajectory of alternating `Thought → Action → Observation`, where reasoning (reason to act) and acting (act to reason) support each other. The analogy is "someone cooking in a kitchen": between chopping they mutter "now boil water", "no salt, use soy sauce", while opening the fridge and flipping a recipe to support the next step.

- **Key formulas / algorithms**
  - Formalization: expand the action space to $\hat{A} = A \cup L$, $L$ the language space. Thought $\hat{a}_t \in L$ produces no environment observation, only a context update $c_{t+1} = (c_t, \hat{a}_t)$. Frozen PaLM-540B few-shot prompting; each in-context example is a human-written "action+thought+observation" trajectory.
  - Three Wikipedia API actions (deliberately weaker than a real retriever, to mimic human search): `search[entity]` (first 5 sentences of the entity page; if missing, 5 similar entities), `lookup[string]` (next sentence on the page containing that string, like browser Ctrl+F), `finish[answer]` (end the task and give the answer).
  - Key prompt format (each line numbered):
    ```
    Question: <question>
    Thought 1: <reason in language>
    Action 1: Search[<entity>]
    Observation 1: <tool return>
    Thought 2: ...
    Action 2: Finish[<answer>]
    ```
  - Two task types use different "thought density": knowledge-intensive tasks (HotpotQA/Fever) use dense thought (write every step); decision tasks (ALFWorld/WebShop) have many actions, so thought appears sparsely at key positions, letting the model decide when to think and when to act.
  - Fallback heuristics combined with CoT: `ReAct→CoT-SC` (if ReAct has no answer within several steps, fall back to CoT self-consistency voting; 7 steps on HotpotQA, 5 on Fever); `CoT-SC→ReAct` (if the majority-vote winner count < n/2, i.e. internal knowledge is not confident, fall back to ReAct to retrieve).

- **Key experimental results** (PaLM-540B prompting, concrete numbers)
  - HotpotQA (EM): Standard 28.7, CoT 29.4, CoT-SC 33.4, Act 25.7, **ReAct 27.4**, ReAct→CoT-SC 35.1, CoT-SC→ReAct 34.2. Fever (Acc): **ReAct 60.9 > CoT 56.3**; both combinations reach 62–64.6.
  - Core gain of the combinations: performance that CoT-SC needs 21 samples for, ReAct+CoT-SC needs only 3–5.
  - ALFWorld (134 episodes, task-specific): ReAct best-of-6 **71%**, Act best-of-6 45%, BUTLER (trained on 105 expert trajectories) 37%, IM-style ablation (ReAct-IM) 53%. ReAct's worst trial (48%) still beats the best Act and BUTLER.
  - WebShop: ReAct success **40.0**, Act 30.1, IL 29.1, IL+RL 28.7, human experts 59.6. About 10 absolute points above IL+RL.
  - Hallucination attribution (human labels on 50 ReAct + 50 CoT HotpotQA trajectories): **56%** of CoT failures are hallucinated reasoning/facts, ReAct **0%**; but ReAct has 23% more "retrieval ineffective" errors and a higher reasoning-error rate (47% vs 16%, including dead loops that repeat the same action). That is why the two should be combined.
  - Finetuning (HotpotQA): finetuning on only 3,000 correct ReAct trajectories, PaLM-8B/62B ReAct already beats all PaLM-540B prompting methods. Under prompting ReAct ranks last; under finetuning it ranks first — "reason while retrieving" is a generalizable skill worth training, not only in-context learning.

- **Relation to the lecture theme**
  ReAct is the tone-setting paper: the first time pure prompting organizes "reason + act + observe" into an interpretable closed loop, showing that **thinking without acting** (CoT hallucination, error propagation) and **acting without thinking** (Act getting lost) are both insufficient. It also gives the tool-interface paradigm (search/lookup/finish) and prompt template, the direct blueprint for implementing the agent loop from scratch in the notebook.

- **Demo-ready code points**
  - Write a ReAct loop from scratch: maintain `messages`, call `get_llm().chat()` for a reply, regex-parse `Action: Search[...]/Finish[...]`, execute a local mini "wiki" tool, splice the observation back, loop until `Finish` or a step limit.
  - Reproduce "think vs act": on the same question, CoT-only, Act-only, and ReAct prompts, compare trajectories (in scripted mode, note that output is a scripted placeholder).
  - Reproduce failure modes: construct a multi-hop question that is wrong without retrieval and right with retrieved material; observe how hallucination is suppressed by injecting observations.
  - Copy the few-shot examples from paper appendix C as the prompt template (HotpotQA Search/Lookup/Finish trajectories can be in-context examples directly).

### Paper 2: RLEF: grounding code LLMs with execution feedback + RL (arxiv:2410.02089, rlef.pdf)

- **Core idea**
  Letting an LLM generate code and execute it repeatedly is a common "self-repair" practice, but existing work (Olausson 2024, Kapoor 2024) shows: **under a fixed sampling budget, independent sampling is often stronger than "repair while trying"** — because the base model does not really use execution feedback and often re-emits the same wrong code. RLEF treats this as a training problem, not a prompting problem: model "multi-turn generation + execution feedback" as an MDP, train end-to-end with PPO, and let the model **learn** to read feedback and change code. In one sentence: not a better scaffold for the agent, but train "using feedback" into the weights.

- **Key formulas / algorithms**
  - Iterative code synthesis as an MDP: initial observation $o_0$ is the problem statement, action $a_t$ is a text reply, observation $o_t$ contains previous actions and **execution feedback**. An episode ends when all public tests pass or a turn limit is reached.
  - Two test sets: **public tests** provide execution feedback at train/inference and decide which solution to pick; **private tests** (hidden) decide final correctness. The split prevents the model from copying test outputs into the answer (shortcut), and saves compute.
  - Reward (PPO, undiscounted $\gamma=1$):
    $$
    R(s_t,a_t) = r(s_t,a_t) - \beta \log\frac{\pi(a_t|c_t)}{\rho(a_t|c_t)},\qquad
    r(s_t,a_t) = \begin{cases} 1, & \text{episode ends and all tests pass}\\ -1, & \text{episode ends and some test fails}\\ -0.2, & a_t \text{ contains no valid code}\end{cases}
    $$
    $\beta$ trades off task reward and KL; a small −0.2 penalty for illegal code on non-final turns mitigates a known failure mode.
  - Engineering: policy is token-level, value function is turn-level (predict whole-turn value from the last token of that turn's prompt; all tokens in a turn share one advantage); the KL term uses the **geometric mean** of token probabilities rather than a product, to cancel bias toward short replies.
  - Execution-feedback template (paper appendix C):
    ```
    Your code failed the following tests:
    - input '...' failed: Expected output '...' but got '...'
    - input '...' failed: <stacktrace>
    - input '...' failed: Execution took too long.
    - input '...' failed: Out of memory.
    Give it another try.
    Your code should be enclosed in triple backticks like so: '''python ...'''
    ```
  - Training config: Llama 3.1 8B/70B Instruct as initial policy, turn limit = 3, 8B trained 12,000 steps, 70B 8,000 steps.

- **Key experimental results** (CodeContests, concrete numbers)
  - 1@3 (one rollout, at most 3 replies): 70B from **27.5 → 40.1** (test), 8B from **10.5 → 16.0**. 10@100: 70B test 50.3 → 54.5.
  - Beats previous SOTA: 70B+RLEF test 40.1, above AlphaCodium(GPT-4)'s 29, with sample budget down from 100 to 3 (an order of magnitude less); 8B+RLEF (16.0) beats AlphaCode 9B (13.3), which used 1,000 samples.
  - Inference behavior (figure 3): RLEF models have fewer first-turn errors, more reliable later-turn repairs, and larger code edits; base models often ignore errors and re-emit the same code. **Random-feedback ablation**: replace feedback with execution results from an unrelated problem, repair ability drops substantially (pass@1 falls and the drop grows with turn limit) — showing that RLEF models are truly "reading feedback", not lucking out via sample diversity.
  - Training-method comparison (table 3a): few-shot prompting harms Instruct models (−); SFT only improves on valid; RLEF leads across the board. Single-turn vs multi-turn training (table 3b): multi-turn training (MT) lifts the most, and the gain transfers at inference to HumanEval+, MBPP+.
  - Landing point: under a fixed sampling budget, RLEF is the first time "iterative repair" stably beats "independent sampling", moving the model from "needs expensive prompt-engineering scaffolds" toward "domain finetuning in exchange for general ability".

- **Relation to the lecture theme**
  RLEF is the advanced form of "code execution as a feedback signal": ReAct feeds execution results to the model as context (read once, use once); RLEF turns "whether we use execution feedback" into a training objective (PPO reward). It answers "how an agent learns to self-improve" — not by writing more repair prompts, but by letting environment criteria enter the gradient. It also previews L6 train-time scaling (GRPO and so on) and post-training evolution.

- **Demo-ready code points**
  - In the notebook, build an "iterative code repair" simulator (no training): a problem statement plus public tests, loop "generate code → local `exec` of tests → format errors back" until pass or turn limit. Reproduce the paper observation: prompted models often do not change or change wrongly.
  - Implement the reward and advantage from scratch: a mini policy gradient (REINFORCE) on a toy "repair a function" environment, showing how "positive reward only if execution passes" changes the sampling distribution (numpy is enough; no external RL framework).
  - Reproduce the random-feedback ablation: randomly replace feedback in the same loop, compare repair success, seeing that "feedback must be related to the error".

### Paper 3: Constitutional AI: harmlessness from AI feedback (arxiv:2212.08073, constitutional-ai.pdf)

- **Core idea**
  RLHF needs tens of thousands of human preference labels, and the labels are opaque — nobody can read the "goal" behind tens of thousands of comparisons. CAI's starting point: **can we give the model only a "constitution" written in natural language (about ten principles), and let all remaining feedback be produced by AI itself?** Human supervision is compressed to "review and approve a principle list"; the rest is the model's self-critique, revision, and mutual scoring. This is the extreme form of "scaling supervision": let AI supervise AI.

- **Key formulas / algorithms**
  - Two stages, SL then RL:
    - **SL stage (Critique → Revision → SFT)**: feed red-team harmful prompts to a model trained only for helpfulness; sample often-harmful initial replies; have the model **self-critique** against a randomly drawn principle from the constitution, then **revise** according to the critique, possibly for several rounds (redraw the principle each round); finally SFT on the "revised replies". Prompt template:
      ```
      Critique Request: Identify specific ways in which the assistant's last response is
      harmful, unethical, racist, sexist, toxic, dangerous, or illegal.
      Critique: ...
      Revision Request: Please rewrite the assistant response to remove any and all
      harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
      Revision: ...
      ```
      Example: on "help me hack my neighbor's wifi", the initial reply suggests tools; after revision it becomes "this invades privacy and may be illegal; I strongly advise against it". Few-shot examples prevent the model from confusing "should I output a critique or a revision".
    - **RL stage (AI pairwise comparison → preference model → RLAIF)**: the SL-CAI model from the SL stage generates a pair of replies per harmful prompt; concatenate "prompt + reply pair + one random principle" into a multiple-choice (`Options: (A) ... (B) ... The answer is:`); the feedback model computes log probabilities of (A)/(B) as soft labels; mix human helpfulness labels + AI harmlessness labels to train a preference model (PM); finally PPO with the PM as reward. The whole RLHF pipeline is unchanged; only the source of harmlessness labels switches from humans to AI.
  - Constitution: 16 harmlessness-related principles; each revision/score randomly draws one. Example principle: "Which of these assistant responses is less harmful? Choose the response that a wise, ethical, polite and friendly person would more likely say." Multi-principle ensemble makes the preference model more robust (using the same principle for all labels is worse).
  - CoT version: the feedback model is guided with `Let's think step-by-step:`, writes reasoning first then picks; CoT probabilities are almost always 0/1 and overconfident, so they need to be **clamped to 40%–60%**, otherwise RL learns to emit extreme replies.

- **Key experimental results** (concrete numbers)
  - Data scale: 42,496 human red-team prompts + 140,335 model-generated = 182,831; 4 critique-revision groups per prompt.
  - Conclusion 1 (figures 2/3/8): at the same helpfulness level, RL-CAI is **significantly more harmless** than HH RLHF trained with human feedback, and crowdworkers prefer RL-CAI over the previous HH model. Key: no human harmlessness labels were used.
  - Conclusion 2 (figure 5): preference-model scores rise monotonically with revision rounds (harmlessness score keeps rising from 0 to 4 rounds), but pure helpfulness scores fall — multi-round revision costs a bit of helpfulness.
  - Conclusion 3 (figure 7): **critiqued revision beats direct revision** (skip critique, revise directly); the gap is clear on small models and almost gone on large models; critiques occasionally exaggerate, but revisions are overall more harmless, and critiques add transparency to the decision.
  - Conclusion 4 (figure 4): the larger the model, the more accurately AI identifies "more helpful/honest/harmless" replies; CoT helps substantially, approaching preference models trained on human labels — prior evidence that "AI-provided feedback is feasible".
  - Known failure modes: RL overtraining leads to Goodharting (over-reacting to red-team prompts, stacking boilerplate such as "you are valid, valued, and cared for"); mitigated by rewriting principles, principle ensemble, soft labels + clamp. Feedback-model soft labels are well calibrated (figure 9).

- **Relation to the lecture theme**
  CAI takes "the source of feedback" to the extreme: tool feedback comes from the environment, execution feedback from rules, and here feedback comes from **AI itself**. It is the philosophical high point of this lecture's "learn from feedback", and a direct source of "self-improving agent" — if an agent wants to self-improve, it eventually needs a feedback channel that does not depend on human labels. For the course: show how a principle list can drive both SL (critique-revision) and RL (RLAIF), and use Chain-of-Thought as a "checkable decision process".

- **Demo-ready code points**
  - With `get_llm()`, demo a critique → revision pipeline: feed a harmful prompt, request an initial reply, then a critique against a chosen principle, then a revision; print the three segments for comparison.
  - Reproduce the multi-round revision curve: run the same prompt for 0–4 rounds of "repeat revision", observe harmlessness/length changes (clear under a real API; placeholders under scripted mode).
  - Demo "AI as judge": for two candidate replies to the same prompt, concatenate a multiple-choice under a principle and let the model pick the more harmless one (soft label via scripted mode or parsed logits).
  - The principle list is the "constitution": let the reader write 3 principles, replace the paper's 16, observe behavior differences — the controllability of "humans only review principles".

---

## Teaching thread (how a Stanford instructor might teach this)

A narrative from "static" to "closed loop" to "self-improvement"; three papers, three steps:

1. **Motivation first: why thinking without acting is not enough.** Start from L3's verifier: however accurate the verifier, the model cannot invent facts that are not in the training data. Give a HotpotQA-style multi-hop question, show CoT inventing a wrong answer with a straight face (hallucination), then: "if the model could look up Wikipedia, would it still be wrong?" — **what the model lacks is not reasoning, but a channel to obtain information.**

2. **Introduce ReAct: think + act = closed loop.** Use a life analogy (cooking) to make Thought useful: it is not decoration; it is "working memory + plan + exception handling". Walk a `Thought → Action → Observation` trajectory by hand, stressing how the three action primitives (search/lookup/finish) cover "find material, look at details, wrap up". The easiest stall: **should the observation be generated by the model?** Repeat that Observation must come from real tool execution, inserted text, not model continuation. Then two counterexamples: Thought-only CoT (hallucination) and Action-only Act (lost in ALFWorld's 50 rooms). Close with the paper's ablation numbers: ReAct vs Act average relative lift 62%.

3. **Tool feedback as a signal: from context to training.** Pose the question: ReAct always splices feedback into the prompt now; does the model "know how" to use feedback? RLEF's answer is not necessarily — table 3a says few-shot prompting is even harmful; the base model ignores errors and repeats the same code. So turn "read feedback, change code" into an RL objective: explain the two test sets (public feedback, private scoring) and the reward (1/−1/−0.2 + KL), and show a sample dialogue of "timeout → add cache" repair. Stress a counter-intuitive conclusion: **under a fixed sampling budget, only a trained model is worth multi-turn iteration; otherwise independent sampling is better.**

4. **AI feedback: the feedback provider need not be human.** From RLEF's reward being "rules", ask: what if the task has no automatically executable criterion? CAI's answer is let AI self-critique, revise, and score each other against a constitution. Read the critique-revision template and example (hack wifi), explain how the SL and RL stages turn "principles" into a "preference model" then into "reward". Close on scaling supervision: human supervision compresses from "tens of thousands of labels" to "a principle list", which is the premise that a self-improving agent can keep evolving.

5. **String it together: three layers of feedback sources (environment / execution / AI), corresponding to planning after L5 and Part 2 RL.** Preview L5 using search for planning, and L6 GRPO reusing this kind of "automatically verifiable" reward design.

Three places readers tend to stall: a) treating Observation as model output; b) thinking RLEF's value is the prompt template rather than the training objective; c) mixing up what CAI's SL stage (SFT on revised results) and RL stage (preference-model scoring) each change in the model.

---

## Code demo ideas (3–6)

1. **Implement a ReAct loop from scratch (with tool execution and observation injection)**: the core demo of this lecture. Write `parse_action(text)` ourselves (regex extract `Action: Search[...]` / `Finish[...]` from the model reply) and a mini tool set (a local "encyclopedia" dict of a few entries + `search`/`lookup`/`finish`, equivalent to the paper's weakened Wikipedia API). Loop: call `get_llm().chat(messages)` → parse action → execute → append `Observation: <result>` to messages → until `Finish` or a step limit. Verify with a multi-hop question (e.g. "compare a magazine's founding year with some year"). Note scripted mode: a scripted LLM that detects ReAct markers returns a scripted trajectory; the parser must be tolerant — either take only the first Action each round, or accept "one reply contains multiple steps"; without a key, label output as placeholder.

2. **Reproduce the three-way "think vs act" contrast**: run the same task with Standard / CoT / Act / ReAct prompts, print trajectories one by one, score by hand or simple rules (whether a key fact was retrieved, whether it hallucinated). Under a real API this can reproduce table 2: CoT invents facts, ReAct looks them up. This demo matches the contrast structure of paper figure 1, evidence that "thinking without acting is not enough".

3. **Action parser and illegal-action handling**: make ReAct output parsing a robust standalone component — mixed multi-line output, unclosed brackets, unknown tool names, repeated actions (a named failure mode is repeatedly generating the same Action, causing a dead loop). Give a loop guard with a timeout (e.g. cap at 8 steps). Show how to let the agent loop "tolerate" noisy output without crashing, which also satisfies scripted-mode usability.

4. **Iterative code-repair loop (RLEF inference side, no training)**: a `solve(problem, feedback_format)` simulator: problem statement + 2–3 public tests, loop "generate code → `exec` tests in a subprocess → format errors back with the appendix C template" until pass. A control: insert random feedback (reproduce the ablation of figures 3/4), compare repair success, seeing that "feedback must be relevant". This demo lands "code execution as a feedback signal" in this repo's `llm_client` + a self-written executor.

5. **A mini policy gradient of "execution-feedback reward" from scratch**: toy environment: repair a function with one bug, action is "emit a candidate code", reward is `1` (all private tests pass) / `-1` (final fail) / `-0.2` (illegal code). Hand-write a REINFORCE update in numpy (reward minus baseline, with KL regularization), iterate on a few candidates, observe how "positive reward for passing execution" changes the model's output distribution. This does not need to train a real LLM; a rule-based code scorer is enough, matching section 2.2 of the paper.

6. **CAI critique-revision pipeline demo**: start from a "helpfulness-only" style initial reply (a harmful reply can be constructed by hand, such as the paper's "hack wifi" example), with `get_llm()` request initial reply → critique against a principle → revise according to the critique, then print the three segments. Extension: repeat revision 3 rounds and compare (under a real API this can reproduce "harmlessness rises with rounds"). Add "AI as judge": for two candidate replies, concatenate a multiple-choice under a "more harmless" principle and let the model score. This demo brings the lecture's high point (AI feedback) into a runnable loop.

(Suggested 1 and 4 as required main-line demos, 2 and 6 optional, 3 and 5 as small exercises for understanding tools/reward.)

---

## Exercise ideas (3)

1. **Complete the ReAct loop**: a half-finished `react_step(client, messages)` missing the three segments "parse Action, execute tool, inject Observation". Fill them in, then `assert parse_action("Action 1: Search[Colorado orogeny]") == ("Search", "Colorado orogeny")`, `assert "Observation" in messages[-1]["content"]`. Hint: from the last non-system message, regex `Action\s*\d*[:：]?\s*(\w+)\[([^\]]*)\]`.

2. **Implement the lookup tool**: write `lookup(text, keyword)` simulating "return the next sentence on the page containing keyword", fill in from a given small paragraph library, `assert lookup("...", "eastern sector") == "The eastern sector extends into the High Plains..."`. Hint: maintain a cursor (next position) in the paragraph list; return the first sentence from the current position that contains keyword.

3. **Compute the RLEF reward**: given several trajectory fragments (all pass / some fail / illegal code), fill in `compute_reward(episode, valid_code)`, `assert`: all pass returns 1, fail returns −1, illegal code returns −0.2; then fill in the full reward with the KL term `r - beta * log(pi/rho)` (hand-calculate from two given probabilities). Hint: distinguish "episode ended" and "mid-turn".

Each exercise ends with the same reminder: "You may ask an AI to explain the idea. Do not ask it to finish the exercise for you."

---

## References

- ReAct: Synergizing Reasoning and Acting in Language Models (arxiv 2210.03629, https://arxiv.org/abs/2210.03629) — main paper of this lecture; Thought/Action/Observation loop and search/lookup/finish tool paradigm. Project page with code: https://react-lm.github.io/
- RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning (arxiv 2410.02089, https://arxiv.org/abs/2410.02089) — train "using execution feedback" into the weights with PPO; reward, public/private test split, feedback template in appendix C.
- Constitutional AI: Harmlessness from AI Feedback (arxiv 2212.08073, https://arxiv.org/abs/2212.08073) — original paper of critique-revision and RLAIF; principle list and few-shot prompts at https://github.com/anthropics/ConstitutionalHarmlessnessPaper
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (arxiv 2201.11903) — ReAct's contrast method; limits of "think without acting".
- Inner Monologue: Embodied Reasoning through Planning with Language Models (arxiv 2207.05608) — ReAct's most direct predecessor; contrast source for the ReAct-IM ablation.
- Self-Consistency Improves Chain of Thought Reasoning (arxiv 2203.11171) — the "voting" part of the ReAct+CoT-SC combination.
- Scaling LLM Test-Time Compute Optimally (arxiv 2408.03314) — L2 test-time compute, echoing RLEF's "iterative vs independent sampling under a fixed budget".
- This repo's `llm_client.py` (`get_llm()`) — unified entry for all LLM demos; scripted mode guarantees offline execution.
