# Lecture 02 — Test-time Compute Scaling (study notes)

> This file is the paper study notes for CS329A lecture 2 (Test-time Compute Scaling), and the source material for `notebooks-en/part1-foundation/02-test-time-compute.ipynb`.
> Source: https://cs329a.stanford.edu/ (Autumn 2025 syllabus)

> **File errata**: `papers/lecture-02/archon.pdf` (labeled 2409.07854 on disk) and `monkey-laws.pdf` (labeled 2410.05468) do not match their titles — the former is actually the algebraic-geometry paper "2-Gorenstein Stable Surfaces", the latter a NeRF/3DGS vision paper "PH-Dropout". The correct arXiv IDs are **Archon = 2409.15254** and **Monkey Laws = 2502.17578** (close-read against those IDs; correct links are in References). If re-downloading PDFs, use the correct arXiv IDs.

## Lecture theme

The core question this lecture answers: **After a model is trained, can we still get more capability by spending more inference compute (test-time compute)? If so, given a fixed inference budget, how do we spend it most efficiently?**

Why it is lecture 2, right after L1 (overview): it is the foundation of everything later about agents. An agent's essence is doing more computation beyond "one call" (search, verification, multi-turn, multi-agent). Those are all variants of test-time compute. This lecture builds a unified view: two knobs, **proposer** and **verifier**, plus "how to allocate a compute budget by problem difficulty". Later L3 verification, L4 tools and code feedback, and L5 planning search all trace back to this frame.

Division of labor among the four papers:

| Paper | Role | Question it answers |
|---|---|---|
| Large Language Monkeys | Phenomenon | How large is the gain from the simplest test-time compute (repeated sampling)? coverage vs precision |
| Snell Test-Time Compute | Method / strategy | Given a budget, how to allocate compute-optimally by difficulty (search + revision) |
| Archon | Engineering / system | How to compose many techniques into an architecture and search for the best combination |
| Monkey Laws | Theory | Why repeated sampling shows power-law scaling, and where the power law comes from |

## Close reading

### Paper 1: Large Language Monkeys (arxiv:2407.21787, `large-language-monkeys.pdf`)

- **Core idea**: treat "inference compute" as a scaling axis parallel to training compute. The method is as simple as it gets: for each problem, temperature-sample k independent candidate solutions, then pick one with a verifier (i.e. pass@k). Gain is governed by two quantities:
  1. **Coverage**: the fraction of problems for which at least one of the k samples is correct — "can the model solve this problem at all".
  2. **Precision**: whether we can pick the correct one from a pile of candidates — "can the verifier find the needle in the haystack".
  In domains with an automatic verifier (unit tests, proof checkers), coverage equals final success rate. In domains without a verifier (math word problems), coverage is only an upper bound; actual success depends on the verification method.

- **Key formulas / algorithms**:
  - **Unbiased pass@k estimator** (from Codex / Chen et al.): generate N samples per problem, C of them correct, then
    $$\mathrm{pass@k} = \frac{1}{P}\sum_{i=1}^{P}\left[1 - \frac{\binom{N-C_i}{k}}{\binom{N}{k}}\right]$$
    Use Chen et al.'s numerically stable implementation (`1 - prod(1 - k/arange(N-C+1, N+1))`). Key: this is not the naive "fraction of problems that have a correct sample"; it is unbiased.
  - **Coverage power law (inference-time scaling law)**: $\log(c) \approx a\, k^{b}$, i.e. $c \approx \exp(a\, k^b)$. Fit examples: Llama-3-8B-Instruct on MATH $(a=-1.33, b=-0.43)$; 70B-Instruct MATH $(a=-0.75, b=-0.46)$; 8B-Instruct CodeContests $(a=-3.88, b=-0.11)$; Gemma-2B CodeContests $(a=-8.54, b=-0.14)$; Pythia-70M MATH $(a=-7.59, b=-0.18)$. Fit method: 40 log-spaced points, SciPy `curve_fit`.
  - **Inference FLOPs estimate** (for cost comparison): $\mathrm{FLOPsPerToken} \approx 2 \times (\text{parameter count} + 2 \times \text{Layers} \times \text{TokenDim} \times \text{ContextLen})$.
  - Sampling hyperparameters: MATH/GSM8K temperature 0.6, 10000 samples per problem; MiniF2F temperature 0.5; CodeContests temperature 0.6 + top-p 0.95; SWE-bench temperature 1.6, 250 attempts.

- **Key experimental results (concrete numbers)**:
  - **SWE-bench Lite** (real GitHub issues, automatic tests): DeepSeek-Coder-V2-Instruct + Moatless Tools, single try 15.9% → 250 samples **56%**, 13 points above single-try SOTA (43%, GPT-4o+Claude 3.5 Sonnet mix).
  - **CodeContests**: Gemma-2B pass@1 only 0.02% → pass@10k 7.1% (**300×**). The full Pythia family has 0 coverage on this set (training data lacks code).
  - **MATH**: Llama-3-8B-Instruct coverage from 82.9% at 100 samples → 98.44% at 10000; Pythia-160M 0.27% → 57%.
  - **Precision problem (the most important counter-intuitive finding)**: on MATH, coverage rises to 95%+, but majority vote only goes from 40.50% to 41.41%; reward model + best-of-N likewise **plateaus** around ~100 samples. Conclusion: without a reliable verifier, repeated-sampling gains stall — "the gap between coverage and success rate grows with sample count".
  - **Cost comparison (Table 1)**: on SWE-bench, DeepSeek-Coder-V2-Instruct sampled 5 times, 29.62% solve rate, total cost $10.8 (1x); GPT-4o once 24%, $39 (3.6x); Claude 3.5 Sonnet once 26.7%, $51 (4.7x). A weaker model + more samples can be cheaper and better than a stronger model once.
  - **Verifier flaws (two warnings)**: SWE-bench Lite has flaky tests on 11.3% of problems (34 problems; 30 of them sometimes fail even the gold solution); of 122 Python3 CodeContests problems, 35 have false negatives where the gold solution also fails the tests.
  - **CoT faithfulness (Table 2)**: human annotation of 105 correct samples' chains of thought, >90% are faithful reasoning (so the verifier has a usable signal; the problem is that the verifier is not good enough).
  - Incidental finding: GSM8K problem 1042 has a wrong ground truth (the answer is 3.5, not 3).

- **Relation to the lecture theme**: this paper defines the two basic quantities of test-time compute scaling, coverage/precision, gives the simplest scaling method and power-law evidence, and points to the core bottleneck "without a verifier we stall" — handing the problem to later lectures (L3 verifiers, L5 search).

- **Demo-ready code points**:
  - Implement the numerically stable unbiased pass@k estimator from scratch; verify by hand on a small example.
  - Simulate pass@k curves from synthetic data (given pass@1 per problem), fit $\exp(ak^b)$, plot log-log.
  - Compare gain curves of "majority vote / best-of-N (oracle)", reproducing plateau vs continued rise.

### Paper 2: Scaling LLM Test-Time Compute Optimally (arxiv:2408.03314, `snell-test-time-compute.pdf`)

- **Core idea**: given a fixed inference budget, **different test-time methods have different efficiency on problems of different difficulty**. Easy problems suit sequential revision (local correction); hard problems suit parallel resampling / tree search over a verifier (global exploration). The optimal policy therefore allocates adaptively by difficulty — **compute-optimal scaling**, which can raise efficiency 2–4× at the same budget.

- **Key formulas / algorithms**:
  - **Unified view (proposer-verifier)**: test-time compute has only two knobs — change the **proposal distribution** (change input tokens, e.g. finetune a self-revising model) or **post-process outputs** (score/select candidates with a verifier, e.g. best-of-N, search over a PRM).
  - **Difficulty definition**: for each problem, sample 2048 times from the base LLM to estimate pass@1, then split into 5 difficulty bins by quintiles (more effective than MATH's hand-labeled difficulty). Two versions: oracle (grade with ground truth) and model-predicted (mean PRM score); the two curves nearly coincide.
  - **PRM training**: no human labels; Monte-Carlo rollout gives each step a soft label (reward-to-go); train with binary cross-entropy; within-step aggregation uses **the last-step score** (better than min/prod; the PRM then approximates an ORM but still beats ORM substantially).
  - **Search methods**: best-of-N weighted (marginalize scores of the same final answer) / beam search (beam width $M$, expand top-$N/M$ each step) / lookahead search (each beam step rollouts $k$ steps at temperature 0, score with the PRM at the end of the rollout; $k=0$ is beam). Lookahead cost is recorded as $N\times(k+1)$.
  - **Sequential vs parallel (revisions)**: finetune a revision model (each correct answer paired with at most 4 related wrong answers by edit distance, then SFT); at inference generate a chain of revisions sequentially; allocate budget by sequential/parallel ratio $\in\{2^{-7},\dots,2^7\}$.
  - **FLOPs exchange rate** (vs training compute): $X = 6ND_{\text{train}}$, $Y = 2ND_{\text{infer}}$; scaling parameters by $M$ makes total FLOPs $M(X+Y)$. Matching that budget with a small model + test-time requires scaling inference compute by $M + 3(D_{\text{train}}/D_{\text{infer}})(M-1)$, so the conclusion depends on $R = D_{\text{infer}}/D_{\text{train}}$.

- **Key experimental results (concrete numbers)**:
  - **Compute-optimal uses about 4× less compute than best-of-N** to reach the same accuracy: 16 vs 64 generations in search; 64 vs 256 in revisions.
  - Search: at low budget, beam search clearly beats best-of-N, but the advantage disappears or reverses as budget grows; lookahead is generally worse at the same budget (over-optimizing the verifier yields 1–2-step "cheat" solutions). After binning by difficulty: easy problems (bins 1–2) see beam search **drop** as budget grows (over-optimization / verifier signal amplified); medium-hard (bins 3–4) beam stably beats best-of-N; hardest (bin 5) no method makes large progress.
  - Revisions: sequential sampling is slightly better than parallel under both verifier and majority selection; about **38% of correct answers are revised into wrong ones** on the next round, so we must pick from the whole chain with a verifier/majority; easy problems are best fully sequential; hard problems have an optimal sequential/parallel ratio (Fig.7).
  - **FLOPs-matched comparison with a 14× larger model**: on easy/medium problems (bins 1–3, sometimes 4) or low inference load ($R\ll1$, self-improvement setting), test-time compute is more efficient than training more parameters; on hard problems (bins 4/5) or high inference load ($R\gg1$, large-scale deployment), scaling training is more efficient. Conclusion: **test-time and pretraining compute are not 1-to-1 exchangeable**.
  - The revision model's pass@1 rises with revision steps (it learns to correct from errors); finetuning details: AdamW lr 1e-5; ReSTEM-style online revision training actually got worse.

- **Relation to the lecture theme**: on top of LLM Monkeys' "more samples = better", this adds **optimal allocation under a budget**. It is the source of the agent-design rule "compute-optimal": pick different strategies and budgets for inputs of different difficulty.

- **Demo-ready code points**:
  - Simulate best-of-N vs different sequential/parallel ratios on synthetic data; plot families of "budget → accuracy" curves.
  - Difficulty bins: sort problems by pass@1 into 5 bins; show that the best strategy differs by bin.
  - Implement a simplified beam search (PRM as a scripted scorer); contrast with best-of-N.
  - FLOPs-matched visualization: plot a 14× larger-model "star" at three x-axis positions R=0.16/0.79/22.

### Paper 3: Archon: An Architecture Search Framework for Inference-Time Techniques (arxiv:2409.15254, `archon.pdf`, note the correct ID)

- **Core idea**: a single test-time technique (repeated sampling, revision, verification, fusion, …) has its own applicable setting; **no architecture is best on every task**. Archon organizes many techniques into a "layered LLM system" (analogy: neural nets — parallel within a layer, sequential across layers), then **Bayesian-optimizes** the combination, turning "build a system" into "tune hyperparameters".

- **Key formulas / algorithms**:
  - **LLM components (all text-to-text operations, no learnable weights)**:
    - Generator: takes an instruction, emits candidates; supports multi-model parallelism (ensembling) and repeated sampling (temperature 0.7).
    - Fuser: merges several candidates into a higher-quality output (average +8.9%).
    - Ranker: pairwise comparison ranking, take top-K (instruction-following +10.8%, only 2.7% from oracle).
    - Critic: lists pros/cons of each candidate for the ranker/fuser (+11.5 points).
    - Verifier: two stages (first a "why correct" reasoning, then [Correct]/[Incorrect]); only verified candidates pass (reasoning tasks +8.4%).
    - Unit Test Generator + Evaluator: the LLM generates 5–10 test statements; the Evaluator scores by pass count (CodeContests Pass@1 17.9%→29.3%, **+56%**).
  - **Structure rules**: Generator only in the first layer; each layer holds one component type; Critic must precede Ranker/Fuser; Unit Test generator precedes evaluator; each layer's I/O is always a list of strings; the last layer takes the first string.
  - **Search space (6 axes)**: top-K generators (1–10), samples per generator (1–5, up to 1000 on CodeContests), fusion layers (1–4, last layer always a single Fuser), Fusers per layer (2–10 step 2), whether to add Critic/Ranker before each layer, whether to add Verifier/Unit-Test before the last layer. After dropping invalid configs, **9576** remain.
  - **Search method**: 20% of data as a dev set; randomly sample a batch of architectures to calibrate a Gaussian-process surrogate, then a "suggest → evaluate → refine" loop until the budget is spent; over-budget architectures are dropped from the search space (for compute-matched comparison). Can optimize accuracy/latency/cost arbitrarily.

- **Key experimental results (concrete numbers)**:
  - Best architecture **beats frontier models (o1, GPT-4o, Claude 3.5 Sonnet) by 15.1% on average**; with open-source models only, still **11.2%** above open-source SOTA on average; 8.4–8.6% above MoA/ADAS/AFlow on average.
  - Efficiency: **20.0% fewer inference calls, 15.1% fewer input tokens, 13.5% fewer output tokens** than alternative frameworks; 12.4% higher at the same token budget. Best targeted architecture beats o1 by 8.1% and o1-mini by 9.7% on average.
  - **Bayesian optimization effectiveness**: 96.0% of searches find the best architecture; 88.5% fewer evaluations than greedy, 90.4% fewer than random. Below 20 calls the BO advantage disappears and greedy is comparable. Returns diminish after about 230–240 calibration samples.
  - **Task differences (the motivation)**: instruction following / reasoning → multi-generator diversity + multi-layer fusion depth; reasoning subtasks each need their own architecture (MixEval +3.7, MixEval-Hard +8.9 points); code → unit tests + high sample count (5 tests + large sampling raises GPT-4o Pass@1 from 40/140 to 58/140, +44.3%).
  - **Generalization**: a general-purpose architecture keeps **91.7–94.2%** of task-specific performance on unseen GPQA/MMLU/MMLU Pro (ADAS/AFlow only 66–74%).
  - 7B open-source models can raise performance 7.5%, but they suit ranker, not critic/fusion; full 70B+ works best. Architecture latency and cost ~5× a single call.
  - Best general-purpose all-source architecture: start with 10 strongest generators → 4 layers of critique+fusion (Fuser counts 8→6→4, funnel) → single Fuser output.

- **Relation to the lecture theme**: unifies all the earlier scattered techniques (sampling, revision, verification, voting) **into a searchable system space**, and replaces hand-built systems with automatic search. It is the engineering endpoint of "test-time compute combinations", and a template for "automatically designing agent architectures".

- **Demo-ready code points**:
  - With `llm_client` (scripted mode), hand-build a small pipeline Generator(3)→Critic→Ranker(top-2)→Fuser and contrast with a single call.
  - On a simplified search space (e.g. only top-K and fusion depth), implement "random search vs grid search" to show Archon's search idea.
  - A "unit-test filter": a scripted LLM generates test statements + scores; observe how pass rate affects the final answer.

### Paper 4: How Do Large Language Monkeys Get Their Power (Laws)? (arxiv:2502.17578, `monkey-laws.pdf`, note the correct ID)

- **Core idea**: answers "why repeated sampling shows a power law". Apparent contradiction: for a **single problem**, failure rate should fall **exponentially** with attempts (i.i.d.); but **aggregated** over a whole benchmark, failure rate falls as a **power law**. The answer: the distribution of single-try success $p_i@1$ across problems is **heavy-tailed** — a few "almost unsolvable" problems drag the aggregate curve into a power law. The "power" of the power law comes from the heavy-tailed task-difficulty distribution, not from each problem's own scaling.

- **Key formulas / algorithms**:
  - **Per-problem exponential decay**: $p_i@k = 1 - (1 - p_i@1)^k$; for large $k$, $-\log(p_i@k) \approx (1-p_i@1)^k$ (exponential decay, confirmed problem-by-problem on Pythia on MATH and HarmBench jailbreak data).
  - **Aggregate definition** (compound binomial): $\mathrm{pass}_D@k = 1 - \int_0^1 (1-p)^k\, p_D(p)\, dp$, where $p_D(p)$ is the density of $p_i@1$ across problems.
  - **Theorem 3.1 (sufficiency)**: if $p_D(p) = C\, p^{\,b-1} + O(p^{\,b-1+\theta})$ as $p\to0^+$, then $-\log(\mathrm{pass}_D@k) \sim C\,\Gamma(b)\,k^{-b}$.
  - **Theorem 3.2 (necessity)**: if $-\log(\mathrm{pass}_D@k) \sim A\,k^{-b}$, then under mild regularity $p_D(p) \sim \frac{A}{\Gamma(b)}\,p^{\,b-1}$. I.e. "aggregate is a power law" $\iff$ "the left tail of the $p_i@1$ distribution is a power law".
  - **Distribution family**: fit $p_i@1$ with a **three-parameter Kumaraswamy / Beta-Binomial with a scale** ($p \in (0,c)$, $c\approx 0.01\sim0.1$, because most single-try rates are far below 1). MLE is done at $1/N$ discretization, including the count of problems that fall in the $(0,1/N)$ left-tail bucket.
  - **New estimator**: first fit $\hat p_D(p_i@1)$, then simulate $\mathrm{pass}_D@k$ from the formula above and extrapolate to arbitrary $k$; regress the power-law exponent in log-log space.

- **Key experimental results (concrete numbers)**:
  - Data: LLM Monkeys' Pythia 70M–12B on MATH 128 problems, and Best-of-N jailbreaking of frontier models (Claude/GPT-4o/Gemini/Llama 3 8B IT) on HarmBench 159 prompts.
  - Per-problem confirmation of exponential decay; aggregate confirmation of a power law (LLM Monkeys' $\approx a k^{-b}$ and Hughes et al.'s $\approx a k^{-b}$, eqs. (2)/(4) in the paper).
  - **Counterexample**: Llama 3 8B IT under jailbreaking has **no** aggregate power law — because every prompt can be jailbroken within budget, the $p_i@1$ distribution has no heavy left tail (Fig.4 shows a non-smooth distribution).
  - Simple families' scaling exponents: Uniform → $k^{-1}$; Beta(α,β) → $k^{-\alpha}$; Kumaraswamy(α,β) → $k^{-\alpha}$; continuous Bernoulli(λ<1/2) → $k^{-1}$; Reciprocal(α,β) → $(1-\alpha)^k/k$ (**not a power law**).
  - **New estimator**: about **one order of magnitude smaller relative error** than least-squares ($|\hat b - b|/b$), equivalent to **2–4 orders of magnitude** less inference compute; robust under distribution mismatch as well.

- **Relation to the lecture theme**: supplies a rigorous theory for paper 1's empirical power law, and connects the coverage scaling exponent $b$ to the measurable object "problem-difficulty distribution". It makes "can repeated sampling keep rising" predictable (first check whether $p_i@1$ has a heavy left tail).

- **Demo-ready code points**:
  - With a synthetic $p_i@1$ distribution (e.g. Kumaraswamy), plot both: a single-problem exponential decay curve + an aggregate power-law curve, so the "exponential vs power law" conflict and unification are visible (reproduce the idea of Fig.2).
  - From the $p_i@1$ distribution, **simulate** $\mathrm{pass}_D@k$; compare the exponent $b$ from the distribution estimator vs least squares.
  - Check whether a distribution "without a heavy tail" (e.g. uniform, away from 0) loses the power law (Llama 3 8B IT counterexample).

## Teaching thread (how a Stanford instructor might teach this)

The instructor organizes this class as one decision chain: "given an inference budget → how to spend it":

1. **Build intuition first: we can still spend money after training**. Humans "think longer on hard problems"; LLMs currently "try each problem once". Pose a seemingly naive question: what if we try several times? → introduce the simplest test-time compute = **repeated sampling**.

2. **Bring in Large Language Monkeys; establish two quantities: coverage and precision**. Large empirical numbers (SWE-bench 15.9%→56% beating single-try SOTA; Gemma-2B pass@1 0.02%→pass@10k 7.1%) show that sample count buys coverage, as a power law $\exp(ak^b)$. Stress two things: (a) details of the unbiased pass@k estimator; (b) **the verifier decides success** — on MATH, coverage reaches 98% while majority vote is only 41%, and the gap grows. This is the first place readers stall: why "more samples" and "finally correct" are not the same thing.

3. **Change the question: not how many samples, but how to allocate the budget → Snell's compute-optimal**. Give the proposer-verifier unified view (two knobs). Note that "the same method has different efficiency on different difficulties" (beam is good on hard problems, over-optimizes easy ones), so bin by pass@1 and pick the best strategy per bin. Key numbers: 4× less compute for the same accuracy; in the FLOPs-matched comparison with a 14× larger model, easy/medium problems favor test-time, hard problems favor pretraining. Explain the FLOPs-exchange pitfall clearly ($R = D_{infer}/D_{train}$ decides the conclusion).

4. **Theory look-back: why a power law → Monkey Laws**. State the contradiction: a single problem should decay exponentially ($1-(1-p_i@1)^k$), the aggregate is a power law. Answer: the heavy-tailed $p_i@1$ distribution. Use theorems 3.1/3.2 to make "left-tail power law $\iff$ aggregate power law" clear. Use a counterexample (Llama 3 8B IT has no heavy tail → no power law) so readers really understand. Side product for engineering: the distribution estimator can save 2–4 orders of magnitude of compute when predicting the scaling exponent.

5. **Engineering endpoint: compose techniques into a system and search automatically → Archon**. Since point techniques each have strengths and "the best architecture varies by task", build a searchable system space (Generator/Fuser/Ranker/Critic/Verifier/Unit-Test) and Bayesian-optimize among 9576 configs. Key numbers: 15.1% above frontier on average, 20% fewer calls. Echo the lecture opening: "spend more compute" is not mindless extra sampling, but **smart assembly + allocate by problem**.

Three places readers tend to get stuck:
- **Unbiased pass@k**: why we cannot simply compute "fraction of problems with a correct sample" (overestimates, especially when k is near N). Need a hand-calculation example.
- **Exponential vs power law**: a "straight line" on a log-log plot is a power law; a single problem is a downward-concave exponential on log-log — superposing them yields the aggregate power law.
- **best-of-N vs majority vote**: the former picks one by verifier score (needs precision); the latter uses the mode (immune to rare correct solutions); so majority vote necessarily plateaus.

## Code demo ideas (3–6)

1. **Unbiased pass@k estimator (from scratch + hand calculation)**: implement `estimate_pass_at_k(num_samples, num_correct, k)` (numerically stable `1 - prod(1 - k/arange(N-C+1, N+1))`). On a small example (N=10, C=3) compute pass@1/pass@5/pass@10, contrast with the naive "fraction with a correct sample", plot the error. **Key observation**: unbiased vs naive differ clearly when k is near N.

2. **Coverage power-law fit (synthetic data, reproducible offline)**: assign each problem a $p_i@1$ (sampled from Kumaraswamy(0.3,1)), simulate k samples to compute pass@k, fit $\exp(ak^b)$ with `curve_fit` on 40 log-spaced points, overlay true and fitted curves on log-log. **Key observation**: fit error is small; reproduces LLM Monkeys Fig.5 (negative a, small-magnitude negative b).

3. **"Single-problem exponential vs aggregate power law" conflict (substance of Monkey Laws Fig.2)**: from the same $p_i@1$ batch, plot (a) $-\log(p_i@k)$ for one hard problem decaying exponentially with k; (b) aggregate $-\log(\mathrm{pass}_D@k)$ as a straight line (power law). Then swap in a "no heavy tail" distribution (e.g. $p_i@1$ concentrated in 0.2–0.4) and watch the aggregate leave the power law for faster decay. **Key observation**: the heavy tail decides the power law; corresponds to the Llama 3 8B IT counterexample.

4. **The verifier decides the gain: three curves oracle / majority / best-of-N**: synthetic $p_i@1$ distribution; simulate three selectors for k=1..1000. Oracle (always picks the correct one) keeps rising toward 1; majority vote and a "weak verifier" plateau around ~100. **Key observation**: reproduces the widening gap in LLM Monkeys Fig.7.

5. **Visualization of Snell's compute-optimal allocation**: split budget k into "parallel N_par + sequential N_seq" (ratio from all-parallel to all-sequential); heat-map the optimal ratio per difficulty bin on synthetic data (x=difficulty bin, y=ratio, color=success rate). **Key observation**: easy problems prefer all-sequential, hard problems a balance — reproduces the right panel of Snell Fig.7.

6. **Simplified FLOPs-matched small vs large model**: with `FLOPs ≈ 2N·tokens`, plot "small model + test-time compute" vs "14× larger model greedy"; place the "large-model star" at three values of $R = D_{infer}/D_{train}$ (0.16/0.79/22) and see whether the star sits above the line (test-time better) or below. **Key observation**: small R (self-improvement) favors test-time; large R (deployment) favors pretraining.

> Implementation note: 1–5 can all run fully offline on numpy synthetic data (scripted-mode compatible, per CLAUDE.md). For a real LLM, `llm_client.get_llm()` can do true sampling on a small GSM8K subset (real mode), and mark scripted output as placeholder. Demos 4 and 5 do not need a real LLM to show paper-consistent trends.

## Exercise ideas (3)

1. **Implement unbiased pass@k**: fill in `estimate_pass_at_k(N, C, k)`; verify pass@5 and pass@20 with (N=20, C=5); after `assert` passes, `print` that "the gap between unbiased and naive estimates grows with k". Hint: `np.prod(1 - k/np.arange(N-C+1, N+1))` holds only when C<k; when C>=k return 1.0 directly.

2. **From a heavy-tailed distribution to a power-law exponent**: given synthetic $p_i@1$ (Kumaraswamy(α=0.4, β=1)), simulate $\mathrm{pass}_D@k$ and fit $b$ on log-log; then use theorem 3.1 with left-tail density power $b-1$ as the theoretical expectation; `assert abs(fit_b - theoretical_b) < 0.1`. Hint: Kumaraswamy density near 0 is $f(p)\propto p^{\alpha-1}$, so the theoretical exponent is $b=\alpha$.

3. **Majority-vote plateau**: on synthetic data implement oracle / majority selectors, scan k to 10000; `assert` majority's gain after k=100 is < 1 percentage point while oracle's is > 10; then swap to a "no heavy tail" distribution and verify that majority's plateau weakens. Hint: majority is sensitive only to the majority answer; a rare correct solution does not change the mode.

## References

- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787) (Brown et al., 2024) — empirical coverage of repeated sampling and the power law; coverage/precision axes. Code https://github.com/ScalingIntelligence/large_language_monkeys
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) (Snell et al., 2024) — compute-optimal scaling: allocate test-time compute by difficulty, revision + PRM search.
- [Archon: An Architecture Search Framework for Inference-Time Techniques](https://arxiv.org/abs/2409.15254) (Saad-Falcon et al., ICML 2025) — layered LLM systems + Bayesian architecture search. Correct ID is 2409.15254 (PDF in the repo is mislabeled). Code https://github.com/ScalingIntelligence/Archon
- [How Do Large Language Monkeys Get Their Power (Laws)?](https://arxiv.org/abs/2502.17578) (Schaeffer et al., ICML 2025 oral) — theory: per-problem exponential decay + heavy-tailed pass@1 ⟹ aggregate power law. Correct ID is 2502.17578 (PDF in the repo is mislabeled).
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (Chen et al., 2021) — source of the unbiased pass@k estimator.
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) (Wang et al., 2023) — majority vote / self-consistency; directly related to LLM Monkeys' precision discussion.
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (Lightman et al., 2023) — source of PRM training and MATH difficulty bins; Snell reuses the 12k/500 split.
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021) — GSM8K and the earliest verifier training.
- [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814) (Li et al., 2022) — pioneer of million-scale repeated sampling; source of CodeContests.
- [Beyond Chinchilla-Optimal: Accounting for Inference in LM Scaling Laws](https://arxiv.org/abs/2401.00448) (Sardana & Frankle, 2023) — scaling laws that count inference FLOPs; basis of Snell's FLOPs formula.
