# Lecture 14 — Agent evaluation and long-horizon tasks (study notes)

> This file is the paper study notes for CS329A lecture 14, and the source material for the matching notebook.
> Source: https://cs329a.stanford.edu/ (Autumn 2025 syllabus)

## Lecture theme

The question this lecture answers: **what counts as a reliable evaluation of an Agent?** Earlier lectures taught how to turn an LLM into an Agent (loops, tools, planning, memory, SWE agents). This lecture turns to "metrology": Agents do multi-step, open, variable-duration tasks; a traditional benchmark's "multiple-choice score" both saturates quickly and cannot convert into value for humans. The core tension: the more realistic the task, the harder it is to score automatically; the cheaper the scoring (LLM judge), the less reliable.

The lecture proceeds in three nested threads:
1. **From ability to task duration** (METR): use "the time a human needs to complete the task" as a ruler for Agent ability, getting a continuous metric (time horizon) comparable across models and years.
2. **From task duration to economic value** (GDPval): duration is not enough; we also look at "how much the work is worth" — real workplace tasks, blind-review win rate against industry experts.
3. **Reliability and holes of automatic evaluation** (DeepScholar-Bench): large-scale evaluation must score automatically (LLM judge), but LLM judges have position bias, self-preference, and other holes, and need calibration against human scores.

Together the three answer "why evaluating Agents is hard": the task distribution is no longer fixed, the success criterion is not unique, and the reliability of the automatic scorer is itself an object under test.

## Close reading

### Paper 1: Measuring AI Ability to Complete Long Software Tasks (arxiv:2503.14499, measuring-long-tasks.pdf)

- **Core idea**: existing benchmarks can only give "a score on some fixed problem set", and cannot answer "how complex a piece of real work this model can actually do". METR proposes translating AI ability into human time units: **X% task-completion time horizon** — the t such that "AI can complete with X% success rate the tasks that a human expert usually needs t hours for". The measurement method: construct a suite of 170 software/ML tasks covering an extremely wide difficulty range (seconds to 8 hours) → have human experts and AI Agents each do them → record human time and AI success rate → fit each model's time horizon corresponding to 50% success → then plot a long curve by model release date. Core finding: **frontier models' 50% time horizon has roughly doubled every 7 months since 2019** (GPT-2 only 2 seconds, o3 about 110 minutes), and after 2024 it may still be accelerating. Extrapolating this trend, AI may reach a 50% time horizon of "one-month-duration tasks" between mid-2028 and mid-2030.

- **Key formulas / algorithms**:
  - A logistic model of success probability (idea from the 2PL of psychometrics Item Response Theory, but difficulty is the human baseline duration rather than a learned parameter):
    $$p_{\text{success}}(\text{agent},\text{task}) = \sigma\!\big((\log h_{\text{agent}} - \log t_{\text{task}})\cdot \beta_{\text{agent}}\big)$$
    where $t_{\text{task}}$ is the geometric mean of successful human-baseline durations, $h_{\text{agent}}$ is the 50% time horizon, $\beta_{\text{agent}}$ is the slope (one parameter per agent). Logistic regression is done separately for each model; tasks are weighted $1/\sqrt{n}$ by family (tasks within a family are strongly correlated, so they are down-weighted to keep diversity).
  - Trend line: OLS of $\log(\text{horizon})$ on release date, slope converted to doubling days; error estimated by a three-level hierarchical bootstrap (family→task→run, 10000 times).
  - Task suite: HCAST 97 tasks (1 minute to 30 hours, 46 families) + RE-Bench 7 eight-hour ML research-engineering tasks + newly added SW AA (Software Atomic Actions) 66 one-to-thirty-second single-step small tasks (file selection, alert triage, request routing, code completion, mental arithmetic), used to measure older models such as GPT-2/GPT-3.

- **Key experimental results**:
  - 12 frontier + 4 near-frontier models (2019–2025), about 8 runs per agent/task combination. 50% time-horizon doubling time = 207 days (95% CI 166–240 days, about ±19%); 80% time horizon is about 4–6× shorter, doubling time similar (204 days) — so the gap between a model's "occasionally succeeding at long tasks" and "steadily handling medium-long tasks" is still large.
  - o3 is above the long-term trend line (p=0.006), suggesting faster growth after 2024; extrapolating 2024–2025 data alone, a one-month horizon could arrive as early as early 2027.
  - External-validity check: the same exponential trend holds on SWE-bench Verified (doubling time about 70 days, steeper than the main suite's predicted 143 days); an internal real-PR experiment finds model performance consistent with "low-context contractor" time, inconsistent with "repo maintainer" time (maintainers are 5–18× faster than contractors).
  - Failure-mode attribution: 12/31 of GPT-4's failures are "repeating a failed action", o1 only 2/32; half of o1's failures are "giving up the task too early" — ability gains mainly come from better tool use, better recovery from errors, stronger logic.
  - "Messiness": score tasks on 16 "real-world dirtiness" factors (dynamic environment, irreversible mistakes, resource limits, needing to actively seek information, etc.), mean 3.2/16; after controlling for duration, each +1 messiness drops success rate by about 8.1%. But time trends of high/low messiness subsets are close — dirty tasks have lower absolute scores, but the progress rate has not slowed.
  - Cost: compared at $143.61/hour (Google L4 engineer average hourly wage), more than 80% of successful runs cost less than 10% of a human doing the same task.

- **Relation to the lecture theme**: this paper is the lecture's "ruler". It gives a methodology for converting any benchmark result into "human time" (human baseline + logistic fit), so GPT-2 and o3 can be compared on the same curve, and it is the motivation source for later GDPval (converting time × wage into economic value) and DeepScholar (automatic scoring of long tasks).

- **Demo-ready code points**:
  - Implement time horizon from scratch: make a synthetic "task duration vs success rate" dataset, hand-write logistic regression in numpy (or use scipy.optimize) to fit $h_{\text{agent}}$ and $\beta_{\text{agent}}$, verify that when $h_{\text{agent}}=t_{\text{task}}$ the success rate is exactly 0.5.
  - Plot a log "time horizon vs release year" figure: OLS on $\log h$, output doubling days, reproducing "doubles every 7 months".
  - Simulate hierarchical-bootstrap error bars (three-level resampling of family/task/run).
  - Demo the gap between 50% vs 80% horizon (fits under different success-rate thresholds).

### Paper 2: GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks (arxiv:2510.04374, gdpval.pdf)

- **Core idea**: METR scales ability by "duration", but duration is not economic value — a sentence "7 hours of work" does not say whether the work is worth money or whether it is well done. GDPval (OpenAI) directly measures **real economically valuable tasks**: from the 9 US industries that contribute most to GDP, pick 44 high-pay, primarily digital occupations, have industry experts with 14 years of experience on average turn their real work products into tasks (each task = one request + one deliverable). The main metric is **blind-review win rate**: experts in that occupation pairwise-compare "model output vs human-expert output" and judge who is better / a tie. Core conclusion: **frontier model quality already approaches industry experts** (the best model Claude Opus 4.1 has 47.6% of outputs "better than or tying humans"), and OpenAI model performance rises **approximately linearly** over time; but once expert review / rework time is counted, the pure speed advantage shrinks sharply. They open-source 220 gold tasks and provide an experimental automatic scoring service (evals.openai.com).

- **Key formulas / algorithms**:
  - Win rate: blind pairwise comparison, scores in {0, 0.5, 1} (model wins / ties / loses).
  - Automatic-scoring agreement (also the formula for "evaluating the evaluator"):
    - Human–auto agreement $A^{\text{HA}}_s = E[\,1 - |H - A|\,]$; human–human agreement $A^{\text{HH}}_s = E[\,1 - |H_1 - H_2|\,]$ (mean of two human scores on the same sample).
  - Economic-value conversion: task dollar value = mean completion duration × that occupation's median hourly wage (OEWS data).
  - Speed / cost benefit model ("try the model first, fix it yourself if unsatisfied"): let $w_i$ be the model's win rate on task $i$, $M_T/M_C$ model time/cost, $R_T/R_C$ expert-review time/cost, $H_T/H_C$ human-completion time/cost:
    - try-1 times: $E[T_{1,i}] = M_{T,i} + R_{T,i} + (1-w_i)\,H_{T,i}$
    - try-n times: $E[T_{n,i}] = (M_{T,i}+R_{T,i})\,\frac{1-(1-w_i)^n}{w_i} + (1-w_i)^n\,H_{T,i}$; as $n\to\infty$, human time is replaced by $(M_T+R_T)/w$.
  - Task-construction pipeline: from 9 major industries (each >5% of GDP, Q2 2024) pick the 5 highest-pay digital occupations per industry (use GPT-4o to label O*NET tasks "whether digital", threshold 60%); experts average 14 years of experience, need interview + background check + training + a test, acceptance rate <10%; each task averages 5 rounds of human review.

- **Key experimental results**:
  - Scale: full set 1,320 tasks (30 per occupation), gold subset 220 (5 per occupation, 69% of tasks need operating on reference files, at most 17 reference files). Gold tasks average 9.49 hours (median 5 hours), average dollar value $398; tasks are almost all multimodal deliverables (CAD, PDF, PPTX, spreadsheets, audio/video).
  - Head win rate: gpt-4o 12.5% → o4-mini 29.1% → o3 35.2% → gpt-5 39.0%, Claude Opus 4.1 best about 47.6% (win+tie) — quality approaches but does not exceed humans.
  - Speed / cost: naive ratios as high as 90–327×; after counting "review + failed rework" (try-1), gpt-4o is only 0.87× (slower), gpt-5 about 1.12×; after try-n, gpt-5 about 1.39×. **Conclusion: the report's 327x naive speedup is overestimated; real gains mainly come from "human in the loop".**
  - Automatic scorer (GPT-5-high) agrees with humans 66%, humans with each other 71% — only 5 percentage points apart; but agreement is lower on strong models' outputs (the model prefers its own outputs). 12/220 tasks are marked "not automatically scorable" (need the network, need non-Python runtime, font rendering, speech transcription, etc.).
  - Prompt engineering + scaffolding (best-of-4 + GPT-5 judge, encouraging multimodal self-check): win rate +5pp, PPT format errors 86%→64%, multimodal self-check rate 15%→97%; higher reasoning effort is better.
  - Weak-context version (prompt shortened to 42% length): models clearly worsen — real work is "figuring out what to do"; missing context is the hardest part.

- **Relation to the lecture theme**: upgrades METR's "duration ruler" to a dual ruler of "economic value + quality", and demonstrates the boundary of automatic evaluation: true-expert blind review is expensive (a single comparison >1 hour) but reliable; automatic scoring is cheap but only reaches 66% agreement. It is also the target of the "LLM judge reliability" discussion (the third paper deepens it).

- **Demo-ready code points**:
  - Implement win-rate statistics and agreement from scratch: given synthetic scoring data, compute win rate, $A^{\text{HA}}$, $A^{\text{HH}}$, compare "automatic scorer" and "humans".
  - Implement the try-n-then-fix expected-cost formula and plot the curve: see at what win rate model assistance becomes worthwhile (crossover analysis).
  - Demo a blind-review scoring flow: use a scripted LLM to generate two deliverable descriptions, let a "human judge" (script or scripted) pairwise-score, understanding the 0/0.5/1 three-valued scores.

### Paper 3: DeepScholar-Bench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis (arxiv:2508.20033, deepscholar-bench.pdf)

- **Core idea**: a new benchmark for evaluating long-horizon Agents of the "deep research / generative research synthesis" class. Real research-synthesis tasks are "open-ended, correctness has no unique standard, depend on live web retrieval", while existing QA benchmarks only measure short answers, and expert hand-annotated sets go stale and get contaminated. DeepScholar-Bench uses a **continuously updated data pipeline** to auto-generate queries from recent high-quality arXiv papers (task: given paper title+abstract, generate a related-work section; the human authors' original text is the exemplar), and proposes a **three-dimension 7-metric fully automatic evaluation**: knowledge synthesis (organization & coherency, nugget coverage), retrieval quality (relevance rate, reference coverage, document importance), verifiability (citation precision, claim coverage). All metrics are implemented with an LLM judge and calibrated against human scores. Conclusion: **every system's geometric mean is below 31%, the benchmark is far from saturated**; even OpenAI DeepResearch's best metric (nugget coverage 39.2%) is far from saturation.

- **Key formulas / algorithms**:
  - Task formalization: given a paper description $d$, a retrieval-source set $S$, generate a synthesis $W$, compare with the human exemplar.
  - Relevance Rate: the LLM judge scores each cited source's relevance $Rel(s)\in\{0,1,2\}$, $RR(S)=\frac{1}{2|S|}\sum_{s\in S}Rel(s)$.
  - Reference Coverage: first label the exemplar's citations "important/not", $RC(S,E)=\frac{1}{|E|}\sum_{s\in S}\mathbb{I}[s\in E]$ (recall of important citations).
  - Document Importance: $DI(S,S^*)=\min\!\big(\frac{\text{median cites}(S)}{\text{median cites}(S^*)},1\big)$, measuring citation "gold content" (median citation count vs the human exemplar).
  - Nugget Coverage: extract information nuggets (essential facts) from the human exemplar, compute what fraction the generated report covers (following Great Nugget Recall's LLM method).
  - Organization & Coherency: LLM-as-judge pairwise comparison of generated report vs human exemplar, swap order twice and average to cancel position bias, report win rate.
  - Verifiability: Citation Precision (whether in-sentence citations truly support some claim of that sentence), Claim Coverage (whether the sentence's claims are fully supported by its citations / citations in a sliding window $w$).
  - Data pipeline: restrict arXiv publication-date interval (avoid Llama-4's training cutoff 2025-04-05 to prevent contamination), keep only v1, only conference-accepted papers, with an explicit Related Work section and a .bib.

- **Key experimental results**:
  - DeepScholar-June-2025: 63 papers, 18 arXiv fields, each related-work has on average 23 unique citations, 63% of citations on arXiv. DeepScholar-Nov-2025 expands to 200 queries, 75+ disciplines.
  - Main result (geometric mean): OpenAI DeepResearch highest 0.309; OpenAI o3 search agent 0.287; DeepScholar-ref (GPT-4.1, Claude) 0.286; all systems <0.31. DeepResearch is still <0.40 on Nugget Coverage (.392), Reference Coverage (.187), Document Importance (.124) — "can organize prose well (Org .857), but misses key facts, fails to cite important literature".
  - DeepScholar-ref (LOTUS semantic operators: semantic filter → semantic top-k → semantic aggregation): a strong open-source baseline, verifiability up to 6.3× higher than DeepResearch, 4.3× cheaper, 2.28× faster.
  - Oracle ablation: after feeding the system "the important literature the correct answer should cite" (oracle retriever), Reference Coverage goes to 1.0, but Nugget Coverage only to ~0.49 — **the bottleneck is mainly not retrieval, but "synthesizing good material into key facts"**.
  - LLM-judge calibration (11 CS PhDs, 300+ annotations): Organization pairwise comparison agrees with humans 71.43%, nugget labeling 83.33%, reference importance 65.9% — LLM judges are usable on long-document tasks but still imperfect.

- **Relation to the lecture theme**: this is a positive demonstration of "automatic-evaluation reliability and holes": an LLM judge supports 7 metrics, yet each must be calibrated against humans, and position bias considered; meanwhile "live benchmark + anti-contamination" is an engineered answer to "benchmark saturation / data leakage", echoing METR's critique of benchmark saturation. It is also the template for section 5 "build your own evaluation harness".

- **Demo-ready code points**:
  - Implement several metrics from scratch: given a set of scripted citations (with citation counts, relevance labels, claims), compute RR, RC, DI, understanding how each metric portrays a different side of a "good synthesis".
  - Implement nugget coverage: use a scripted LLM to extract nuggets from the exemplar, then judge whether the generated report covers them.
  - LLM-judge position-bias demo: score the same pair of answers in both orders, count the disagreement rate, then see how "swap average" cancels bias.
  - Geometric-mean aggregation: explain why "geometric mean rather than arithmetic mean" (any dimension at 0 collapses the total, no single-item slacking allowed).

## Teaching thread (how a Stanford instructor might teach this)

1. **First set motivation: what existing benchmarks are actually misleading about.** Throw examples: the same model near full marks on MMLU, yet struggling on SWE-bench; HellaSwag, Humanity's Last Exam are "reverse-filtered" problems made specifically hard for models. Give readers an intuition anchor — **"90 on a multiple-choice test" cannot translate into "can finish several hours of work"**. From this introduce the three papers' common enemies: benchmark saturation, no cross-model unified scale, tasks that are not real.
2. **The first paper METR builds a "ruler": convert ability into human time.** The instructor first gives an intuitive analogy: like measuring a race's difficulty by "how long a human takes to finish". Then unpack three steps: construct a task suite from seconds to 8 hours → find 800+ human baselines for duration (geometric mean) → logistic-fit each model to get a 50% time horizon. Walk readers through one logistic calculation: why when $h_{\text{agent}}=t_{\text{task}}$ the success rate is 0.5. Finally reveal that "doubles every 7 months" log curve, and stress that **the slope is more trustworthy than any single model's absolute height** (errors are highly correlated). Stall hint: readers easily take "task duration" as "model runtime"; stress that duration is what humans did, success rate is what AI did, and the two are joined through the model.
3. **From time to money: GDPval changes the ruler.** Continue: "METR tells how long the work is, not how much it is worth or whether it looks well done." Explain how GDPval top-down picks occupations (9 industries × top-5 pay occupations per industry), how experts make tasks and label them "dollar value = duration × wage". The core demo is the **blind-review win rate** mechanism: pairwise, anonymous, expert judges 0/0.5/1. Here the instructor plants a seed: true-expert scoring is too expensive (one comparison >1 hour), so they trained an automatic scorer — only 66% agreement with humans, which naturally leads to the third paper.
4. **Reliability and holes of automatic evaluation: DeepScholar-Bench closes.** Explain "since humans are too expensive, let the LLM score", but demo three pits one by one: LLM judges have position bias (so pairwise swap order and average), models prefer their own outputs (in GDPval the automatic scorer agrees less on strong models), and the "correctness" of evaluation itself needs re-evaluation (DeepScholar calibrates 71%/83%/66% agreement on 300+ human annotations). Finally use DeepScholar's three-dimension 7 metrics to unpack "good deep research", and use the oracle experiment to point out "the bottleneck is synthesis not retrieval". **The sentence that closes the lecture**: evaluating long-horizon Agents is hard because each of four links has a layer of error — whether the task distribution is real, whether the success criterion is unique, whether the automatic scorer is reliable, and whether those errors themselves have been calibrated.
5. **Down to hands-on: let readers write an eval harness themselves.** In a scripted environment run each of the above metrics once (time-horizon fit, win rate, agreement, nugget coverage), experiencing that "evaluating an evaluation" and "evaluating an Agent" are the same thing.

## Code demo ideas (3–6)

1. **Implement an Agent evaluation harness from scratch**: define a small task pool (e.g. "rewrite a function + unit tests auto-score"), run N times with `llm_client`'s scripted mode, count success rate, gradually construct the minimal closed loop "task-pool definition → run → binarize → aggregate". Expected output: each task's success/fail table and mean success rate, so readers see "evaluation harness = environment + scoring function + aggregate statistics".
2. **Long-task time–completion curve and time-horizon fit**: construct synthetic data `tasks = [(log_time, true_rate)]`, fit $p=\sigma((\log h-\log t)\cdot\beta)$ with `scipy.optimize` or hand-written gradient descent, output the fitted $h$ (50% time horizon), plot data points + fitted curve, mark "the human duration corresponding to this model's 50% success". Then OLS several "models" (different $h$), plot $\log h$ vs time as a line, report doubling days.
3. **Win rate and scoring-agreement computation**: given several groups of {model, human, tie} three-valued scoring simulation data, compute model win rate, then implement $A^{\text{HA}}_s=E[1-|H-A|]$ and $A^{\text{HH}}_s=E[1-|H_1-H_2|]$, contrast "automatic scorer vs humans" and "humans vs humans" as two baselines, reproducing GDPval's "66% vs 71%" structure.
4. **LLM-judge bias demo**: let a scripted/real LLM pairwise-compare the same pair of answers in both orders, count position bias (fraction of order disagreements); then demo how "swap average" removes bias's effect on win rate, and "self-preference" (the judge more easily picks output of its own style / model family).
5. **Try-n-then-fix cost–benefit curve**: using $E[T_n]=(M_T+R_T)\frac{1-(1-w)^n}{w}+(1-w)^n H_T$, fix $M_T,R_T,H_T$, sweep $w$ (model quality) and $n$ (try count), plot "total time vs win rate", find above what model win rate model assistance is cheaper than pure human (crossover), intuitively showing "why naive 327x is distorted".
6. **Three retrieval-quality metrics + oracle ablation**: given a set of scripted citations (each with related/important/citation-count/corresponding claim), implement RR, RC, DI; then simulate "feed the oracle's important literature directly to the system", see RC max out while nugget coverage does not rise with it, reproducing "the bottleneck is synthesis not retrieval".

## Exercise ideas (3)

1. **Fit time horizon**: given a small table of "task human duration + each model's per-task success rate", fit logistic with scipy to obtain a model's $h$ and $\beta$, then assert `abs(h_est - true_value) < tolerance`, and that when $h=t_{\text{task}}$ the predicted success rate is about 0.5. Hint: take log of duration first then logistic-regress; the objective is negative log-likelihood.
2. **Implement win rate and agreement**: given a scoring table of {model_grade, human1_grade, human2_grade}, compute that model's win rate, $A^{\text{HA}}$, $A^{\text{HH}}$, assert that the automatic scorer "is less consistent than humans with each other" and "more consistent on weak models". Hint: $|H-A|$ is computed directly only when both H and A are the three values 0/0.5/1; first think through what a tie should count as.
3. **Implement nugget coverage**: given n nuggets from a human exemplar and a list of sentences from a scripted generated report, use simple keywords / embedding similarity or a scripted LLM to judge whether each nugget appears, compute coverage, assert "report B covers more fully so scores higher". Hint: a nugget is an atomic fact at "essential fact" level; do not treat a whole sentence as a nugget.

## References

- METR. *Measuring AI Ability to Complete Long Software Tasks* (arxiv:2503.14499) — proposes the "50% time horizon" metric, finds frontier Agent long-task ability doubles every 7 months.
- OpenAI. *GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks* (arxiv:2510.04374) — real economically valuable tasks across 44 occupations in 9 industries, blind-review win rate approaching human experts.
- Patel et al. *DeepScholar-Bench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis* (arxiv:2508.20033) — three-dimension 7-metric automatic evaluation of deep-research Agents, no system's geometric mean above 31%.
- Ngo, Richard. *Clarifying and Predicting AGI* (LessWrong, 2023) — source of the "1-month AGI (167 hours)" definition METR uses for extrapolation.
- OpenAI. *Introducing SWE-bench Verified* (2024) — the industry-standard software-engineering benchmark METR uses for external-validity checks.
- OpenAI. *evals.openai.com* — entry to the GDPval gold subset (220 tasks) and experimental automatic scoring service.
- DeepScholar-Bench official repo: https://github.com/guestrin-lab/deepscholar-bench — live data pipeline and evaluation code.
- Rein et al. *HCAST: Human-Calibrated Autonomy Software Tasks* (forthcoming, 2025) — source of one third of METR's task suite.
- Wijk et al. *RE-Bench: Evaluating Frontier AI R&D Capabilities of Language Model Agents Against Human Experts* (arxiv:2411.15114) — 8-hour ML research-engineering tasks and human-expert baselines.
- Panickssery, Bowman, Feng. *LLM Evaluators Recognize and Favor Their Own Generations* (arxiv:2404.13076) — LLM-judge self-preference, cited in the explanation of GDPval's automatic scorer.
