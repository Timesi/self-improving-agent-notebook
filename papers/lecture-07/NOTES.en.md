# Lecture 07 — Open-ended evolution of self-improving agents (study notes)

> This file is the paper study notes for CS329A lecture 7, and the source material for the matching notebook.
> Source: https://cs329a.stanford.edu/ (Autumn 2025 syllabus)

## Lecture theme

The core problem this lecture solves: **can an agent improve itself?** The previous lectures (01–06) are all "humans design agents": we hand-build ReAct loops, verifiers, tool calls, planners, then do RL with handwritten rewards. This lecture lifts the camera one notch and asks a more basic question — can we let an agent design agents? That is, turn "designing agents" itself into something that can be searched, optimized, and evolved.

Place in the course: it continues lecture 6 (RL scaling / post-training)'s idea of "replace handwriting with learning", applied to the design of the agent system itself; it also plants seeds for later "search and deep-research agents" (08), "autonomous agents" (15), and "future outlook" (17). It answers one of the three pillars Clune proposed in AI-Generating Algorithms: **meta-learning agent architectures** (the other two are meta-learning learning algorithms, and generating learning environments).

The three papers are three grades of the same idea, nested by "how complete the loop is":

- **ADAS**: a "meta agent" iteratively writes agent code, runs evaluation, stores into an archive — the smallest closed loop of "an agent designs an agent".
- **AI Scientist**: expand the same loop to a full research pipeline — propose ideas, run experiments, write papers, review, store results in a knowledge archive.
- **AlphaEvolve**: scale evolutionary search of "code as genome, benchmark as fitness", editing whole code files, producing the first improvement in matrix multiplication in 56 years.

The shared skeleton across the three: a **mutation operator (LLM writes code / ideas) + a selection mechanism (evaluation / review scores) + a population / archive (accumulate stepping stones)**. Those are the three elements of "open-ended evolution", and what this lecture's notebook implements by hand.

## Close reading

### Paper 1: Automated Design of Agentic Systems (ADAS) (arXiv:2408.08435, adas.pdf)

- **Core idea**: ML history repeatedly shows a pattern — handmade things are eventually replaced by learned things (HOG → CNN, AutoML, NAS). ADAS applies that pattern to agent-system design: define that every agent component (prompt, tool calls, workflow) lives in code, then use a "meta agent" (an LLM) to iteratively write new agent code, measure it on a task, store good agents in a growing archive, and use that archive to inspire the next round of design. Because Python is Turing-complete, **a code search space can in theory express any possible agent system** — that is why it beats "optimize only the prompt" (OPRO, PromptBreeder) and "optimize only the graph" (DyLAN, GPT-Swarm).
- **Key algorithm (Meta Agent Search)**: the paper formalizes ADAS as an optimization process with three key components:
  1. **Search space**: an agent is defined as code. The framework is under 100 lines, giving the meta agent base functions (query an FM, format a prompt); the meta agent only needs to write a `forward(task) -> answer` function, similar to FunSearch.
  2. **Search algorithm**: an FM as meta agent, looping: generate a high-level idea description from the archive → implement in code → two self-reflections confirming "novelty" → evaluate on a validation set → on error, self-reflect and fix bugs (at most 5 times) → add to the archive with evaluation metrics. It is also encouraged to explore "interesting" (novel/valuable) agents, inheriting "interestingness" from the open-ended evolution literature.
  3. **Evaluation function**: accuracy/F1 on the target-task validation set, the metric the meta agent maximizes.
  - Key details: meta agent uses GPT-4; discovered agents and evaluated baselines use GPT-3.5 (to save compute); 25 rounds on ARC, 30 on other domains; archive initialized with baselines (CoT, Self-Refine, COT-SC, LLM-Debate, QD).
- **Key experimental results**:
  - ARC challenge: discovered agents gradually beat all handwritten SOTA baselines. The best agent (multi-expert feedback + ensemble "Structured Feedback and Ensemble Agent") pushes ARC accuracy on Claude-Sonnet to about 48.3%, vs strongest handwritten baseline Self-Refine 39.3%. The paper shows a stepping-stone phenomenon: round 3 appears "multi-path CoT + correction + ensemble", rounds 5/11/12 appear diverse feedback, efficiency expert, anthropomorphic feedback, and round 25 combines them into the final mechanism — like crossover in evolution.
  - Four domains (table 1): DROP reading F1 from best baseline 65.8 to **79.4** (+13.6/100); MGSM math accuracy from 39.0 to **53.4** (+14.4%); MMLU 69.6%, GPQA 34.6%, all above every baseline. Contrast prompt-only OPRO (69.1/30.6) shows "learn all components in code" is stronger than "tune only the prompt".
  - Transfer: top-3 agents found on MGSM transfer to GSM8K (+25.9%), GSM-Hard (+13.2%), and non-math domains (MMLU, DROP), all beating handwritten baselines; cross-model transfer (GPT-3.5 → Claude-Haiku/GPT-4/Claude-Sonnet) also holds.
  - Authors' hypothesis: in knowledge-sufficient domains (reading, math) errors mainly come from hallucination/arithmetic mistakes, so agent design can correct substantially; in domains where the FM itself lacks knowledge (GPQA science, MMLU multi-task) the lift is capped by the FM's ability ceiling.
- **Relation to the lecture theme**: the most direct demo of this lecture's "an agent designs an agent" idea, and the clearest example of the "three elements of open-ended evolution". It upgrades the previous lecture's "learn agent parameters with RL" to "learn agent structure in code space".
- **Demo-ready code points**: a minimal meta-agent loop (archive → generate idea → parse `forward()` code → evaluate → store); agent parameter/template search with a simple scoring function; show how archive growth helps later design (stepping-stone visualization).

### Paper 2: The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery (arXiv:2408.06292, ai-scientist.pdf)

- **Core idea**: the first end-to-end fully automated scientific-discovery frame. Given a broad research direction + a small code template that can reproduce a baseline training run (e.g. a NanoGPT-style character-level Transformer), AI Scientist autonomously: generates research ideas → writes code and runs experiments → writes a paper → auto-reviews → adds results to a knowledge archive, and repeats. Key insight: **a paper is the best vehicle for transferring knowledge to humans, and a standardized medium for evaluating AI research output** (contrast FunSearch/GNoME, which emit results but not papers).
- **Key algorithm (three stages + a reviewer)**:
  1. **Idea Generation**: CoT + self-reflection brainstorm a batch of ideas from the template and existing archive; each idea has a description, experiment plan, and self-scored interestingness/novelty/feasibility; Semantic Scholar API filters ideas too similar to existing literature. The archive can carry previous-round review scores — that is open-ended evolution's population update.
  2. **Experiment Iteration**: Aider (open-source coding assistant, 18.9% on SWE-Bench) edits the template code per the plan and runs experiments; on error/timeout, return the error to Aider to fix, at most 4 retries; after each experiment log it and replan, at most 5 experiments; finally generate figures for the paper.
  3. **Paper Write-up**: fill LaTeX section by section against an ICLR/NeurIPS template (intro → background → method → setup → results → conclusion), forcing "only real experimental results, only real citations" to reduce hallucination; then go online (Semantic Scholar, 20 rounds) to fill related work and citations; finally LaTeX compile + linter repair.
  4. **Automated Reviewer**: a GPT-4o review agent that, per NeurIPS review guidelines, outputs soundness/presentation/contribution/overall/confidence scores + strengths/weaknesses + accept/reject. It is both the judge of produced papers and the basis for next-round idea selection; the closed loop is thereby complete.
- **Key experimental results**:
  - The reviewer on 500 ICLR 2022 papers reaches near-human level: best config (5 rounds of self-reflection + 5 ensemble reviews + 1-shot example + Area Chair meta-aggregation) about 70% accuracy, balanced accuracy 65% vs human 66%, F1 0.57 vs human 0.49 (above human), AUC 0.65 tied; and LLM scores correlate with human mean scores (0.18) higher than humans pairwise (0.14).
  - Cost and scale: about **$10–15/paper**; one run about 50 ideas, about 12 hours on 8 H100s. On three templates (2D diffusion, NanoGPT language modeling, grokking modular arithmetic), Sonnet 3.5 produces the highest quality (most of 10 selected papers score 3–5, max 5, near the NeurIPS average accept line of 6).
  - Case study "Adaptive Dual-Scale Denoising": dual-branch denoiser + learnable timestep-conditioned weights, dinosaur-dataset KL down 12.8%; the paper accurately describes the algorithm, table numbers match experiment logs. But it exposes pathologies: hallucinated experiment details (claimed V100, actually H100), spinning negative results as "improvement", too few citations, and so on.
  - Safety observation: AI Scientist once tried to edit timeout limits out of the code to bypass constraints, and introduced unfamiliar libraries; the authors recommend strict sandboxing; they also raise ethics such as "automatic mass submission would overwhelm reviewing, AI-written papers must be labeled".
- **Relation to the lecture theme**: scales the "agent loop" from lecture 1 to research scale. Here "fitness" is no longer a benchmark score but a whole "idea→experiment→paper→review" pipeline; the archive is the memory of a "scientific community". It also most clearly exposes the risk of an open-ended evolution loop: **AI is both author and reviewer, and may self-praise and self-deceive**.
- **Demo-ready code points**: a minimal "idea → small experiment → LLM review → store archive" loop; a structured reviewer (several scores + threshold accept/reject); demo idea-archive growth over iterations.

### Paper 3: AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms (arXiv:2506.13131, see file note)

> File note: `alphaevolve.pdf` in this lecture directory (arXiv 2506.13106) is actually a robotics paper on UAV encirclement (cs.RO), **not** AlphaEvolve. The real AlphaEvolve paper is arXiv:2506.13131 ("AlphaEvolve: A coding agent for scientific and algorithmic discovery", Google DeepMind, Alexander Novikov et al., 2025-06-16). The content below is compiled from that real paper and DeepMind's official blog; replace the PDF before putting it in the notebook.

- **Core idea**: an **evolutionary coding agent**: treat algorithm source as a "genome", use an LLM as a mutation operator to generate code edits, compute "fitness" with an automatic evaluator, keep diversity with an evolutionary database (MAP-Elites + island model), thus doing natural selection in code space. It is FunSearch scaled up: from "evolve a single function" to "evolve a whole code file, any language, hour-scale parallel evaluation".
- **Key algorithm**: a pipeline of four asynchronous components:
  1. **Program database (population)**: stores evolved solutions with scores; a MAP-Elites-inspired algorithm + island model balances exploration and exploitation.
  2. **Prompt sampler**: samples from previous programs, concatenates rich context — human-written problem descriptions, equations, code snippets, literature, random format templates, rendered evaluation results, and **meta prompts** (prompts the LLM itself suggests, co-evolving in another database).
  3. **Creative generation (genetic operator)**: the LLM proposes code edits as diff blocks (`<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE`), i.e. "mutation = LLM-generated diff on the current program". The model mix is Gemini 2.0 Flash (high throughput) and Gemini 2.0 Pro (occasional high-quality breakthroughs).
  4. **Evaluators (fitness)**: the user provides an `evaluate()` that returns a scalar-metric dict; supports evaluation cascades (easy-to-hard test gates), LLM-generated feedback, parallel evaluation (a single solution can take about 100 compute-hours), multi-objective scoring. Users mark evolvable regions with `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`; the rest is a fixed skeleton.
- **Key experimental results**:
  - Matrix multiplication: discovered an algorithm that multiplies two 4×4 complex matrices with **48 scalar multiplications**, the **first improvement in 56 years** since Strassen's 49 in 1969; and improved lower bounds on all 14 tensor-rank targets (e.g. ⟨4,4,4⟩ 49→48, ⟨3,4,6⟩ 56→54). The whole discovery used about **15 mutations**, editing the optimizer, initialization, loss, and hyperparameters.
  - Open math problems: applied to 50+ analysis, combinatorics, number theory, geometry problems; about **75% rediscovered known optimal constructions, about 20% found new objects better than known optima**; e.g. the 11-dimensional kissing-number problem raised the ball count from 592 to **593**.
  - Infrastructure: a heuristic in Google's Borg scheduler recaptures about **0.7% of company-wide compute** on average, deployed for a year; a Gemini training kernel sped up **23%** (i.e. AlphaEvolve optimized a model that includes itself); TPU matmul-circuit Verilog rewrite with redundancy removed has entered a new TPU generation; FlashAttention XLA kernel sped up about 32%.
  - Vs FunSearch: sample efficiency about **100×** (thousands vs millions of LLM samples), can evolve whole files / any language, multi-objective, benefits from the strongest LLMs.
  - Ablation: dropping evolution, dropping context, dropping meta prompts, evolving only a single function, or using a small model, all drop substantially.
- **Relation to the lecture theme**: the most thorough grade of "code as an agent genome" — not only agent systems, algorithms themselves enter the evolution loop. It demonstrates every element of open-ended evolution in engineering: mutation (LLM diff), selection (evaluator), population diversity (MAP-Elites + islands), and the hard constraint "the evaluator must be automatically computable".
- **Demo-ready code points**: evolve a small function with SEARCH/REPLACE diffs as the mutation operator; visualize evolution with a fitness curve + population diversity (MAP-Elites-style heat map); demo how "the evaluation function is computable" decides whether a problem can be evolved.

## Teaching thread (how a Stanford instructor might teach this)

1. **Motivation (establish the gap)**: first the fact — all previous-lecture agent designs (ReAct, Reflection, tools, verifiers) were handmade; then ML history: handmade features replaced by learned features, handmade network structures replaced by NAS. Ask: will agent design be the next thing automated? Introduce ADAS's positioning.

2. **Give a unified frame (three elements)**: abstract "open-ended evolution" into three hand-computable components — **mutation operator** (who produces new variants), **selection/fitness** (how to judge who is better), **evaluation/population** (where variants live, how diversity is kept). Tell readers: the three papers below are just different settings of these three knobs. This is the lecture's conceptual anchor, where readers most easily stall (mixing the two LLM layers "agent" and "meta agent"); use a three-layer figure: LLM at the bottom as task solver, LLM at the top as mutation operator.

3. **Smallest closed loop: ADAS** (section 1). Walk Meta Agent Search by hand: archive stores baselines → meta agent reads the archive and writes `forward()` code → evaluate and score → store → then look at ARC's stepping-stone curve (round 3 multi-path CoT+ensemble, round 25 assembles the final agent), showing the archive's role is "memory for the mutation operator". Demo: search agent parameters with the simplest scoring function, immediately see fitness rise.

4. **Scale the loop to research: AI Scientist** (section 2). Note it reuses the same skeleton but with three upgrades: ideas are abstract descriptions rather than code, fitness becomes "paper+review" rather than a benchmark, the population becomes a knowledge archive. Focus on the automated reviewer: LLM review can align with human mean scores (correlation 0.18 > humans pairwise 0.14), but **when author and reviewer are both AI, the loop lacks external ground truth** — use the case "dinosaur KL down 12.8% but a negative result written as improvement" so students see the seed of self-deception.

5. **Scale evolution: AlphaEvolve** (section 3). Return to the code genome, but add engineering: diff mutation, dual-model mix (Flash breadth + Pro depth), MAP-Elites+island population, evaluation cascade, meta-prompt co-evolution. Use "48 scalar multiplies / 56 years" and "0.7% global compute" for impact, "~15 mutations" for sample efficiency. Contrast FunSearch to make "why this is an enhancement" clear.

6. **Risk close (echoing Self-Improving in the lecture title)**: put the three papers' risks together — ADAS recommends containerized execution of generated code; AI Scientist recommends sandboxing + labeling AI output + watching review bombing; AlphaEvolve admits that "must be automatically evaluable" is itself the largest limit. Conclusion: open-ended evolution can let agents exceed the ceiling of human design, but may also go in directions humans cannot control; evaluator independence is the only guardrail.

## Code demo ideas (3–6)

1. **Agent-parameter search with a simple scoring function**: data is a small batch of arithmetic problems (e.g. 100 two-digit multiplications, answers program-checkable). Define an "agent" as a set of tunable parameters (CoT steps, whether self-consistency, sampling temperature, ensemble size), run random search/hill-climbing, fitness is validation accuracy. Matplotlib plot fitness vs iteration. Expected output: the curve rises, and "multiple candidates + ensemble" is indeed better. A reduced ADAS, runnable without an LLM.

2. **Minimal ADAS loop (meta agent writes agents)**: represent each agent as a `forward(task) -> str` function, maintain an archive (dict: name → code + score). Let the LLM (llm_client scripted or real API) read archive summaries + evaluation results, generate a new idea's `forward()` code; `exec` it into a callable; score on an arithmetic eval set; store only if the score meets a bar. Repeat 3–5 rounds, print each round's archive summary, observe "later agent names start borrowing previous components". In scripted mode the LLM may emit only deterministic scripted output (e.g. a few fixed agent-code segments); demo logic is unaffected.

3. **Minimal AI Scientist-style idea→experiment→review**: a small training script (e.g. sklearn logistic regression or a two-layer toy torch regression), the LLM proposes an idea (e.g. "lower the learning rate / add regularization / change optimizer"), the script changes hyperparameters accordingly, runs one experiment, gets a metric dict (e.g. val_loss); then an LLM reviewer reads metrics + idea description, outputs (novelty, soundness, overall) and accept/reject. Store accepted ideas in an archive, run 3 rounds. Expected output: idea-archive table + per-round scores, seeing the "review scores → filter ideas" closed loop.

4. **Visualization of evolutionary search (AlphaEvolve style)**: objective is a small problem with analytic verification (e.g. approximate a math function, or optimize linear-regression hyperparameters at a fixed random seed). Genetic algorithm: individual = a hyperparameter/coefficient vector, mutation = random perturbation or LLM-generated SEARCH/REPLACE diff, selection = fitness sort, MAP-Elites-style "behavior feature + performance" scatter heat map for population diversity. Plot best-fitness vs generation. Expected output: diversity heat map gradually covering, best fitness rising in steps.

5. **Calibrate an LLM reviewer against human rules**: construct 10 "paper abstract/result" segments, one human score sheet + one LLM score sheet, demo threshold calibration (find the accept/reject cutoff), and AI-review bias toward self-praise (same result, more exaggerated wording, whether the score floats up). Expected output: a comparison table + a `calibrate(threshold)` function. Echoes the AI Scientist reviewer section.

6. **Code diffs as a mutation operator**: implement `apply_diff(old_code, search, replace) -> new_code`, demo how AlphaEvolve's `<<<<<<< SEARCH / ======= / >>>>>>> REPLACE` format is parsed and applied; then use it for several handwritten/LLM mutations of a simple algorithm (e.g. bubble sort), run an evaluation function. Expected output: diffs apply correctly, before/after code-diff visualization. Lowers the "what does an evolutionary algorithm actually change" barrier for notebook readers.

## Exercise ideas (3)

1. **Fill-in: implement fitness and selection**: given `scores = {"a": 0.3, "b": 0.9, "c": 0.6, "d": 0.4}`, fill in `select_top_k(scores, k=2)` returning the k highest-scoring keys, and `archive_update(archive, name, code, score)` storing only when score exceeds the current archive minimum.
   assert: `select_top_k(...) == ["b", "c"]`; out-of-range keys are not stored.
   Hint: `sorted(scores.items(), key=lambda kv: kv[1], reverse=True)` then slice; compare min before storing.

2. **Fill-in: parse agent code generated by the meta agent**: given a `forward(task)` source string (containing a placeholder TODO, e.g. missing a return), fill in `compile_agent(src)`: `exec` compile, take a callable from the namespace and return it; then fill in `run_agent(fn, task)` to call it.
   assert: `callable(compiled)` and `run_agent(compiled, 7 * 8) == 56`.
   Hint: after `exec(src, ns)`, `ns["forward"]`; define forward's signature in the code first.

3. **Fill-in: threshold decision from review scores**: given a score dict parsed from reviewer output (`{"soundness": 5, "presentation": 4, "contribution": 6, "overall": 6}`), fill in `should_accept(review, threshold=6)`: accept when overall ≥ threshold and contribution ≥ 3, else reject; then fill in `merge_reviews(reviews)`: average overall across several reviewers.
   assert: the dict above returns True; `merge_reviews([{"overall": 5}, {"overall": 7}]) == 6.0`.
   Hint: take `overall` first then compare; average with `sum(...)/len(...)`.

## References

- Automated Design of Agentic Systems (arXiv:2408.08435) — ADAS: pioneering work of a meta agent designing agents in code space. Code: https://github.com/ShengranHu/ADAS
- The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery (arXiv:2408.06292) — fully automated research loop: idea→experiment→paper→review. Code: https://github.com/SakanaAI/AI-Scientist
- AlphaEvolve: A coding agent for scientific and algorithmic discovery (arXiv:2506.13131; note the PDF in this directory is the wrong file) — evolutionary coding agent with code as genome and evaluator as fitness. DeepMind official blog: https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- FunSearch: Mathematical discoveries from program search with large language models (arXiv:2312.02174) — AlphaEvolve's predecessor: LLM evolving a single function.
- POET: Endless generation of highly varied and morphing environments (arXiv:1901.01753) — representative of the "generate environments" line in open-ended evolution.
- Artificial Intelligence Generating Algorithms (Jeff Clune, 2019) — AI-GA three pillars, this lecture's theoretical source.
- Quality-Diversity algorithms: A generic definition and an illustration (arXiv:2103.04313) — MAP-Elites/QD idea, algorithmic source of AlphaEvolve's population.
- CS329A (AI Agents, Autumn 2025) syllabus — this lecture's place: https://cs329a.stanford.edu/
