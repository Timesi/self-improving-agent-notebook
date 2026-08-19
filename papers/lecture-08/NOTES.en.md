# Lecture 08 — Search and deep-research agents (study notes)

> This file is the paper study notes for CS329A lecture 8, and the source material for the matching notebook.
> Source: https://cs329a.stanford.edu/ (Autumn 2025 syllabus)

## Lecture theme

This lecture answers one core question: when "generate one answer once" is not enough, how an agent uses **search** to substantially lift performance on complex tasks. The search object can differ a lot: AlphaCode searches in **program space** (generate a million candidate programs for one problem, then pick the ones most worth submitting); Search-o1 searches in **knowledge space** (retrieve external knowledge on demand during long reasoning to fill knowledge gaps). Both share the same skeleton: **generate → verify/filter → pick/continue**.

This lecture's place is important. Earlier L6 covers test-time compute scaling (more inference compute for better results) and L7 covers open-ended evolution (search+verify loops for self-improvement). This lecture pushes "search" into two concrete, hands-on system cases, and supplies primitives for later deep-research-agent workflows (section 4): **reasoning-actor (decides when to search) + retrieval-critic (refines retrieval results)**. It is the bridge from "the model introspects its reasoning" to "the model actively calls the outside world", and it connects to L9's post-training evolution.

## Close reading

### Paper 1: Competition-Level Code Generation with AlphaCode (arXiv:2203.07814, alphacode.pdf)

- **Core idea**: program synthesis is a problem of searching a **huge and structured** program space with extremely sparse reward — a single character change can change program behavior, legal solutions to the same problem can look completely different, and "whether this program is close to correct" is hard to judge. AlphaCode's response is to model the problem as sequence-to-sequence translation (natural-language statement → program), with a large encoder-decoder transformer, and at inference do **large-scale sampling + behavior-based filtering + clustering**, compressing each problem to at most 10 submissions. This is the first AI system to reach "median contestant" level in programming contests.

- **Key formulas / algorithms**:
  - **Data**: first pretrain on 715.1 GB of curated GitHub code; then finetune on the self-built CodeContests dataset (train 13328 / valid 117 / test 165 problems, strict time split to prevent leakage). **Test mutation generation** adds extra tests (flip bits, randomly add/delete integers, change strings), pushing the dataset's false-positive rate from 62% down to 4%.
  - **Model**: encoder-decoder transformer, 300M to 41B parameters; asymmetric structure (encoder 1536 tokens / decoder 768 tokens, statements are on average twice as long as solutions); multi-query attention mainly for **faster sampling**.
  - **Finetuning**: GOLD objective + tempering + value conditioning/prediction + random tags/ratings conditioning.
  - **Sampling**: at most 10^6 candidates per problem; half Python half C++; randomize tags in the prompt (randomly from the 50 most popular) and difficulty ratings (uniform 800–3500) to increase diversity; sampling temperature 0.25. Sampling can run in parallel, the key lever of performance.
  - **Filtering (verifier)**: execute each candidate on the **sample tests** given in the statement, drop those whose output does not match. This removes about 99% of samples (samples that pass sample tests <1%); about 10% of problems have no sample that passes sample tests at all. Filtering is "compressing candidates under an unsupervised verification signal".
  - **Clustering (diversity management)**: after filtering, thousands of candidates may remain, but there are only 10 submission slots. AlphaCode trains another "**test-input generation model**", generates new test inputs from the statement, runs all remaining programs, and clusters programs with the same output by **behavior signature**. Correct solutions behave similarly and form the **largest cluster**, so taking the top 10 clusters by size and one submission per cluster maximizes coverage of different correct solutions.
  - **Metrics**: `10@k` (pass rate allowing n=10 submissions picked from k samples) and `pass@k` (submit all k samples, i.e. k@k, a theoretical upper bound); bootstrap estimate of expectation to lower variance. The gap between `10@k` and `pass@k` measures **picker quality**.

- **Key experimental results**:
  - 10 recent Codeforces contests (each >5000 participants) average rank **top 54.3%**, estimated Codeforces rating 1238 (above 72% of users who competed in the last 6 months); 2.4 submissions on average per solved problem.
  - CodeContests: 41B + clustering, validation `10@1M = 34.2%`, test `10@100k = 29.6%` (fully unseen problems with no leakage).
  - **Solve rate grows approximately log-linear with sample count**, and larger models have a steeper slope (figure 6); raising the sampling budget is exponential cost for logarithmic gain.
  - Ablation: filter without clustering does not raise solve rate (figure 8 three-curve contrast); **filter+cluster is required to turn "more samples" into "higher solve rate"**; stacking each enhancement (MLM, tempering, tags/ratings, value, GOLD, clustering) raises `10@100k` from 15.2% to 24.1%.
  - The model does not memorize problems: longest-common-substring analysis is at human-contestant magnitude, mainly input-parsing boilerplate, not core logic.

- **Relation to the lecture theme**: AlphaCode is the textbook case of "**program synthesis as search**", and the first building block of this lecture. It lands L6 test-time compute scaling in a concrete system: inference spends large sampling compute to **explore** candidate space, sample tests as a **cheap verifier**, behavior clustering as **de-duplicated picking**. All three links are required; this is the prototype of every later "generate → verify → pick" agent loop.

- **Demo-ready code points**:
  - Implement a "sample → filter → cluster" pipeline from scratch (`llm_client` scripted or a real model generates candidates, Python executes sample tests to filter, random inputs produce behavior signatures then cluster).
  - Plot two log-linear curves of `10@k` and `pass@k` vs sample count, seeing "filter+cluster" pull 10@k toward pass@k.
  - Demo behavior clustering in a synthetic small program space: correct solutions form one large cluster, wrong solutions scatter into small ones.

### Paper 2: AlphaCode 2 Technical Report (2023-12-06, alphacode2-report.txt / alphacode2-report.pdf)

- **Core idea**: AlphaCode 2 is the upgraded AlphaCode system, swapping the base to Gemini Pro, keeping and strengthening the "**large-scale sampling + filtering + clustering**" trio, and adding a **reranking (scoring model)** step, solving 43% of Codeforces problems, reaching an estimated **85th percentile** (AlphaCode only 46%). More striking is sample efficiency: AlphaCode 2 with about 100 samples matches AlphaCode with 1 million (**>10000× fewer samples**).

- **Key formulas / algorithms**:
  - **Policy model family**: start from Gemini Pro, two GOLD finetune rounds (first CodeContests v2, then a higher-quality dataset), vary hyperparameters to get a family of policy models, using **a family rather than a single model** to maximize output diversity.
  - **Sampling**: at most 10^6 samples per problem, random sampling temperature, randomize prompt metadata (difficulty rating, category tags); budget split evenly across the family; **C++ only** (experiments found C++ sample quality higher).
  - **Filtering**: execute sample tests, drop output mismatches and compile failures (<5%), removing about **95%** of samples on average.
  - **Clustering**: after filtering, about 50k candidates remain on average; reuse AlphaCode's idea — separately train a test-input generation model, run new inputs for behavior signatures, sort by cluster size, keep only the **10 largest clusters**, avoiding redundant submissions.
  - **Scoring model (new step)**: finetune a second Gemini Pro specifically to give programs a 0–1 "estimated correctness" score; within each cluster pick the best candidate by score, assembling the final 10 submissions. This upgrades "pick one per cluster" to "pick one by estimated correctness within the cluster".

- **Key experimental results**:
  - 12 recent Codeforces contests (>8000 participants, div2 or 1+2), 77 problems: solved **43%**, about 1.7× AlphaCode (25%).
  - Estimated rank **85th percentile** (between Codeforces Expert and Candidate Master); the two best contests beat 99.5% of contestants.
  - **Sample efficiency**: about 100 samples already reach AlphaCode's 1-million-sample performance, **>10000×** more sample-efficient; solve rate still grows approximately log-linear with sample count (figure 3).
  - With human collaboration (additional filter rules supplied) scores exceed the 90th percentile.

- **Relation to the lecture theme**: AlphaCode 2 shows the **evolution direction** of a "search system": stack a **learned reranker** on the same trio, and swap in a stronger base. Two lessons: (1) filter/cluster/rerank is the engineering core of "turning massive generation into a limited number of high-quality submissions"; (2) **base-model quality and required search volume have a strong tradeoff** — each bit better the model, required samples fall exponentially, explaining why continuously lifting the model (post-training, RL) and doing search well are equally important.

- **Demo-ready code points**:
  - Add a simple reranker to an AlphaCode-style pipeline: rank candidates by "behavior/output consistency on extra random inputs" or a finetuned scorer; contrast random submit, filter-only, filter+cluster, filter+cluster+score `10@k`.
  - Simulate "model quality ↑ → required samples ↓": plot families of log-linear curves with simulated generators of different quality, observe high-quality models shift left (fewer samples for the same solve rate).
  - Contrast "one from each of the 10 largest clusters" vs "highest-scored within each cluster".

### Paper 3: Search-o1: Agentic Search-Enhanced Large Reasoning Models (arXiv:2501.05366, search-o1.pdf)

- **Core idea**: o1-class **long reasoning models** (LRMs, e.g. OpenAI o1, QwQ, DeepSeek-R1) train stepwise long-chain reasoning with large-scale RL, but often fail when knowledge is insufficient — statistics show high-frequency uncertainty words in reasoning ("perhaps" appears 30.4 times per output on average). Standard RAG retrieves once by the **original question**, almost useless for different knowledge gaps at each step of multi-step reasoning. Search-o1 embeds **agentic RAG** (on-demand, multi-round, self-triggered retrieval during reasoning) into the reasoning chain, and adds a **Reason-in-Documents** module that **refines** retrieved documents before injecting them, avoiding long redundant documents breaking reasoning coherence.

- **Key formulas / algorithms**:
  - **Reasoning-actor (when to search)**: while generating reasoning chain $R$, on uncertain knowledge the reasoning model itself emits a search query wrapped in special symbols: `<|begin_search_query|>...<|end_search_query|>`. Detecting the end marker **pauses reasoning**, extracts query $q^{(i)}_{search}$, calls retrieval $D^{(i)}=\mathrm{Search}(q^{(i)}_{search})$ (implemented with Bing Web Search API, top-k=10).
  - **Retrieval-critic (how to use)**: the Reason-in-Documents module runs independently of the main chain: given current chain $R_{<i}$, search query $q^{(i)}_{search}$, retrieved documents $D^{(i)}$, first generate intermediate analysis $r^{(i)}_{docs}$ (understand the documents), then produce refined knowledge $r^{(i)}_{final}$, inserted back into the main chain as `<|begin_search_result|>...<|end_search_result|>`, then reasoning continues.
  - **Reasoning loop**: generate a reasoning step → on a query, retrieve+refine+inject → continue generating → until EOS. The whole process can **trigger multiple retrieval rounds inside one reasoning chain**, even if each step's knowledge need differs.
  - **Batch inference**: for a batch of problems, maintain in parallel "unfinished sequence set S / finished set F", all sequences generate in parallel to EOS or a query end marker, then batch retrieve, batch refine, batch inject, raising throughput.
  - **Generation settings**: max 32768 tokens, temperature 0.7, top_p 0.8, top_k 20, repetition penalty 1.05, backbone the open-source QwQ-32B-Preview; if retrieval finds no answer, fall back to direct reasoning.
  - Mathematical form: model reasoning as $P(R,a|I,q,D)=\prod_t P(R_t|R_{<t},I,q,D_{\le t})\cdot\prod_t P(a_t|a_{<t},R,I,q)$, retrieval inserted into the reasoning chain at token level.

- **Key experimental results**:
  - GPQA diamond (198 problems, PhD-level science QA): Search-o1 overall **63.6%**, above direct-reasoning QwQ (58.1%) and agentic RAG baseline RAgent (61.6%); on the GPQA extended set overall **57.9%**, above human expert groups by field (physicists 39.9 / chemists 48.9 / biologists 37.2).
  - Five complex-reasoning tasks average: Search-o1 4.7% above RAgent, 3.1% above direct QwQ.
  - On multi-hop open-domain QA, agentic RAG raises average EM **23.2%** over standard RAG; single-hop tasks have no significant difference (47.8 vs 47.6) — evidence that "on-demand multi-round retrieval" is valuable when knowledge really needs combining.
  - **top-k scaling**: retrieving **1** document already beats direct reasoning and standard RAG with 10 documents, showing the refine module amplifies retrieval quality.
  - Ordinary instruction models (Qwen2.5-32B) + agentic RAG do not lift on GPQA and fall on math/code — **ordinary LLMs do not treat search as a tool**; this is a capability specific to reasoning models.

- **Relation to the lecture theme**: Search-o1 generalizes "search" from program space to **knowledge space**, and unifies the skeleton of this lecture's two systems — AlphaCode explores program-candidate space at inference, Search-o1 explores external knowledge at inference; both are "**generate → judge whether more is needed → obtain external information → continue**". The split of reasoning-actor vs retrieval-critic (decide when to search vs refine how to use) is the core primitive of section 4's "deep-research workflow", and a form of "self-improvement": the model is no longer stuck at the knowledge ceiling of training time, and can dynamically fill knowledge at inference.

- **Demo-ready code points**:
  - Implement a Search-o1-style **retrieval-injection loop** from scratch: `llm_client` (scripted mode gives a scripted trajectory) generates a reasoning chain with `<|begin_search_query|>`, parse the query → look up a small knowledge base (dict/DataFrame) → refine and inject → continue generating, count retrieval rounds.
  - **Uncertainty-word detection**: count "perhaps / wait / likely" in the reasoning chain as a quantitative "knowledge gap" signal, bar chart reproducing figure 1.
  - Contrast **Standard RAG (one retrieval at question level) vs agentic RAG (multi-round on demand)** on a problem that needs multi-step fact-checking.

## Teaching thread (how a Stanford instructor might teach this)

1. **Motivation: why a single answer is not enough.** Open with a contest problem (e.g. Backspace, Codeforces rating 1500): let an LLM generate a program and submit, often failing at "turning the statement into algorithmic reasoning" rather than API calls. That introduces the first main line: program synthesis = search in a huge structured program space, sparse reward, correct solutions are not unique. The instructor's analogy: find a few right needles in a haystack, and you cannot see the whole needle (hidden tests), so first filter with visible sample tests.

2. **The trio: sample → filter → cluster.** Cover why each step is necessary:
   - **Sampling**: one generation is not enough; high temperature, randomized prompt metadata, half Python/C++, generate at million scale; show `10@k` rising approximately log-linear with sample count (figure 6).
   - **Filtering**: sample tests as verifier, cut about 99% of candidates; stress this is "a cheap correctness signal under no supervision".
   - **Clustering**: after filtering still thousands of candidates but only 10 submissions; cluster by behavior signature (outputs on generated test inputs); correct solutions form large clusters. Hand-calculate a 3-program × 3-test-input signature matrix, so readers see "same output = same behavior = cluster". Stall warning: readers often mix `pass@k` (submit all, upper bound) and `10@k` (can pick only 10), and forget that filtering can only use sample tests, not hidden tests.

3. **AlphaCode → AlphaCode 2: add a reranker, swap a stronger base.** Show how the same system evolves: after filtering about 50k candidates remain, a dedicated scoring model picks the best within each cluster; swapping in stronger Gemini Pro lets 100 samples match AlphaCode's 1 million (figure 3). The instructor stresses the tradeoff: **model quality and search volume can be exchanged** — the stronger the model, the less search needed, the lower the cost.

4. **From contest code to deep research: Search-o1.** Abstract "search" into a more general primitive. First a failure case: long-reasoning models emit many uncertainty words (perhaps 30 times on average), standard RAG (retrieve once) is useless because each reasoning step needs different knowledge. Then Search-o1's two roles: **reasoning-actor** (the model itself decides when to emit a search query, `<|begin_search_query|>` triggers retrieval) and **retrieval-critic** (Reason-in-Documents first refines a long document into a paragraph before injecting, keeping reasoning coherent). Contrast with AlphaCode: one searches program space, one searches knowledge space, both skeletons are "generate → judge → obtain → continue".

5. **Deep-research workflow.** Finally assemble the three papers: a deep-research agent = decompose a complex problem into subproblems (planning), search on each (agentic RAG), then synthesize a report. AlphaCode's "filter+cluster" and Search-o1's "reasoning-actor + retrieval-critic" are reusable building blocks of this workflow. Stall warning: readers easily think "one retrieval is enough" (Standard RAG thinking), or mix "refine documents" with "stuff long text directly" — refine's value is keeping the reasoning chain coherent.

## Code demo ideas (3–6)

1. **Implement a sample-filter-cluster code-generation pipeline from scratch**: with `llm_client` (scripted mode provides deterministic candidates, or a real model) generate N candidate programs for a small problem, execute sample tests to filter, generate behavior signatures on random inputs and cluster, finally pick 10 submissions. Key code idea: `exec`/`subprocess` run Python to get output, `collections.defaultdict(list)` group by signature. Expected output: filter pass rate, cluster count and sizes, `10@k` vs `pass@k`.

2. **Log-linear curve of solve rate vs sample count**: take sampling budgets 10 / 100 / 1k / 10k / 100k, plot a `10@k` curve, approximately a straight line after log x; then plot a family with simulated generators of different "quality", observe high-quality models shift left. Expected output: two log-linear fit lines and slopes.

3. **Small behavior-clustering experiment**: generate several correct/wrong solutions to the same problem, execute on K random test inputs, output vectors are signatures; correct solutions' signatures are almost identical (one large cluster), wrong ones scatter. Use `torch.manual_seed(42)` for reproducibility. Expected output: cluster-ID distribution, correct-solution cluster share is clearly larger.

4. **Search-o1-style retrieval-injection loop**: with `llm_client` generate a scripted reasoning chain containing `<|begin_search_query|>...<|end_search_query|>`, parse the query → look up a 20-entry knowledge dict → with a second prompt (Reason-in-Documents) refine the hit document into one sentence → inject as `<|begin_search_result|>` and continue generating. Expected output: retrieval rounds, injected refined text, final answer.

5. **Uncertainty-word detection**: count frequency of "perhaps / wait / likely / not sure" in reasoning output, bar chart reproducing paper figure 1; can sit before demo 4 as visualization of a "knowledge-gap signal". Expected output: bar chart of mean occurrences per word.

## Exercise ideas (3)

1. **Implement sample-test filtering**: tests "verifier" logic. Given a candidate-code list + sample input/output pairs, write `filter_by_example_tests(candidates, tests)` returning code that passes sample tests; after filling, assert: pass rate in the expected range, and filtered-out code indeed mismatches. Hint: first `exec` each candidate for output, then compare sample by sample.

2. **Implement behavior-signature clustering**: tests the "behavior equivalence" concept. Given a "program × test input" output matrix (shape `(num_programs, num_inputs)`, strings or numbers), write `cluster_by_signature(outputs)` grouping programs with the same output vector into a cluster, return by cluster size descending; after filling, assert the correct-solution cluster is largest and cluster count matches. Hint: output tuples can be dict keys directly.

3. **Implement Search-o1's retrieval-trigger state machine**: tests the agentic loop. Given a reasoning-chain text containing `<|begin_search_query|>...<|end_search_query|>`, write `extract_queries(text)` returning queries in order of appearance; then write `inject_results(chain, query, result)` inserting refined results as `<|begin_search_result|>...<|end_search_result|>`. After filling, assert: query count matches special-symbol pairs, wrap symbols are complete after injection. Hint: `str.split` by the special symbols, keep appearance order.

## References

- Competition-Level Code Generation with AlphaCode (arXiv:2203.07814; Science 2022, doi:10.1126/science.abq1158) — founding system of program synthesis as search; original source of the sample-filter-cluster trio.
- AlphaCode 2 Technical Report (Google DeepMind, 2023-12-06, DeepMind official PDF) — evolution of the same system, adding a reranker and pushing solve rate to Codeforces 85th percentile.
- Search-o1: Agentic Search-Enhanced Large Reasoning Models (arXiv:2501.05366) — frame for on-demand retrieval and document refine in long-reasoning models; project page https://search-o1.github.io/, code https://github.com/sunnynexus/Search-o1.
- CodeContests dataset (https://github.com/deepmind/code_contests) — AlphaCode's matching train/eval dataset, strict time split + generated tests.
- QwQ-32B-Preview (arXiv:2412.10903) — Search-o1's reasoning backbone, an o1-class open long-reasoning model.
- ReAct: Synergizing Reasoning and Acting in Language Models (arXiv:2210.03629) — source of agentic RAG's "think then do / decide to call" idea.
- Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (arXiv:2310.11511) — judging retrieval necessity and self-reflection, same lineage as Search-o1.
- CS329A syllabus Lecture 8: Search & Deep Research Agents (https://cs329a.stanford.edu/) — this lecture's place on the course map.
