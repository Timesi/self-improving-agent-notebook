# Lecture 12 — LLM Reasoning (study notes)

> This file is the paper study notes for CS329A lecture 12, and the source material for the matching notebook.
> Guest: Denny Zhou (Google DeepMind, co-author of all three papers).
> Source: https://cs329a.stanford.edu/ (Autumn 2025 syllabus)

## Lecture theme

The core question this lecture answers: **where does an LLM's "reasoning ability" come from, and how can we elicit it without changing the model or fine-tuning?**

The three papers give a self-consistent chain of answers:
1. **The ability exists**: at sufficient scale, large models "emerge" with reasoning abilities that small models lack (emergent abilities).
2. **The ability can be elicited**: a tiny change to the few-shot prompt format (replacing exemplars from ⟨question, answer⟩ with ⟨question, reasoning steps, answer⟩) can "hook" that ability out with prompts (chain-of-thought prompting).
3. **The ability can be aggregated**: a single greedy decode walks one reasoning path and easily lands in a local optimum; switching to "sample many times + majority vote on the final answer" further improves via "self-ensemble" (self-consistency), and incidentally yields an uncertainty estimate.

Why it sits here in an Agent course: everything about agents rests on the model's abilities — multi-step task decomposition, generating intermediate reasoning steps, judging confidence in one's own answer ("know when it doesn't know"). This lecture lays the theoretical and intuitive foundation of the "reasoning engine" for later agent construction (planning, tool use, reflection, self-correction). The guest (Denny Zhou) also strings reasoning together with program synthesis, Least-to-most, Tree of Thoughts (ToT), and later work.

## Close reading

### Paper 1: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (arxiv:2201.11903, cot.pdf)

- **Core idea**: when giving a large model few-shot exemplars, change the "input → output" pair into an "input → **intermediate reasoning steps** (chain of thought) → output" triple, and the model can solve multi-step reasoning tasks it previously could not. The core claim has three layers: (1) intermediate steps let the model decompose a multi-step problem, allocating "computation" to harder questions; (2) reasoning steps provide an interpretability window, making it easier to locate which step went wrong; (3) **this is the prompt format's doing, not training** — no model is fine-tuned at any point. The paper calls standard prompting "only a lower bound on model capability".

- **Key formulas / algorithms**: the core is a prompt format, with no training objective. The format is `Q: <problem>\nA: <step-by-step reasoning…>. The answer is <answer>.`; greedy decoding. Arithmetic tasks share 8 hand-written CoT exemplars (AQuA, being multiple-choice, uses 4); common-sense tasks use 4–7 each; symbolic tasks each have a hand demonstration. Evaluation covers 5 model families: GPT-3 (text-ada/babbage/curie/davinci-002 ≈ 350M/1.3B/6.7B/175B), LaMDA (0.4B–137B), PaLM (8B/62B/540B), UL2-20B, Codex. Three benchmark classes: arithmetic (GSM8K, SVAMP, ASDiv, AQuA, MAWPS), common sense (CSQA, StrategyQA, Date/Sports Understanding, SayCan), symbolic (last letter concatenation, coin flip).

- **Key experimental results**:
  - **GSM8K rises substantially**: PaLM 540B + 8 CoT exemplars reaches 57% solve rate, far above standard prompting (18%) and fine-tuned GPT-3 175B + verifier (33%), resetting then-SOTA (Figure 2: 33 / 55 / 18 / 57).
  - **CoT is an emergent ability at scale**: small models (<~100B) not only gain nothing, they often produce "fluent but illogical" reasoning chains, performing worse than standard prompting; clear gains appear only around ~100B. Harder problems gain more (GSM8K more than doubles; single-step SingleOp has no gain).
  - **Ablations isolate "natural-language intermediate steps" as the key**: equation only, a string of dots simulating "variable compute", and putting reasoning after the answer, all match the baseline → the gain comes from "natural-language steps as sequential reasoning", not equations, not compute, not knowledge activation.
  - **Robustness**: across annotators (A/B/C), exemplar sets (including GSM8K training-set samples), order, and count, CoT always substantially beats the baseline (though prompt engineering still introduces variance, e.g. coin flip from 99.6% to 71.4%).
  - **Symbolic reasoning + length extrapolation (OOD)**: PaLM 540B is near 100% in-domain; standard prompting fails completely on longer sequences, while CoT makes length generalization possible (2-word names → 4 words, 2 coin flips → 4).
  - **Error analysis**: of 50 correct samples, 48 have logically correct reasoning chains; of 50 error samples, 46% are "almost correct" (calculator / symbol mapping / missing a step), 54% are semantic-understanding or coherence problems; scaling PaLM from 62B to 540B mainly fixes "missing a step" (12 of 18) and semantic-understanding errors (6 of 20).

- **Relation to the lecture theme**: this paper is the anchor of lecture 12 — it shows that "reasoning can be elicited by prompting, and only in a sufficiently large model", welding "emergence" and "prompting" together. It is also the direct foundation of later self-consistency (the paper itself names majority vote as a direction for improvement).

- **Demo-ready code points**: hand-write standard vs CoT few-shot templates and contrast outputs; implement parsing of "The answer is X."; reproduce the three ablations (equation only / dots / answer-first); with rules or a small model, reproduce last-letter and coin-flip toy tasks and OOD length extrapolation.

### Paper 2: Self-Consistency Improves Chain of Thought Reasoning (arxiv:2203.11171, self-consistency.pdf)

- **Core idea**: CoT's greedy decode walks one path and easily falls into a local optimum or a single-step error. Self-consistency replaces decoding with "**sample-and-marginalize**": first sample a set of **diverse reasoning paths**, then **majority-vote** their respective final answers. Intuition: a hard reasoning problem usually has many paths to a unique correct answer; correct paths, even when diverse, "converge" on the final answer, while incorrect paths rarely coincide on the same answer. This resembles human experience — confidence is higher when several lines of thought yield the same answer. The method is fully unsupervised, needs no verifier / reranker / fine-tuning, and is a "self-ensemble" on a single model.

- **Key formulas / algorithms**: three steps: (1) CoT prompting; (2) sample m paths from the decoder (temperature + top-k); (3) take argmax majority vote over the final-answer set `argmax_a Σ_{i=1..m} 𝟙(a_i = a)`. Optional weighting: length-normalized conditional log-probability `P(r_i,a_i|prompt,q) = exp((1/K)·Σ_k log P(t_k | …))`. **Key finding: unweighted majority vote ≈ normalized weighted sum (74.4 vs 74.1), and both clearly beat unnormalized weighting** — because the model's path probabilities are similar (the model is poorly calibrated), so simple majority vote is enough. Sampling config: UL2/LaMDA T=0.5, top-k=40; PaLM T=0.7, top-k=40; GPT-3 T=0.7 no top-k; default sample 40 paths, average over 10 runs.

- **Key experimental results**:
  - **Broad large gains**: on PaLM-540B, GSM8K 56.5→74.4 (**+17.9%**), SVAMP +7.6%, AQuA +12.5%, StrategyQA 75.3→81.6 (+6.4%), ARC-c 85.2→88.7 (+3.9%). Summary numbers: GSM8K +17.9%, SVAMP +11.0%, AQuA +12.2%, StrategyQA +6.4%, ARC-c +3.9%.
  - **Gains grow with scale**: UL2-20B only +3–6%, LaMDA-137B and GPT-3 reach +9–23%; still significant on already-strong models (AQuA/GSM8K +12–18%). Beats supervised SOTA on almost every task, fully unsupervised.
  - **Beats other decode / ensemble methods**: at the same sample count, significantly beats sample-and-rank; beats beam search (beam diversity is low, so it is worse); beats prompt-shuffle ensembles (40 permutations 19.2, 3 prompt sets 18.6 vs self-consistency 27.7, LaMDA GSM8K); multi-model ensembles lose to single-model self-consistency (PaLM 540B self-consistency 74.4 vs ensemble max 36.9).
  - **Robust and general**: robust to sampling strategy / hyperparameters (T, top-k, nucleus p); can repair imperfect prompts (17.1→14.9, +SC→23.4), equation prompts, and zero-shot CoT (PaLM zero-shot 43.0→69.2, +26.2%); on ordinary NLP tasks where CoT is harmful, it can still overtake standard prompting.
  - **Consistency is uncertainty**: inter-sample consistency (max vote share) is highly correlated with accuracy → it can serve as a "know when it doesn't know" confidence signal (Figure 5).
  - **Path-count vs accuracy saturation curve**: more paths are better but saturate quickly; in practice 5–10 paths capture most of the gain.

- **Relation to the lecture theme**: the direct sequel to CoT (also Denny Zhou's work), advancing "reasoning" from a one-shot output to "multi-path decision + aggregation", and incidentally giving agents the **uncertainty estimate** and calibration view they need. It shows that "without more data/training, a small change on the decode side" can amplify reasoning — a canonical example of "reasoning engineering" via prompt/decode tricks.

- **Demo-ready code points**: numpy majority vote and edit-distance (for free-text answers) self-consistency aggregation; reproduce Table 1's four aggregation strategies; a toy probability model of "incorrect paths do not converge"; plot the "path count vs accuracy" saturation curve; plot "consistency vs accuracy" to reproduce the uncertainty signal.

### Paper 3: Emergent Abilities of Large Language Models (arxiv:2206.07682, emergent-abilities.pdf)

- **Core idea**: a survey / conceptual paper (Jason Wei et al., with Denny Zhou) that formalizes "emergent abilities": **an ability is emergent if small models do not have it, large models do, and it cannot be predicted by extrapolating small-model performance**. The definition is rooted in Philip Anderson's 1972 "More Is Different": a **quantitative change** in a system leads to a **qualitative change in behavior**. Drawn as a scaling curve (x-axis training FLOPs), emergence looks like: performance hugs chance at small scale (flat), then after a critical scale **jumps** well above chance — a "phase-transition" shape. The paper stresses that emergence is "a function of many related variables" (compute, parameter count, data, even WikiText103 perplexity); the critical scale is not an intrinsic property of the ability, and changes with data quality, architecture, and training method.

- **Key formulas / algorithms**: no method; mainly a **taxonomy + evidence summary**. Two emergence settings: (a) few-shot prompting ability itself (Figure 2's 8 examples); (b) **augmented prompting / fine-tuning tricks that only work on large models** (Figure 3) — if a trick is ineffective or even harmful on small models and only starts working at sufficient scale, it is also an emergent ability. Mechanism intuition given: multi-step reasoning needs sequential computation of O(l) layers of depth; closed-book QA needs enough parameters to "compress a knowledge base".

- **Key experimental results**:
  - Emergent few-shot tasks (Figure 2): modular arithmetic (GPT-3 jumps above chance at 2×10²² FLOPs/13B, LaMDA at 10²³/68B), IPA transcription, letter rearrangement, Persian QA, TruthfulQA (Gopher 280B/5×10²³ FLOPs jumps 20%+ above chance), grounded mappings, MMLU (all ≤~10B near chance, 70B–280B significantly above chance), WiC.
  - Emergent augmented tricks (Figure 3 + Table 1): **CoT at ~10²³ FLOPs (~100B)**; instruction-tuning FLAN at 68B; **scratchpad (8-bit addition) emerges at only 40M parameters / 8.9×10¹⁹ FLOPs** (the threshold can be low); P(True) calibration at 52B. Table 1 explicitly lists **self-consistency as an emergent trick** (1.3×10²³ FLOPs / 68B LaMDA), closing the loop of the three papers.
  - **Historical case WiC**: GPT-3/Chinchilla stay at chance even at max scale; Brown et al. blamed architecture / the autoregressive objective; then PaLM 540B (2.5×10²⁴ FLOPs) broke through by continuing to scale — a "negative result" may just mean scale has not arrived.
  - **Emergence ≠ only scale**: Sanh et al. used encoder-decoder T0 to get instruction following at 11B; InstructGPT used RLHF so that 1.3B beat larger models; PaLM 62B beat chance on 14 BIG-Bench tasks where LaMDA/GPT-3 stayed near chance to the max (data quality and architecture also matter). **An ability can appear first on a small model, then be "unlocked" to smaller scale as training methods improve**.
  - **A self-critique of metrics (§5.1)**: exact match / scoring only the final answer can hide incremental local improvement, disguising smooth progress as emergence; but analyzing six emergent tasks with cross-entropy loss, CE does steadily fall at small scale (Outcome 2), and classification tasks also emerge, and switching to partial-credit metrics such as BLEU/ROUGE still shows an emergence shape → metrics can only partly explain, not fully attribute, emergence.
  - **BIG-Bench keyword analysis**: the highest emergence share is analogical reasoning, word-sense disambiguation, truthfulness, social reasoning, emotion understanding; math/arithmetic share is actually low; vision, non-language, and context-length classes are mostly flat curves. There is no clear regularity of "which tasks will emerge".

- **Relation to the lecture theme**: it supplies the conceptual umbrella for the whole lecture: the CoT paper's "CoT is scale emergence" and the self-consistency paper's "self-consistent decoding is also an emergent trick" both hang on it. It also introduces an honest reflection on **ability vs metric**, the key text for a teaching "critical turn". We can contrast Yi et al. (2204.07646, "Are Emergent Abilities of Large Language Models a Mirage?"): Yi et al. redraw the same family of curves with per-token accuracy, Brier score, partial credit, and other continuous/smooth metrics, find the curves become smooth and predictable, and argue that "emergence" is mainly an artifact of **discontinuous metrics (exact match, acc)**; this paper replies with CE analysis and emergence on classification tasks against "purely a metric". The essential disagreement: whether the underlying ability improves smoothly (Yi) or there is a genuine qualitative threshold (Wei).

- **Demo-ready code points**: visualize "metrics manufacturing emergence" — take a smooth underlying-ability curve and measure it with exact match vs per-token accuracy; on a multi-step task, reshape a smooth per-step correctness p into a compounding final accuracy p^L, showing a jump; reproduce Yi et al.'s core claim on synthetic data.

## Teaching thread (how a Stanford instructor might teach this)

1. **A failure case sets motivation**: first show a GSM8K multi-step word problem failing under standard few-shot prompting — the model gives a wrong answer directly, and no matter how the model is scaled, the scaling curve stays nearly flat. Raise the problem: compute went up, reasoning did not.
2. **CoT arrives**: only change exemplars from ⟨Q,A⟩ to ⟨Q, reasoning steps, A⟩, and PaLM 540B's GSM8K goes from 18%→57%, beating fine-tuned + verifier GPT-3. Emphasize two surprises: no training is needed; it only works above ~100B. Use Figure 1's contrast to make the prompt format clear, and have students hand-write one exemplar.
3. **Ablations build intuition**: strip CoT's ingredients one by one — equation only (not "can compute equations"), dots only (not variable compute), reasoning after the answer (not knowledge activation) → what remains is "sequential reasoning in natural language" itself. Then error analysis (what 62B→540B fixed) builds intuition for "what scale buys the model".
4. **Rise to the emergence frame**: introduce the emergent-abilities paper, definition + phase-transition curve + BIG-Bench/MMLU examples, and put CoT/self-consistency into Table 1's list of emergent tricks.
5. **Critical turn (where students stall / want to object)**: pause to discuss whether these "jumps" are only an artifact of exact-match metrics. Introduce Yi et al.'s "mirage" argument, and draw two curves live: the same underlying smooth ability, exact match showing a jump, per-token accuracy showing smoothness. Clarify that "metric honesty" is a necessary discipline of reasoning research (echoing the paper's §5.1 self-critique).
6. **Self-consistency closes**: back to the decode side — greedy has one path, easily a local optimum. Intuition analogy: let several "thinkers" each solve then vote; the correct ones converge, the incorrect ones diverge. Show sample-and-marginalize, aggregation-strategy contrast (majority vote ≈ normalized weighting), the path-count vs accuracy saturation curve, and consistency–accuracy correlation (uncertainty signal).
7. **One sentence**: reasoning ability = pretraining gives potential + scale triggers emergence + prompt format elicits + decode aggregation amplifies + metric-honest verification. Those five words preview four things later agents need: prompt design, multi-path thinking, confidence estimation, verifiable evaluation.

## Code demo ideas (3–6)

1. **Hand-write few-shot CoT prompt templates**: with a small model or an OpenAI-compatible API, on 5–10 GSM8K-style math problems, construct standard (⟨Q,A⟩) and CoT (⟨Q, steps, A⟩) prompts (8 exemplars each, copy the paper format `…The answer is X.`). Key code: template strings + answer regex parse + per-problem output comparison table. Expected output: CoT has clearly higher accuracy, and we can print the model's generated intermediate steps to observe the "reasoning chain".

2. **Numpy majority vote and edit-distance self-consistency aggregation**: given N "path+answer" samples for one problem (from an API or pre-stored text), implement four aggregations: majority vote, normalized log-probability weighting, unnormalized weighting, and the centroid answer after clustering by edit distance. Contrast the four strategies' accuracy on 100 problems, reproducing Table 1's intuition (majority vote ≈ normalized weighting > unnormalized). Expected output: a bar chart of aggregation strategy × accuracy.

3. **A toy probability model of "incorrect paths rarely converge"**: let a single path give the correct answer with p=0.5–0.9, and with (1-p) throw uniformly at k wrong answers; simulate voting accuracy for K=1..40 paths with numpy. Plot a heatmap of "voting accuracy vs path count × p". Expected output: as long as p>1/k, voting accuracy rapidly approaches 1 with K — making self-consistency's success attribution clear.

4. **Metric-view visualization of emergence curves (reproducing Yi et al.)**: construct a synthetic model whose "underlying per-token accuracy rises smoothly with scale", and draw curves with (a) exact match / scoring only the final answer, (b) per-token accuracy, (c) multi-step final accuracy = p^L. Expected output: the same underlying ability looks like a "phase transition / emergence" under exact match and smooth under per-token accuracy, intuitively showing that "emergence may be manufactured by the metric", and discussing the stance difference from Wei's CE analysis.

5. **Symbolic reasoning + length extrapolation**: implement last-letter concatenation and coin flip as toy tasks; few-shot exemplars contain only 2-word names / 2 flips; test OOD samples with 3–4 words / 4 flips; contrast "pseudo CoT" (rules simulating step-by-step output) with giving the answer directly. Expected output: direct answers are nearly all wrong OOD; step-by-step correctness rises interpretably with scale/rules.

6. **Self-consistency as uncertainty estimation**: for a batch of problems, sample N=20 paths each, compute consistency (max vote share), and draw a binned scatter of "consistency vs whether that problem is correct". Expected output: a clear positive correlation (reproducing Figure 5), and a simple strategy of using it as a confidence threshold to "refuse to answer" — connecting directly to an agent's "know when it doesn't know".

## Exercise ideas (3)

1. **Hand-write CoT and ablate**: given 8 GSM8K training samples, students write CoT annotations themselves, then run three prompts (standard / CoT / equation-only) on 30 problems each. Fill-in: complete the exemplar template string, complete the `parse_answer(text)` regex; assert: CoT accuracy > standard and > equation-only, and write one sentence explaining why equation-only fails on GSM8K.

2. **Self-consistency sampling and aggregation**: on the same problem, temperature-sample K=20 paths, complete `majority_vote(answers)` and `normalized_logprob_weighted` (implement the paper's equation (1) length normalization); answer: why is unnormalized weighting worse than majority vote? Assert: majority vote ≥ greedy-decode correctness, and 40 paths ≥ 5 paths in accuracy (does not fall).

3. **Testability of emergence metrics**: given synthetic scaling data (underlying ability grows smoothly), complete two metric functions (exact-match and per-token accuracy) and plot; write a simple "emergence detector" (e.g. mean-slope change on either side of a threshold) that judges whether a given curve "emerges". Assert: the exact-match curve is judged emergent, the per-token curve smooth; then in 1–2 sentences reflect: how the paper's CE analysis (Outcome 2) replies to "purely a metric".

## References

- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (arxiv:2201.11903) — this lecture's anchor: few-shot CoT prompt format and the emergence scale
- Self-Consistency Improves Chain of Thought Reasoning (arxiv:2203.11171) — sample-and-marginalize decoding: sampling + majority vote
- Emergent Abilities of Large Language Models (arxiv:2206.07682) — definition, taxonomy, and evidence summary of emergent abilities
- Are Emergent Abilities of Large Language Models a Mirage? (Yi et al., arxiv:2204.07646) — the opposing view: an "emergence mirage" manufactured by discontinuous metrics, to contrast with this lecture's third paper
- Training Verifiers to Solve Math Word Problems (Cobbe et al., arxiv:2110.14168) — GSM8K dataset and the "fine-tune + verifier" baseline
- Large Language Models are Zero-Shot Reasoners (Kojima et al., arxiv:2205.11916) — zero-shot CoT: "Let's think step by step" (the self-consistency paper also verifies compatibility)
- Least-to-Most Prompting Enables Complex Reasoning (Zhou et al., arxiv:2205.10625) — CoT's advanced decomposition strategy (also Denny Zhou's team)
- Show Your Work: Scratchpads for Intermediate Computation (Nye et al., arxiv:2112.00114) — intermediate-computation prediction, emerges at 40M parameters, supporting the emergence paper's evidence
- Finetuned Language Models are Zero-Shot Learners / FLAN (Wei et al., arxiv:2109.01652) — emergence of instruction tuning (Figure 3B)
- BIG-bench: Beyond the Imitation Game (arxiv:2206.04615) — the main source of emergence evidence (200+ tasks)
