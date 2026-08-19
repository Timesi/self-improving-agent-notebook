# AlphaProof / AlphaGeometry 2 — DeepMind blog notes

Source: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/
("AI achieves silver-medal standard solving International Mathematical Olympiad problems", 2024-07-25)

## AlphaProof: the formal language Lean + reinforcement learning

- A self-training system that proves mathematical statements in Lean, extending AlphaZero-style RL from board games to proof search
- Benefit of a formal language: the reasoning can be machine-checked line by line (the verifier is reliable)
- Bottleneck: humans write very little formal data. The fix: finetune Gemini to auto-translate natural-language problems into formal statements,
  and build a large formal problem bank of about one million problems of varying difficulty
- Training loop: about 1 million informal problems → a formalizer network translates them into a formal language → a solver network searches Lean for a proof or a counterexample →
  AlphaZero gradually self-improves, with problem difficulty increasing. During the contest the system keeps running, generating variants of the problems and proving them for further RL

## AlphaGeometry 2: neural–symbolic hybrid

- A large upgrade of the previous AlphaGeometry: the language model is trained from scratch; synthetic data grows by an order of magnitude; the symbolic engine is two orders of magnitude faster
- Knowledge-sharing: combining different search trees to solve harder problems
- Handles moving objects and equations over angles / ratios / distances
- Before the contest it could solve 83% of IMO geometry problems from the past 25 years (previous generation: 53%)

## IMO 2024 results

- 4 of 6 problems solved (AlphaProof: 2 algebra + 1 number theory; AlphaGeometry 2: 1 geometry); 2 combinatorics problems unsolved
- 7 points per problem, 28 points total = high silver; gold threshold 29 (58 of 609 contestants)
- The hardest problem was solved by only 5 human contestants; AlphaProof solved it as well
- Scores were judged under IMO rules by Fields medalist Timothy Gowers and others
- Solve time: from a few minutes up to three days (humans have two 4.5-hour sessions)

## Link to earlier work

- AlphaProof reuses the AlphaZero RL paradigm: search possible proof steps → each verified proof strengthens the language model
- Self-play iterative improvement is the core trait of the AlphaZero family

## Teaching points (for lecture 13)

- The verifier here is formal and reliable — a Lean checker, not LLM-as-judge
- Search + reinforcement learning are the two main threads
- Neural–symbolic combination: the neural net supplies candidate steps; the symbolic engine supplies deterministic reasoning
- Continues lectures 02/03 (test-time compute, robust verification)
