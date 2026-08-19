# AlphaGeometry — GitHub README notes

Source: https://github.com/google-deepmind/alphageometry
The paper appeared in Nature (Trinh, Wu, Le, He, Luong, 2024, DOI: 10.1038/s41586-023-06747-5). There is no arXiv version.

## System architecture (neural–symbolic)

- **DDAR**: a symbolic engine that combines a Deductive Database (DD, Horn-clause deduction rules) with Algebraic Reasoning (AR, equations over angles / ratios / distances).
- **AlphaGeometry** = DDAR + a language model. The LM is a 150M-parameter decoder-only transformer
  (JAX + beam search). When DDAR stalls, it proposes auxiliary constructions.
- Auxiliary constructions are the hard part of contest geometry. They are equivalent to "generating extra terms".

## Synthetic data

- About 1 billion random theorem premises are sampled. The symbolic engine produces 100 million synthetic theorems and proofs (many proofs are >200 steps, 4× a typical contest proof).
- The LM is pretrained on the synthetic data, then finetuned on about 9 million proofs that need auxiliary constructions (about 9%).

## Results

| Solver | IMO-AG-30 | jgex_ag_231 |
|:---|:---|:---|
| DDAR alone | 14 | 198 |
| AlphaGeometry | 25 | 228 |

- IMO-AG-30: 25/30, above the previous best Wu method (10/30), near average gold-medal level.
- Outputs human-readable proofs; solves all geometry problems from IMO 2000 and 2015; finds a generalization of an IMO 2004 theorem (identifies unused premises).

## Reproduction parameters and hardware

- Paper results: `BATCH_SIZE=32, BEAM_SIZE=512, DEPTH=16`
- Hardware: 4 V100 GPUs + 250 CPU workers (to meet the IMO time limit)
- The public code drops some optimizations for clarity (parallel GPU inference, multi-CPU DDAR, LM/DDAR parallelism)

## Teaching points (for lecture 13)

- Neural–symbolic combination: the LM supplies candidate steps (auxiliary constructions); the symbolic engine supplies deterministic verification and search
- Continues lecture 03 on answer verification: here the verifier is formal and reliable
- Contrast with AlphaProof (Lean + RL): geometry uses a symbolic engine; number theory / algebra uses formal proof
