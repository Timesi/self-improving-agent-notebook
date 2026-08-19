# Lecture 16 — Multimodal robotics agents (study notes)

> This file is the paper study notes for CS329A lecture 16, and the source material for the matching notebook.
> Source: https://cs329a.stanford.edu/ (Autumn 2025 syllabus). Guest: Danny Driess (Physical Intelligence).

## Lecture theme

This lecture answers a core question: an Agent cannot only "think", it must also "move" — when an Agent has a physical body and must complete manipulation tasks in the real world, how do perception, language, and action fuse into one unified model? The answer is a **Vision-Language-Action Model (VLA)**: discretize the robot's continuous actions into tokens, attach them directly after a VLM for next-token prediction, so one end-to-end model can both understand language instructions and image scenes and output end-effector control commands.

This lecture is the "ultimate form of an Agent" in the course: Agents in L1–L15 all act in the **digital world** (tool calls, code, search, reasoning); this lecture puts the Agent loop into the **physical world** — perceive → decide → act → observe feedback, the most complete Agent loop. It also ties back to the two main threads that recur in the course: **scale** (transferring a VLM's web-pretraining knowledge to robots, bridging the magnitude gap of robot data being only hundreds of thousands to a million trajectories) and **learning from the physical world** (teleoperation data collection, closed-loop evaluation on real robots, sim-to-real feedback). The lecture is also placed before L17's future directions, closing "general embodied agents".

## Close reading

### Paper 1: RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (arXiv:2307.15818, rt-2.pdf)

- **Core idea**: robot data is too scarce (13 robots, 17 months, only tens of thousands of trajectories), while the web has billions of image-text tokens of pretraining. Can we, without inventing a new architecture, directly fine-tune an **existing VLM** into a policy that outputs low-level control commands? RT-2's method is unusually simple: **treat actions as "another language"** — discretize continuous actions into tokens, put them in the VLM's vocabulary, and train together with natural language. This model class is called VLA (vision-language-action); RT-2 is its first large-scale instance. Core insight: the model need not learn new actions; what web knowledge gives it is the ability to "put already-learned actions to use in new ways" (semantic understanding, symbol recognition, cross-lingual, elementary reasoning).

- **Key formulas / algorithms**:
  - **Action space**: 7DoF mobile manipulator. Action = 6-D end-effector pose displacement ($\Delta\text{pos}_{x,y,z}$, $\Delta\text{rot}_{x,y,z}$) + gripper opening + 1 terminate command. Except the terminate command, each continuous dimension is **uniformly discretized into 256 bins**, represented by bin index, 8 integers in total.
  - **Action-string format**: `"terminate Δpos_x Δpos_y Δpos_z Δrot_x Δrot_y Δrot_z gripper_extension"`, e.g. `"1 128 91 241 5 101 127"`. Concatenate the action vector into a string; action tokens enter the training set together with natural-language tokens; the model answers in standard VQA format `"Q: what action should the robot take to [task]? A:"`.
  - **Token assignment**: PaLI-X has an independent token for each integer 0–1000, so bin indices bind directly to the corresponding integer tokens; PaLM-E does not have that convenience, so it directly **overwrites the 256 least-used tokens**. Overwriting existing tokens is a form of symbol tuning.
  - **Co-fine-tuning (key detail)**: not fine-tuning on robot data only, but fine-tuning **together** robot-trajectory data and original web vision-language data, and raising the sampling weight of robot data in each batch. This prevents the model from forgetting abstract visual concepts learned from the web, and is the key to generalization gains.
  - **Output constraint**: when decoding robot tasks, mask the vocabulary so that **only legal action tokens may be sampled**; ordinary VQA tasks can still output the full natural-language vocabulary.
  - **Realtime inference**: the 55B model on multi-TPU cloud, queried over the network, 1–3 Hz; the 5B version about 5 Hz.
  - **CoT variant**: extra fine-tune a few hundred steps, augmenting data into `"Instruction: I'm hungry. Plan: pick rxbar chocolate. Action: 1 128 124 136 121 158 111 255."`, so the model first generates a natural-language plan then outputs actions.

- **Key experimental results** (about 6000 real-robot evaluations):
  - **Seen tasks** comparable to RT-1; **generalization (unseen objects / backgrounds / environments)** on average about **2×** RT-1 and MOO, about **6×** VC-1/R3M.
  - **Emergent abilities** in three classes: symbol understanding ("move apple to 3"), reasoning ("move the apple to the cup with same color", math, multilingual), human recognition ("move coke can to the person with glasses"); the best RT-2-PaLI-X mean success is **more than 3×** RT-1.
  - **Ablations** (Figure 6b): training 55B/5B from scratch is very poor (5B from scratch already collapses, 55B not even measured); co-fine-tuning beats robot-data-only fine-tuning; **larger models generalize better**.
  - Language-Table simulation: RT-2-PaLI-3B reaches **90 ± 10**, far above BC-Zero 72±3, RT-1 74±13, LAVA 77±4.
  - Limits: web data does not bring **new actions** (physical skills remain limited to the robot-data distribution); large-model inference is costly.

- **Relation to the lecture theme**: RT-2 is the founding paper of VLA, establishing two principles that run through this lecture: (1) **actions as tokens** — discretize the action space + reuse an existing VLM vocabulary; (2) **web-knowledge transfer** — use scale to patch robot-data scarcity, so the model can generalize to instructions and scenes never seen in training. It is also the starting point of the paradigm "one model as both VLM and policy"; OpenVLA is its open-sourcing and engineering.

- **Demo-ready code points**:
  - Implement 256-bin uniform discretization from scratch and encode/decode of the `"1 128 91 241 5 101 127"` action string.
  - Implement **action-token vocabulary masking** at decode time (output constraint).
  - With `llm_client` demo RT-2's `Instruction → Plan → Action` chain-of-reasoning format and parse out action tokens.

### Paper 2: OpenVLA: An Open-Source Vision-Language-Action Model (arXiv:2406.09246, openvla.pdf)

- **Core idea**: RT-2 has two obstacles to adoption: the model is closed and not open-source, and it did not study how to efficiently fine-tune to new tasks. OpenVLA is the **first open-source general VLA**: 7B parameters, fine-tuned on 970k real-robot trajectories from the Open X-Embodiment dataset, controls several robots out of the box, and systematically studies parameter-efficient fine-tuning (LoRA) and quantized inference, so a VLA can be fine-tuned and served on a consumer GPU. It shows that "small, open-source, fine-tunable" can beat "large and closed": **7× fewer parameters** than RT-2-X (55B) yet **16.5%** higher mean success on 29 tasks.

- **Key formulas / algorithms**:
  - **Architecture**: Prismatic-7B VLM = vision encoder + 2-layer MLP projector + Llama 2 7B language backbone. The vision encoder is a **fused dual encoder**: SigLIP (high-level semantics) + DINOv2 (low-level spatial detail) features concatenated on the channel axis; adding DINOv2 substantially improves spatial reasoning (especially important for robot control).
  - **Action tokenization**: follows RT-2's 256-bin discretization, but bin width is a uniform cut of the **1st–99th percentile** interval of actions on the training data (RT-2 uses min-max, easily stretched by outlier actions, lowering effective resolution). An N-D action → N integers in $[0,255]$.
  - **Token assignment**: the Llama tokenizer reserves only 100 special tokens, not enough for 256 action tokens, so it overwrites the vocabulary's **last 256 least-used tokens** (i.e. last 256 tokens).
  - **Training objective**: standard next-token prediction, cross-entropy **computed only on action tokens**. Small-scale design experiments first on BridgeData V2 (runs fast), then full-data training.
  - **Data**: Open X-Embodiment (70+ sub-datasets, >2M trajectories) filtered: keep only **manipulation** datasets with at least 1 third-person camera and single-arm end-effector control, mixed with Octo's mixing weights; DROID was once added at 10% weight, then removed in the last 1/3 of training because action-token accuracy would not rise. Final 970k trajectories.
  - **Design decisions** (from small-scale experiments): fine-tuning the vision encoder is crucial (freezing it clearly drops performance — pretrained visual features are not fine-grained enough); 224×224 vs 384×384 images have no performance difference (the latter trains 3× slower); LR fixed at 2e-5; training runs **27 epochs** (LLM/VLM usually 1–2), real performance keeps rising until action-token accuracy exceeds 95%.
  - **Infrastructure**: 64 A100s train 14 days (21,500 A100-hours total), batch 2048; bf16 inference occupies 15GB, about 6 Hz on an RTX 4090.

- **Key experimental results**:
  - **Out-of-the-box evaluation** (BridgeData V2 170 rollouts / Google robot 60 rollouts, A/B same conditions): OpenVLA substantially beats RT-2-X (55B) on BridgeData V2, comparable to RT-2-X on the Google robot; both far above RT-1-X (35M) and Octo (93M). RT-2-X is higher only in the "semantic generalization" category (its co-fine-tuning retained more web knowledge; OpenVLA only fine-tunes on robot data).
  - **Credit of data and components**: OpenVLA uses 970k trajectories (vs RT-2-X's 350k), cleaned all-zero actions in the Bridge dataset, uses a fused vision encoder.
  - **Fine-tuning to a new robot** (Franka, 10–150 demonstrations): under full-parameter fine-tuning OpenVLA has the highest aggregate performance, the only method ≥50% on every task; Diffusion Policy is finer on narrow single-instruction tasks, but on multi-object, multi-instruction, language-grounding tasks OpenVLA dominates, **20.4%** higher than Diffusion Policy on average.
  - **LoRA parameter-efficient fine-tuning** (Table 1): LoRA rank=32 needs to fine-tune only **1.4% of parameters** to match full-parameter fine-tuning (68.2% vs 69.7%), done in 10–15 hours on one A100 (1/8 the compute of full-parameter); fine-tuning only the last layer (30.3%) or freezing the vision encoder (47.0%) both fail.
  - **Quantized inference** (Table 2): int4 quantization matches bf16 (71.9% vs 71.3%), memory from 16.8GB down to 7.0GB; int8 is instead slower (quantization overhead).
  - Limits: only supports a single image, no proprioception / history; inference throughput limits high-frequency tasks such as ALOHA at 50Hz; success still <90%.

- **Relation to the lecture theme**: OpenVLA is the **open-sourcing and engineering** close of VLA. It turns RT-2's paradigm into a reproducible, downloadable ecosystem that can be fine-tuned on a consumer GPU, directly echoing the course's "implement by hand + experimentable" goal: in the notebook readers can load OpenVLA (HuggingFace) or implement its core from scratch (action tokenization, mini VLA forward, LoRA). It is also, before Physical Intelligence's later work (π0) where Danny Driess works, the community's most general VLA baseline.

- **Demo-ready code points**:
  - Implement **quantile binning** (OpenVLA-style) from scratch and contrast uniform binning (RT-2-style) sensitivity to outlier actions.
  - In torch build a **mini VLA** (vision patch embed + instruction tokens + a small transformer, predicting 256-class × 8-D action tokens).
  - Implement **LoRA** low-rank fine-tuning from scratch, count trainable-parameter share, contrast full-parameter fine-tuning.

## Teaching thread (how a Stanford instructor might teach this)

1. **Motivation: an Agent's ultimate form has a body.** Open with a failure case: an Agent that can write code and search materials faces a real robot arm and cannot grasp a cup. From this raise "embodied intelligence = an Agent put into the physical world": perceive → decide → act → observe feedback, the most complete Agent loop; tool calling, verification, reasoning from all earlier lectures here become "how to use a body to do things". The instructor's analogy: a language Agent is like pointing on a chessboard; an embodied Agent actually places the piece.

2. **Stall: robot data is too scarce.** Give a number contrast — the web has tens of billions of image-text tokens; the largest robot dataset (Open X-Embodiment) has only about 1 million trajectories. Any policy trained from scratch cannot obtain a web model's semantic common sense. From this raise the core question: can we directly reuse a VLM? Introduce the leap "from VLM to VLA".

3. **Method A: RT-2 — actions are another language.** The core step is **action-space discretization**: a continuous 7-D action (6-D pose displacement + gripper + terminate), 256 bins per dimension, concatenated into the string `"1 128 91 241 5 101 127"`, so action tokens mix among natural-language tokens for next-token prediction. Hand-calculate one continuous action → bin index → token string; readers most easily stall on "why 256? why overwrite vocabulary tokens?". Then **co-fine-tuning**: train robot data + web data together, raise robot sampling weight, prevent forgetting. Finally use emergent abilities (move apple to 3, move to cup with same color, pick rock as hammer) to show that web knowledge really transferred in. Stall warning: readers easily think web data brings new actions; it only brings new semantics — action skills remain limited by the robot-data distribution.

4. **Method B: OpenVLA — open source + efficient fine-tuning.** First point out RT-2's two problems: closed, not fine-tunable. Then show OpenVLA's architecture trade-offs: Prismatic = SigLIP+DINOv2 fused vision + 2-layer MLP + Llama 2 7B; quantile binning is more outlier-resistant than min-max; training 27 epochs before converging. Then "why a small model can beat a large one": more data (970k vs 350k) + cleaner cleaning + fused visual features. Finally land on **deployability**: LoRA trains only 1.4% of parameters to match full-parameter, int4 quantization halves memory without dropping performance — letting VLAs move from the cloud onto consumer GPUs, echoing the course's "reproducible" spirit.

5. **A feedback closed loop of learning from the physical world.** Close by stressing that the whole loop is not "train once and done": teleoperate to collect data → train a VLA → closed-loop evaluate on a real robot → failed samples flow back. Add a future view: Physical Intelligence's π0, where Danny Driess works, uses a flow-matching continuous action head, another route besides "discrete action tokens"; and RT-1 (35M discretized transformer), RT-X (cross-embodiment) form the upstream and downstream of this lineage.

## Code demo ideas (4–6)

1. **Hand-write action-token discretization**: implement RT-2-style uniform binning and OpenVLA-style quantile binning from scratch. With numpy generate a set of 7-D continuous actions, use min-max vs 1–99 percentiles to set bin intervals, compare the two methods' effective resolution after mixing in outlier actions; then concatenate bin indices into `"1 128 91 241 5 101 127"` and write reverse de-tokenize. Expected output: both methods' bin intervals, one action's token string, error of inverting back to a continuous action.

2. **Mini VLA forward (torch from scratch)**: use a tiny transformer-decoder as the "VLM backbone": an image through a small patch embed becomes visual tokens, instruction text through an embedding layer becomes text tokens, concatenated then causally decoded, last layer predicting 256-class action tokens at 8 positions. Train a few steps on synthetic episodes, plot the action-token accuracy rising curve; then at the decode end implement **vocabulary masking** (sample only action tokens). Expected output: loss/acc curves + a sampled action-token string.

3. **Teleoperation data format and batching**: simulate Open X-Embodiment-style episode data (dict or npz: each episode contains a set of images, one language instruction, a set of 7-D actions), demo the whole data pipeline from raw trajectory → discretize → action-token string → batch by (image, instruction, action tokens) (padding + attention mask), count action-token count in one epoch. Expected output: data-volume stats, shape of the first batch, one parsed `Plan/Action` sample.

4. **A small co-fine-tuning anti-forgetting experiment**: on the mini VLA first do one round of "web task" (image classification / caption) pretraining, then contrast two fine-tunes: (a) robot data only; (b) robot data + a little mixed web data (raise robot sampling weight). Measure accuracy on both task classes, intuitively seeing that robot-data-only collapses web-task accuracy, while co-fine-tuning keeps it. Expected output: two "forgetting" curves, reproducing RT-2 Figure 6b's conclusion.

5. **LoRA low-rank fine-tuning from scratch**: add LoRA on the mini VLA's linear layers ($W = W_0 + BA$, low rank $A,B$), train only $B$ and $A$, count trainable-parameter share and contrast full-parameter fine-tuning on a new task in performance and memory / parameter cost. Expected output: a parameter-count contrast table, reproducing OpenVLA's "1.4% of parameters matching full-parameter" order of magnitude.

6. **Plan + Action chain-of-reasoning (llm_client)**: parse RT-2's CoT data format `"Instruction: ... Plan: ... Action: 1 128 ..."`, use `llm_client` (scripted mode gives a scripted trajectory) to let the model first generate a plan then action tokens, demoing "language reasoning bridged to low-level control". Expected output: parsed Plan text and decoded 7-D action, and note that action tokens need vocabulary masking.

## Exercise ideas (3)

1. **Implement action discretization**: examines binning and outlier resistance. Given a continuous-action array and bin count 256, write `quantile_binning(actions, lo=1, hi=99)` returning each dimension's bin interval, and use it to implement `discretize(a, bins) -> int` and `detokenize(i, bins) -> float`; after filling in, assert: reverse error is within bin width, and after mixing in outlier actions the quantile-bin interval width is substantially smaller than min-max. Hint: take percentiles by dimension then cut; clip out-of-range values.

2. **Implement action-token string encode/decode (RT-2 format)**: examines sequence format. Given 7-D bin indices (including a terminate bit), write `to_action_string(ids) -> str` and `parse_action_string(s) -> list[int]`, so `"1 128 91 241 5 101 127"` holds both ways; after filling in, assert: the string is space-separated, length matches dimension count, roundtrip indices are the same. Hint: `str.join` and `str.split`; put the terminate bit first.

3. **Implement action-token vocabulary masking at decode time**: examines output constraint. Given full-vocabulary logits (shape `(V,)`) and an action-token set `action_ids`, write `mask_for_robot_task(logits, action_ids)` returning a log-probability distribution that keeps only action tokens; after filling in, assert: after masking other positions' logits are `-inf`, after softmax non-action-token probability is 0. Hint: fill `-inf` with `torch.full_like` then `scatter` the action positions back.

## References

- RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (arXiv:2307.15818; https://robotics-transformer2.github.io) — founding VLA work: actions as tokens + co-fine-tuning + web-knowledge transfer.
- OpenVLA: An Open-Source Vision-Language-Action Model (arXiv:2406.09246; https://openvla.github.io) — the first open-source general VLA, a systematic study of LoRA parameter-efficient fine-tuning and quantization.
- RT-1: Robotics Transformer for Real-World Control at Scale (arXiv:2212.06817) — 35M-parameter discretized-action transformer, RT-2's foundation and data source.
- Open X-Embodiment: Robotic Learning Datasets and RT-X Models (arXiv:2310.08864; https://robotics-transformer-x.github.io) — 70+ sub-datasets, cross-embodiment RT-1-X / RT-2-X, OpenVLA's training-data source.
- PaLM-E: An Embodied Multimodal Language Model (arXiv:2303.03378) — the VLM backbone of RT-2-PaLM-E, an early representative of embodied multimodal language models.
- π0: A Vision-Language-Action Flow Model (Physical Intelligence, 2024) — work from this lecture's guest's team: replacing discrete tokens with a flow-matching continuous action head, the other VLA route.
- Prismatic VLMs (Karamcheti et al., 2024, https://prismatic-vlms.github.io) — OpenVLA's VLM backbone, SigLIP+DINOv2 fused vision encoder.
- LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685) — the low-rank adaptation method OpenVLA's parameter-efficient fine-tuning depends on.
- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (arXiv:2303.04137) — the from-scratch learning baseline OpenVLA's fine-tuning experiments compare against.
- CS329A syllabus Lecture 16: Multimodal AI Agents in Robotics (https://cs329a.stanford.edu/) — this lecture's place on the course map.
