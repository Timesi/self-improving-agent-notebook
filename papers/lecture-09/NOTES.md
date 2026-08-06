# Lecture 09 — 后训练演进：从 Chatbot 到 Agent 研读笔记

> 本文件是 CS329A 第 9 讲（嘉宾 Melvin Johnson，Google DeepMind）"Evolution of Post-training from Chatbots to Agents" 的研读笔记，是编写对应 notebook 的素材。
> 来源：https://cs329a.stanford.edu/（Autumn 2025 课程大纲）
> 说明：本讲不指定论文，是"主题精读"而非"论文精读"。素材来自 InstructGPT、RLHF、Constitutional AI、DeepSeek-R1、RLEF、WebRL、WebShop 等一组代表性工作 + WebSearch 补充的 Agent 化后训练脉络。

## 课程主题

这一讲要回答的核心问题：**预训练只是给了模型"续写文本"的能力，它如何变成"听人话的助手"，再变成"会用工具完成任务的 Agent"？这条路上每一步改的是什么？** 后训练（post-training）指预训练结束之后、为了让模型服务于具体目标而做的所有额外训练：监督微调（SFT）、从人类反馈学习的强化学习（RLHF）、基于可验证奖励的强化学习（RLVR）、以及直接在最细粒度的 Agent 轨迹上做强化学习。

它在课程里的位置是 Part 2（Training & Evolution，L6-L9）的**收尾总纲**。前三讲各自给出一块拼图：L6 的训练期缩放给出了 RL 的算法引擎（STaR 的自举、GRPO 的组内 advantage、DAPO 的工程）；L7 的开放进化展示"Agent 自己改进自己"的目标形态；L8 的搜索与深度研究展示一个具体的 Agent 形态。这一讲把前八讲的零散方法收拢成一条**时间线**，解释"信号来源"如何从人类标注一路迁移到环境执行结果——这正是"自改进"得以成立的前提：**奖励不再依赖人，而来自任务本身是否被验证成功**。后面 Part 3 的 L10（SWE 智能体）、L11（记忆）都是这条时间线在工程端的落地。

选 Melvin Johnson（Google DeepMind）做嘉宾有双重意义：其一，Gemini 是 Agent 化后训练最激进的前沿模型之一，他的视角是"产业里真实的后训练管线长什么样"；其二，DeepMind 自己的 Agent RL 工作（里程碑奖励、子目标规划）正好补齐"轨迹级稀疏奖励"这一段。

整讲的叙事是一个**信号来源的迁移**：
- SFT 用**人类写的行为示范**当信号（学"怎么做"）；
- RLHF 用**人类偏好比较**当信号（学"什么是好"）；
- RLVR 用**可验证的答案正确性**当信号（学"什么是正确"）；
- Agent 后训练用**环境执行结果**当信号（学"什么动作序列能完成任务"）。

每一次迁移都因为前一种信号太贵、太稀疏或太容易被"骗"（reward hacking），而 Agent 时代把这条链推到了尽头：奖励直接来自环境。

## 主题精读（按演进阶段展开）

### 阶段 0：起点 — 预训练模型"不会说话"

预训练模型是在"互联网文本续写"上训练的：给定上文，预测下一个 token。它学会了语法、知识和世界模型，但不会听指令（指令被当成文本的一部分续写下去）、不会停止（不会说"我不知道"）、不会使用工具。这个阶段没有"后训练"这个动作，只有 base model。

- **核心目标**：让模型学会"对话"这种交互格式。
- **信号来源**：无（还没有对齐信号）。
- **产物**：GPT-3 / PaLM / 各开源 base model。
- **为什么需要后训练**：base model 的能力很强，但"方向"是错的——它优化的是下一个 token 的似然，不是用户意图。

### 阶段 1：SFT 时代（2021-2022）— 模仿人类示范

**核心目标**：让模型学会"按指令输出"这一行为本身。做法是收集人工书写的 (指令, 期望回复) 示范，直接做监督微调（SFT）。FLAN（arXiv:2109.01652）用文本到文本的任务并集做 instruction tuning，证明"把行为模板教给模型"能泛化到新指令。

**方法**：SFT 是标准的监督学习，损失就是输出序列的交叉熵
$$L_{\mathrm{SFT}}(\theta) = -\mathbb{E}_{(x, y_{\mathrm{demo}})\sim D}\left[ \sum_t \log \pi_\theta(y_{\mathrm{demo},t} \mid x, y_{\mathrm{demo},<t}) \right].$$
梯度只把"人类示范过的行为"的概率推高。**InstructGPT**（arXiv:2203.02155，OpenAI）是这一步的标杆：约 40 名标注员写 ~13K 条示范，微调 16 个 epoch，得到 SFT 模型。**RLEF 的表 3a 里 "SFT only improves on valid set" 这类结论说明 SFT 的边界：它只能学到数据里出现过的行为，无法超越示教者的水平。**

**信号来源**：人类撰写的示范文本（行为级的模仿信号）。
**产物**：InstructGPT 的 SFT 模型、FLAN 系列。**局限**：(a) 人工示范贵且不统一；(b) 模型只能模仿，"不会想"——遇到示范之外的情况就退化；(c) 无法表达"有些行为更好"的偏好强度。

### 阶段 2：RLHF 时代（2022-2023）— 对齐人类偏好

**核心目标**：不只学"怎么做"，而是学"什么是好"。做法是让模型先输出多个候选，人类给它们排序，训练一个**奖励模型（RM）**，再用 RL 优化这个 RM。**RLHF 的思想源头是 Christiano et al.（arXiv:1706.03741，2017）"Deep Reinforcement Learning from Human Preferences"**，把奖励工程换成"偏好学习"；InstructGPT 把它工业化成三段式管线，ChatGPT（GPT-3.5）就是这条管线的直接产物。

**方法**（InstructGPT 的三段式）：
1. **SFT**：人类示范微调，得到初始策略 $\pi_{\mathrm{SFT}}$。
2. **RM**：对同一 prompt 生成 $K=4\text{-}9$ 个回复，标注员排序，用成对排序损失训练 RM：
$$L_{\mathrm{RM}}(\theta) = -\frac{1}{\binom{K}{2}}\mathbb{E}\left[\log \sigma\!\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right],$$
其中 $y_w$ 是被偏好的一方。InstructGPT 用 6B RM（175B RM 训练不稳定）。数据量 ~33K prompt。
3. **PPO**：最大化 RM 分数，同时用 KL 约束不偏离 SFT 模型太远；PPO-ptx 变体还混入预训练梯度 $\gamma \mathbb{E}[\log \pi_\theta(x)]$，缓解"alignment tax"（对齐导致公共基准倒退）。

**关键数字**：1.3B InstructGPT 在人类评估中比 175B GPT-3 更受偏好（**100 倍参数差被打平**）；175B InstructGPT vs 175B GPT-3 偏好率 85%±3%；闭域任务幻觉率从 41% 降到 21%。

**信号来源**：人类偏好比较（标了序的候选对）。**代价**：RLHF 的 RM 是从几千条偏好里学出来的**代理目标**，会 Goodhart——模型可以骗过 RM 拿到高分（"reward hacking"）。Constitutional AI（arXiv:2212.08073，L4 精读）正是想压缩这个成本：用一条原则清单让 AI 自己批评、修订、打分，把人工标注从几万条压到一份审阅。**局限**：RM 分数是标量标度、人对长回复的排序噪声大、"有帮助 vs 无害"互相打架，且 RM 只衡量"这段文本好不好"，衡量不了"这个动作序列能不能完成任务"。

### 阶段 3：RLVR / 推理时代（2024-2025）— 可验证奖励

**核心目标**：去掉学习式 RM，改用**规则化的、可验证的奖励**——答案与标准答案是否一致、代码是否通过测试。RLVR = Reinforcement Learning with Verifiable Rewards。**DeepSeek-R1**（arXiv:2501.12948）是标志性工作：R1-Zero 从 base model 出发、**不做任何 SFT、纯 RLVR**，用 GRPO（L6 精读）优化"答案正确 + 格式规范"两类规则奖励，结果**长链推理（thinking）自发涌现**——模型自己学会先想再答，因为想得越多越容易拿分。这一步证明：**奖励只要"对"、且可验证，模型就能自己发现更聪明的策略，不需要人示范。**

**方法**：奖励是规则函数 $R(y) = \mathbb{1}[\text{答案正确}] + \mathbb{1}[\text{格式合法}]$（DAPO 用 $\pm 1$ 的规则奖励）；advantage 用 GRPO 的组内归一化 $(R_i - \mathrm{mean}(R))/\mathrm{std}(R)$；与阶段 2 的本质区别是**没有 RM 可以骗**（DeepSeek-R1 论文原话：不存在 reward hacking 的中间层）。

**关键数字**：R1 系列在 AIME 2024 达到 ~79%（训练前基座 <15%）；DeepSeekMath-RL 7B 用 RLVR 把 MATH 从 46.8% 推到 51.7%（L6 已有）；DAPO 用 Qwen2.5-32B 在 AIME 上做到 50 分。

**信号来源**：可验证的正确性（答案精确匹配、测试通过、规则判定）。**局限**：只能用于"能写判据"的任务——数学、代码、谜题；开放问题（"写一封好的邮件"）没有验证器。这正是向 Agent 时代迁移的动机：**Agent 任务虽然开放，但环境本身可以当验证器**（任务成功=测试通过/目标达成/状态收敛）。

### 阶段 4：Agent 时代（2024-2026）— 环境反馈

**核心目标**：把训练单位从"一段文本输出"升级为"一整条 Agent 轨迹"——模型在循环里反复调用工具、读取观察、修正动作，训练信号来自**环境在轨迹末端给出的成败**，以及执行过程中产生的步级信号。这一步回答 L4 埋下的问题：RLEF 已经证明"执行反馈 + RL"能教会模型修代码（70B：27.5→40.1 on CodeContests），Agent 后训练把同样的思路推到**长程、多工具、真实环境**。

**方法**（三类递进的 Agent RL）：
1. **工具调用级**：把"调用哪个 API / 写什么工具参数"当动作。ToolLLM（arXiv:2307.16789）用 API-RL 微调让模型学会 16000+ 真实 API 的调用；WebShop（arXiv:2207.01206）用 IL+RL 训练购物 Agent（但 L4 记过它的教训：IL+RL 28.7 只与 IL 29.1 打平，说明那时环境信号还不够强）。
2. **环境反馈级**：RLEF 把"代码修复 MDP"整个丢进 PPO；**WebRL**（arXiv:2411.02337）把 WebArena 这类网页任务建成自演化在线课程——先让模型在线探索，用**结果监督的奖励模型（ORM）**判定子目标成败，再把失败任务加工成新训练数据。**关键数字**：Llama-3.1-8B 在 WebArena-Lite 上 4.8%→42.4%，GLM-4-9B 6.1%→43%，超越 GPT-4-Turbo 的 17.6%。
3. **全轨迹级**：OpenAI《Optimizing Agent Trajectories》（2025-10，技术报告）把 RL 直接作用在**远程沙箱 Linux 环境里跑出的完整 Agent 轨迹**上（终端、文件、代码执行），奖励是规则化环境信号（测试是否通过、终端任务是否成功），advantage 用无 critic 的组间相对估计，模型规模到前沿推理模型级别。DeepMind 的《A Subgoal-driven Framework for Improving Long-Horizon LLM Agents》（MiRA）则解决 Agent RL 的**稀疏延迟奖励**问题：用里程碑式（milestone）密集奖励做离线 RL，Gemma3-12B 在 WebArena-Lite 上 6.4%→43.0%。

**信号来源**：环境执行结果（测试通过、工具返回、终端状态、子目标达成）——从"人类写示范"到"人类排偏好"再到"规则判定"，信号的**可获得性**和**可自动化程度**在这一步达到最高，这正是自改进 Agent 能闭环的原因：**奖励生成完全不需要人**。**新问题**：稀疏延迟奖励导致 credit assignment 困难（模型不知道第几步做错了）；环境本身要沙箱化、可回放、可并行；reward 判据要防 hack（比如模型学会"刷日志假装工作"）；策略分布漂移（在线探索会遗忘）。这些工程问题直接接上 L10 的 SWE-Agent 工程和 L14 的 Agent 评测。

### 一张表收束：四个阶段的信号来源迁移

| 阶段 | 信号来源 | 数据形态 | 方法 | 目标 | 代表 |
|---|---|---|---|---|---|
| SFT | 人类示范 | (指令, 期望回复) | 交叉熵微调 | 学"怎么做" | FLAN、InstructGPT-SFT |
| RLHF | 人类偏好 | 候选回复排序 | RM + PPO | 学"什么是好" | InstructGPT、ChatGPT |
| RLVR | 可验证正确性 | 规则判据 | GRPO（无 RM） | 学"什么是正确" | DeepSeek-R1、DAPO |
| Agent RL | 环境执行结果 | 整条轨迹 + 成败 | 轨迹级 RL / 里程碑奖励 | 学"什么动作序列能完成任务" | RLEF、WebRL、MiRA、OpenAI Agent RL |

## 教学主线（想象 Melvin Johnson 会怎么讲）

Melvin 是产业派，最自然的讲法就是**跟一个模型的一生**——从 base model 出生到成为 Agent，每喂一种新信号就长大一截，每次"喂不进去"就是下一阶段的动机：

1. **从 base model 讲起，建立"模型不会说话"的直觉**：base model 是文本续写器，你把指令喂给它，它续写出一段"看起来像在回答问题但其实没按指令办"的东西。一个演示：给 base model 一段指令，观察它如何把指令当正文。**动机**：需要教它"对话"这个行为。

2. **SFT：教它"怎么做"**。把人类示范直接当监督。讲清楚 SFT 就是交叉熵，它只做一件事——把示范行为的概率推高。然后立刻给出它的天花板：模型只会模仿见过的行为；示范里没有"判断好坏"的信息。用一个反例：SFT 后模型对"helpful vs harmful"的区分没有概念，什么都照做。**动机**：需要"好坏"这个维度。

3. **RLHF：教它"什么是好"**。引入三段式（SFT→RM→PPO），重点讲 RM 是在学一个**代理目标**，KL 约束在防止模型走太远，PPO-ptx 在对抗 alignment tax。关键直觉：RM 是标量、是近似、是**可以被骗的**。给一个 reward hacking 的例子（模型学会说套话拿高分），引出 Constitutional AI 想解决的成本问题（L4 已精读，这里一句话带过）。**动机**：RM 骗得动，且只评"文本好坏"，评不了"任务成没成"。

4. **RLVR：把 RM 换成验证器**。这是全场最关键的一步翻转：与其学一个奖励函数，不如直接检查答案对不对。讲 R1-Zero 的戏剧性——**纯 RL、无 SFT、thinking 自发涌现**，说明"信号对了，聪明的策略不用教"。指出 RLVR 的边界：只有能写判据的任务适用。**动机**：开放任务没有标准答案，但环境能当验证器。

5. **Agent 后训练：奖励来自环境**。把"一段文本"换成"一条轨迹"，把"奖励模型"换成"环境判据"。用 RLEF（修代码）、WebRL（网页购物）、MiRA（里程碑奖励）、OpenAI 的远程环境 RL 讲清楚：动作空间从"下一个 token"扩展成"工具调用序列"，奖励从"文本质量"变成"任务成功"。**读者最容易卡住的地方**：(a) 以为 Agent RL 和 RLVR 是一回事——RLVR 的奖励是任务开始前就写死的判据，Agent RL 的奖励是环境跑出来的、可能稀疏且延迟；(b) 分不清 credit assignment 为什么更难——轨迹有几十步，只有最后一步给了 +1，中间哪一步错了没有直接信号；(c) 以为 SFT/RLHF 被淘汰——真实管线是**叠加**的，Agent 模型也要先 SFT、再对齐、再 RLVR、最后才轨迹级 RL。

6. **收束到课程地图**：把整条时间线映射回前八讲——L4 的 RLEF/CAI 是 Agent 反馈的种子、L6 的 GRPO/DAPO 是 Agent RL 的算法引擎、L8 的深度研究是"环境当验证器"的具体形态；再预告 L10/L11/L14 会反复用到"轨迹、环境反馈、评测"这三个词。一句话收束全课："后训练的历史，就是奖励信号从人类手里，交到验证器手里，最后交到环境手里的历史。"

## 代码演示点子（5 个）

1. **手算 SFT / RLHF / RLVR 的 loss 形态**：同一组 toy 数据（一条指令、若干候选输出、带标签：示范文本 / 偏好对 / 答案对错），用 numpy 分别实现三段损失——SFT 的交叉熵（只学示范），RLHF 的 PPO clipped surrogate + KL（优化 RM 分数但限制偏离），RLVR 的 GRPO 组内 advantage（$r$ 归一化后对每组 token 加权）。打印同一候选在不同范式下的梯度方向与大小。期望输出：SFT 只推高"示范过"的概率、RLHF 推高"被偏好但别太飘"、RLVR 推高"本组内更接近正确"的——直观看到三个时代的优化目标差异。

2. **numpy 在 toy bandit 上对比 PPO 与 GRPO**：一个 10 臂 bandit（动作=输出一个答案，环境判定对错），实现 (a) 带价值网络 + GAE 的 PPO，和 (b) 组内归一化 advantage 的 GRPO，各自迭代若干轮对比采样分布收敛。再复现 RLVR 的一个著名失败模式：**全对组 / 全错组 advantage 全为 0**，梯度为零、batch 白算（引向 DAPO 的 Dynamic Sampling 与 RL-ZVP）。期望输出：PPO 需要价值网络收敛、GRPO 零 critic 但全对组停摆的对比曲线。

3. **奖励信号来源的可视化**：画一张"信号来源迁移"图——四个阶段各自的数据成本（标注成本）、可自动化程度、密集/稀疏、是否可被 hack，用条形图或散点图把"人类示范 / 人类偏好 / 规则判据 / 环境结果"四个信号投影到"成本 vs 信息量"平面上。期望输出：一张一眼看懂"为什么一路迁到环境反馈"的图；顺带标注每阶段代表工作。

4. **轨迹级 vs 步级奖励的 credit assignment 对比**：一个 toy 3 步环境（只有第 3 步决定成败），用 REINFORCE 分别在"轨迹级奖励"（最后给 ±1）和"步级密集奖励"（每步给子目标信号，模拟 MiRA 的里程碑奖励）下训练，比较梯度方差与收敛步数。期望输出：稀疏轨迹级奖励方差巨大、收敛慢；密集步级奖励方差小、收敛快——直接解释"为什么 Agent RL 需要过程信号 / 里程碑奖励"。

5. **奖励信号来源的飞轮模拟（toy）**：不训大模型，用规则环境模拟一个 SWE-bench/WebArena 式的循环——"模型输出动作序列 → 环境判定成败 → 成败信号当奖励 → 过滤成功轨迹 → 再采样"（STaR/WebRL 的骨架）。用 llm_client 的 脚本化 模式跑一个多轮循环，展示"环境反馈让成功轨迹占比逐轮上升"，并演示失败任务的两种命运（丢弃 vs 改造成新训练样本，对应 WebRL 的自演化课程）。期望输出：成功率的上升曲线 + 数据飞轮的膨胀图。

## 作业点子（3 个）

1. **后训练四阶段分类填空**：给出 6 条训练设置描述（如"用人工书写的 (prompt, 回复) 直接做交叉熵微调""用规则函数判断答案与标准答案是否一致做 GRPO""让标注员对 6 个候选排序后训练 RM 再 PPO""在沙箱环境里跑完整 Agent 轨迹、用测试是否通过当奖励"），填空写出属于哪个阶段（SFT/RLHF/RLVR/Agent RL），assert 分类正确。小提示：先找"信号来源"——示范、偏好、规则判据、还是环境结果。

2. **RLVR advantage 填空**：给定 `rewards = np.array([1.0, -1.0, 0.0, 1.0])`，填空实现组内归一化 advantage `(r - mean) / std`，断言已知结果；再断言"全对组 `[1,1,1,1]` 的 advantage 全为 0、梯度为零"，并回答为什么这浪费算力（引向 Dynamic Sampling）。小提示：`np.std` 的 ddof 要与教程口径一致，先算均值再减均值除标准差。

3. **为 toy Agent 任务写环境奖励函数**：给一个 3 步工具调用任务（如"查数据库 → 过滤 → 返回结果"），每步有子目标（子目标达成与否可判定），填空实现两个奖励函数——`trajectory_reward`（只有最后成功给 +1）和 `milestone_reward`（每步子目标达成给 +0.3、最后成功给 +1），assert 两者数值，并打印"稀疏 vs 密集"带来的梯度信号差异。小提示：里程碑奖励其实是在给模型"过程信号"，这正是 MiRA 的动机；先判每步子目标是否达成，再累加。

## 参考资料

- Training language models to follow instructions with human feedback（arXiv:2203.02155，InstructGPT）— SFT→RM→PPO 三段式 RLHF 的标杆，ChatGPT 的技术前身；1.3B 打平 175B 的关键数字
- Deep Reinforcement Learning from Human Preferences（arXiv:1706.03741，Christiano et al. 2017）— RLHF 的思想源头：用人类偏好学奖励模型，而非手写奖励
- Constitutional AI: Harmlessness from AI Feedback（arXiv:2212.08073）— 用 AI 反馈压缩人工标注成本；L4 已精读
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning（arXiv:2501.12948）— RLVR 的标志：R1-Zero 无 SFT 纯 RL、thinking 自发涌现；"没有可 hack 的 RM"原文出处
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（arXiv:2402.03300）— GRPO 提出；L6 已精读
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale（arXiv:2503.14476）— RLVR 的工程化（全对/全错组零梯度、Clip-Higher 等）；L6 已精读
- RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning（arXiv:2410.02089）— "执行反馈 + RL"教会模型修代码，Agent 轨迹 RL 的最小原型；L4 已精读
- WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning（arXiv:2411.02337）— 网页 Agent 的自演化在线课程 RL；Llama-3.1-8B 4.8%→42.4% @ WebArena-Lite
- WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents（arXiv:2207.01206）— 早期 IL+RL 网页购物 Agent，Agent RL 的对照起点
- ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs（arXiv:2307.16789）— 工具调用训练的早期代表
- OpenAI《Optimizing Agent Trajectories》（2025-10 技术报告，随博客 "Advancing RL for agentic systems" 发布）— 在远程沙箱 Linux 环境做全轨迹级 RL 的前沿案例；奖励来自测试/终端成败
- DeepMind《A Subgoal-driven Framework for Improving Long-Horizon LLM Agents》（MiRA/SGO）— 里程碑式密集奖励解决 Agent RL 的稀疏延迟奖励；Gemma3-12B 6.4%→43.0% @ WebArena-Lite
- FLAN: Finetuned Language Models are Zero-Shot Learners（arXiv:2109.01652）— instruction tuning / SFT 时代的代表作
- CS329A 课程大纲（https://cs329a.stanford.edu/）— 本讲在课程中的定位；嘉宾 Melvin Johnson（Google DeepMind）
