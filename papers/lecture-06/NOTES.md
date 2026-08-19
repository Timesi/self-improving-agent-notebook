English: [NOTES.en.md](NOTES.en.md)

# Lecture 06 — 训练期缩放与强化学习 研读笔记

> 本文件是 CS329A 第 6 讲（Train Time Scaling / Scaling RL）的论文研读笔记，是编写对应 notebook 的素材。
> 来源：https://cs329a.stanford.edu/（Autumn 2025 课程大纲）
> 论文：STaR（2203.14465）、DeepSeekMath（2402.03300，含 GRPO）、DAPO（2503.14476）

## 课程主题

前五讲（L1-L5）都在讲 **test-time compute**：模型训练好了、权重冻结，我们在推理时花更多算力——重复采样（L2 的 Large Language Monkeys）、self-consistency 投票、best-of-n 重排、验证器打分（L3）、代码执行反馈（L4）、树搜索（L5）。这些方法的前提是"模型本身已经会一点推理"：你只能采样出模型分布里存在的东西。

第 6 讲换一个维度：**train-time scaling**——与其在推理时多花算力，不如在训练时多花算力，让模型本身"变得更会推理"。核心工具是**强化学习（RL）**：给模型一个奖励信号（答案对不对），让它在训练中自己探索出更好的推理方式。这一讲的位置很关键：它是 Part 2（训练与进化）的开篇，回答"Agent 的智能从哪里来"。L7 的开放进化（Agent 自己设计 Agent）、L8 的搜索增强、后面 L9 的后训练演进都建立在"训练期强化"这个地基上。

三篇论文构成一条递进线索：
1. **STaR**（2022）：最小化的自举循环。不需要奖励模型、不需要价值网络，用"答案对不对"做过滤信号，让模型用自己的正确推理训练自己。思想起点。
2. **DeepSeekMath + GRPO**（2024）：把 RL 规模化。提出 GRPO，去掉 PPO 的价值网络，用"组内相对"奖励算 advantage。思想成方法。
3. **DAPO**（2025）：工程化的最后一步。论文公式能跑通，但要在千卡规模上稳定训练，还需要四个工程技巧。方法成系统。

## 论文精读

### 论文 1：STaR — Self-Taught Reasoner（arxiv:2203.14465，star.pdf）
- **核心思想**：用推理自举推理（bootstrapping reasoning with reasoning）。先让模型"显式生成中间推理步骤（rationale）再给答案"（即 CoT），把生成对了的样本过滤出来拿去微调，用微调后的模型再生成、再过滤、再微调，如此循环，模型逐步能解更难的题。全程不需要人写 rationale，也不需要单独的奖励模型或价值函数，只需要数据集里已有的"问题 + 标准答案"。作者声称这是第一个让预训练 LLM 用自己的语言建模能力自我改进的方法。
- **关键公式/算法**（Algorithm 1）：
  1. `M_0 ← M`（保留原始预训练模型，每轮都从它重新微调，避免过拟合）；
  2. 每轮 `n`：用 `M_{n-1}` 对每个问题 few-shot 生成 rationale 和答案 `(r̂_i, ŷ_i)`；
  3. **Rationalization**：对生成答案错误的问题，把正确答案作为 hint 塞进 prompt，让模型"反着推"生成 rationale `(r̂^rat_i, ŷ^rat_i)`；
  4. 过滤：`D_n = {(x_i, r̂_i, y_i) | ŷ_i = y_i}`（生成就对的），`D^rat_n = {(x_i, r̂^rat_i, y_i) | ŷ_i ≠ y_i 且 ŷ^rat_i = y_i}`（靠 hint 才对），训练时**不把 hint 放进 prompt**，假装是模型自己想出来的；
  5. `M_n ← train(M, D_n ∪ D^rat_n)`。
  只有 P=10 个带 rationale 的 few-shot 示例，配合一个没有 rationale 的大数据集。
  作者把 STaR 解释成**策略梯度的近似**：把 M 看作离散潜变量模型 `p_M(y|x) = Σ_r p(r|x)p(y|x,r)`，用指示函数奖励 `1(ŷ=y)`，则期望奖励 `J = Σ_i E[1(ŷ_i=y_i)]`，其梯度
  `∇J = Σ_i E_{r̂,ŷ~p_M}[ 1(ŷ_i=y_i) · ∇ log p_M(ŷ_i, r̂_i | x_i) ]`（log-derivative trick）。过滤掉错误答案的 rationale 等价于丢弃指示函数为零的样本的梯度。STaR 的两个工程近似：(a) 用贪心解码降低方差（牺牲探索），(b) 同一批数据做多步梯度更新。
- **关键实验结论**（基座 GPT-J 6B）：
  - **算术**：16 轮外层循环后整体准确率 89.5%；对照组（1 万条无 rationale 数据直接微调 5000 步）只有 76.3%。few-shot 基线极低（2 位数加法 <1%）。加 rationalization 后，1 轮微调 2 位数加法从 <1% 跳到 32%。
  - **CommonsenseQA dev**：Few-shot Direct 20.9% → Few-shot CoT 36.6% → Direct Finetune 60.0% → STaR 无 rationalization 68.8%（只用 69.7% 训练数据）→ STaR + rationalization 72.5%（86.7% 数据）。对比 30 倍大的 GPT-3 Direct Finetune 是 73.0%——**STaR 用 6B 模型打平 30 倍大的模型**。
  - **GSM8K test**：Few-shot CoT 3.1% → Direct Finetune 5.8% → STaR 无 rationalization 10.1%（只用 25.0% 数据）→ 有 rationalization 10.7%。
  - 人类评估：评审员 30% 更可能把 STaR 生成的 rationale 排在 few-shot rationale 前面（p=.039）；74% 更偏好 STaR 的 rationale 胜过人工写的（p<.001）。
  - 消融：微调时带上 few-shot prompt 有用（60.9→68.8 / 69.9→72.5）；**提高采样温度不能替代 rationalization**（0.5/0.7 温度反而更差，容易得到"答案对但推理错"的样本）。
- **与课程主题的关系**：STaR 是"训练期缩放"的最小原型。它第一次证明"模型自己的正确推理可以成为自己的训练数据"，把 test-time 的"采样-过滤"思想搬进了训练循环。没有奖励模型、没有价值函数，为后面 GRPO"去掉 critic"埋下伏笔；它的 `p(r|x,y)` 反推（rationalization）也是"过程监督"的雏形。
- **可演示的代码点**：在任意小数据集上实现 STaR 外层循环的控制流（生成 → 过滤 → 反推 → 合并数据集 → 微调）；用 numpy 验证 Eq.2 策略梯度等价性（log-derivative trick 下的期望梯度）。

### 论文 2：DeepSeekMath（arxiv:2402.03300，deepseek-math.pdf，含 GRPO）
- **核心思想**：两手抓——(1) 构建高质量数学语料 120B tokens（从 Common Crawl 用 fastText 分类器迭代筛选 + 人工标注，规模是 OpenWebMath 的 9 倍），把 DeepSeek-Coder-Base-v1.5 7B 继续预训练成 DeepSeekMath-Base；(2) 提出 **Group Relative Policy Optimization（GRPO）**，在指令微调后用 RL 进一步提升推理。GRPO 是 PPO 的变体：**砍掉 critic（价值网络）**，改用"同一个问题的一组采样输出的平均奖励"作为 baseline，显著降低显存与算力开销。
- **关键公式/算法**：
  - **PPO 参照**（Eq.1）：`J_PPO(θ) = E_{q,o}[ 1/|o| Σ_t min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`，其中 `r_t(θ) = π_θ(o_t|q,o_<t)/π_θold(o_t|q,o_<t)`，advantage `A_t` 由 GAE 用**学习到的价值函数** `V_ψ` 算；KL 惩罚是加在每一步奖励里的：`r_t = r_φ(q,o_≤t) − β log(π_θ/π_ref)`。问题：价值网络和策略一样大，显存翻倍；且奖励只给最后 token，逐 token 的价值很难训准。
  - **GRPO objective**（Eq.3）：
    `J_GRPO(θ) = E_{q, {o_i}_{i=1}^G}[ 1/G Σ_i 1/|o_i| Σ_t { min( r_{i,t}(θ) Â_{i,t}, clip(r_{i,t}(θ), 1-ε, 1+ε) Â_{i,t} ) − β D_KL(π_θ ‖ π_ref) } ]`。
    对每个问题采样 **G 个输出**组成一组，advantage 完全由**组内相对奖励**决定。
  - **Outcome supervision**（结果监督）：`Â_{i,t} = r̃_i = (r_i − mean(r)) / std(r)`，把该输出归一化后的奖励赋给输出里所有 token。
  - **Process supervision**（过程监督）：先对每一步（以结束 token index(j) 标识）归一化 `r̃_{index(j)}`，再 `Â_{i,t} = Σ_{index(j)≥t} r̃_{index(j)}`（该 token 之后所有步的归一化奖励之和）。
  - 与 PPO 的两点关键差异：(a) KL 从奖励里挪到 loss 里直接加 `−β D_KL(π_θ ‖ π_ref)`，用无偏估计 `D_KL = π_ref/π_θ − log(π_ref/π_θ) − 1`（保证非负，Schulman 2020）；(b) advantage 用组内统计，不需要价值网络。
  - Algorithm 1（Iterative GRPO）：每轮把 reference 设为当前策略，采样 batch → 更新 old policy → 每问采样 G 个输出 → 算奖励 → 算 `Â` → 做 μ 次 GRPO 更新；奖励模型用 replay（含 10% 历史数据）持续更新。
  - 超参：策略 LR 1e-6、KL 系数 β=0.04、每问采样 64 个输出、max length 1024、batch 1024。
  - **统一范式**（Eq.5）：`∇J_A(θ) = E_{(q,o)~D}[ 1/|o| Σ_t GC_A(q,o,t,π_rf) ∇ log π_θ(o_t|q,o_<t) ]`。所有方法（SFT/RFT/DPO/Online RFT/PPO/GRPO）都拆成三件套：数据来源 D、奖励函数、梯度系数 GC。GRPO 与 Online RFT 的唯一区别就是 GC 随奖励大小变化——对错题不仅不奖励还按幅度惩罚。
- **关键实验结论**：
  - DeepSeekMath-Base 7B：GSM8K 64.2%、MATH 36.2%，**超过 Minerva 540B**（参数不是唯一因素，数据质量很关键）。
  - GRPO 只用 GSM8K/MATH 的 SFT 子集（约 144K 题）做 RL，得到 DeepSeekMath-RL 7B：GSM8K 82.9%→**88.2%**、MATH 46.8%→**51.7%**（开源界首个非工具 MATH 破 50）；**域外** CMATH 84.6%→88.8% 也提升。MATH 用 64 路 self-consistency 到 **60.9%**。
  - 消融（图 5，1.3B）：Online RFT > RFT（在线采样优于离线）；**GRPO > Online RFT**（按奖励幅度调节梯度系数更高效）；**GRPO+PS > GRPO+OS**（过程监督更细）。迭代 RL 提升显著，第一轮最明显。
  - 预训练教训：代码训练提升数学推理（两阶段 code 400B → math 150B 最佳）；arXiv 论文语料对数学推理几乎无用。
- **与课程主题的关系**：GRPO 是整讲的**技术核心**——把"RL 改进推理"变成可复现、显存可控的公式。它示范了从 STaR 的"指示函数奖励"进化到"连续奖励 + 组内归一化 advantage"，是后面 DAPO 的直接起点。统一范式（数据源/奖励/梯度系数）给了读者一个把各种训练方法装进同一框架的思维工具。
- **可演示的代码点**：numpy/torch 从零实现 GRPO 的 advantage 与 loss；用代码对比 PPO（带价值网络 + GAE）和 GRPO（组内归一化）两种目标；演示结果监督 vs 过程监督的 advantage 传播差异；用统一范式把 SFT/RFT/GRPO 写成同一个 `gradient_coefficient` 函数。

### 论文 3：DAPO — Decoupled Clip and Dynamic sAmpling Policy Optimization（arxiv:2503.14476，dapo.pdf）
- **核心思想**：OpenAI o1 和 DeepSeek R1 的技术报告都隐藏了 RL 训练细节，社区复现困难。作者用 naive GRPO 在 Qwen2.5-32B 上只做到 AIME 30 分（DeepSeek 47 分），于是把"从论文公式到千卡规模稳定训练"之间缺失的工程细节补全，提出 DAPO 算法并**全开源**（代码基于 verl 框架 + 数据 DAPO-Math-17K），用 Qwen2.5-32B 在 AIME 2024 上做到 50 分。核心是四个技巧：**Clip-Higher、Dynamic Sampling、Token-Level Policy Gradient Loss、Overlong Reward Shaping**。另外 DAPO 直接**删掉 KL 惩罚**（长 CoT 模型分布与参考模型差异本来就大，KL 约束不必要）。
- **关键公式/算法**：
  - **规则奖励**（Eq.7）：`R(ŷ,y) = 1 if is_equivalent(ŷ,y) else −1`。可验证任务直接判答案对错，不用学习式奖励模型，避免 reward hacking。数据集把答案统一转成整数便于解析。
  - **DAPO objective**（Eq.8/12）：
    `J_DAPO(θ) = E[ 1/Σ_i|o_i| Σ_i Σ_t min( r_{i,t}(θ)Â_{i,t}, clip(r_{i,t}(θ), 1−ε_low, 1+ε_high)Â_{i,t} ) ]`，约束 `0 < #{o_i 正确} < G`；
    advantage 仍是 `Â_{i,t} = (R_i − mean({R_i}))/std({R_i})`。
    与 GRPO 的区别：归一化因子从 `1/G Σ_i 1/|o_i|`（样本级）改成 `1/Σ_i|o_i|`（**token 级**）；上下裁剪阈值解耦为 `ε_low`、`ε_high`。
  - **Clip-Higher**：默认 ε=0.2 时上界限制了"探索性 token"的概率提升——低概率 token（如 π=0.01）上限只有 0.012，很难升上去；高概率 token（0.9）上限 1.08，本来就不受限。导致熵崩坍（entropy collapse）、组内输出趋同。解法：`ε_low=0.2, ε_high=0.28`，抬高上界给低概率 token 留出上升空间，保住探索。下界不动（再大容易把概率压到 0、采样空间坍缩）。
  - **Dynamic Sampling**：若某 prompt 的 G 个输出**全对或全错**，组内 advantage 为 0 → 梯度为 0，batch 有效样本数随时间下降，梯度方差变大。解法：**过采样并过滤掉正确率 = 1 或 0 的 prompt**，只保留 `0<正确数<G` 的样本，保证每个 batch 都有有效梯度。
  - **Token-Level Loss**：GRPO 样本级 loss 对每个样本等权，长回答里单个 token 的梯度贡献被稀释，长样本里的胡话/重复模式得不到惩罚 → 熵和长度不健康地涨。改成 token 级（`1/Σ|o_i|`）后，同一模式无论出现在长短回答里都被同样促进/抑制。
  - **Overlong Reward Shaping**：默认给截断样本惩罚性奖励会引入噪声（推理过程是对的，只是太长）。先做 **Overlong Filtering**（mask 掉截断样本的 loss）稳住训练，再加 **Soft Overlong Punishment**：
    `R_length(y) = 0`（`|y| ≤ L_max−L_cache`）；`(L_max−L_cache−|y|)/L_cache`（`L_max−L_cache < |y| ≤ L_max`）；`−1`（`|y| > L_max`）。
    超参：max length 16384 + cache 4096 = 20480。
- **关键实验结论**（AIME 2024，avg@32）：
  - DAPO + Qwen2.5-32B = **50 分**，超过 DeepSeek-R1-Zero-Qwen-32B（47 分），且**只用其 50% 的训练步数**。
  - 逐项叠加：Naive GRPO **30** → +Overlong Filtering **36** → +Clip-Higher **38** → +Soft Overlong Punishment **41** → +Token-level Loss **42** → +Dynamic Sampling（DAPO）**50**。每个技巧贡献若干分。
  - 训练动态监控三件套：生成长度（探索空间的代理指标，可能停滞/回落）、奖励曲线（训练集奖励与验证准确率相关性弱，注意过拟合）、生成熵（过低=失去探索，过高=胡话）。
  - 超参：AdamW、LR 1e-6、prompt batch 512、每问 16 个响应、mini-batch 512、16 次梯度更新/rollout、eval 温度 1.0 / top-p 0.7。
- **与课程主题的关系**：DAPO 回答"公式跑通 ≠ 系统跑通"。它把 GRPO 从论文升级为工业级 RL 系统，四个技巧全是稳定训练的工程权衡（探索 vs 利用、样本效率、token 权重、截断噪声）。对 Agent 课程的意义：真实的 Agent 后训练（L9）就是 DAPO 这种规模的系统，本讲给出监控指标与失败模式清单。
- **可演示的代码点**：数值演示 entropy collapse（算 ε 上下界对低/高概率 token 可提升幅度的差别）；模拟 zero-gradient 组并展示 dynamic sampling 过滤前后的有效梯度；用 numpy 手算 soft overlong punishment 的分段函数；对比样本级 vs token 级 loss 在长短样本上的权重。

## 教学主线（想象 Stanford 老师会怎么教）

1. **承接上一讲，翻转维度**：先回顾 test-time compute——"模型不会做的题，采样一万次也不会做，只是碰运气"。所以与其在推理时多花算力，不如在训练时让模型真的学会推理。用 L2 的 self-consistency 曲线引出：同样的钱花在训练上，效果是模型自身能力的提升。
2. **STaR 建立最小自举循环**：问题——构造 rationale 数据集很贵（人工标注）或需要已有解法模板。STaR 的答案：不用构造，让模型自己生成，**用答案对不对当过滤器**。画图（Fig.1 的循环）：生成 → 过滤正确的 → 微调 → 再生成。关键直觉：正确性是一个二元奖励，STaR 就是一个"奖励=指示函数、无 critic"的策略梯度（Eq.2）。这里读者容易卡住的问题："模型只会生成它会的题，那难题永远学不会"——这正是为什么要 rationalization：把正确答案当 hint，让模型"倒着推"，训练时假装没给 hint。用算术实验曲线（1 位数先会、2 位数才会、……）说明阶段式提升和 rationalization 如何一次学多个长度。
3. **DeepSeekMath 把 RL 规模化**：STaR 的奖励是二元的（对/错），现在换连续奖励模型 + 显式 RL。先给 PPO 公式（Eq.1）：要训策略 + 价值网络两个大模型，显存翻倍，且逐 token 价值难训。**GRPO 的关键动作**：把价值网络换成"同一问题的 G 个采样的平均"，advantage 变成组内 z-score（Eq.3）。手推一遍：`r̃_i = (r_i − mean(r))/std(r)`，全对组 advantage=0（为 DAPO 埋雷），一半对一半错组才有正负分化。再讲结果监督 vs 过程监督的 advantage 传播。用统一范式（数据源/奖励/梯度系数）串起 SFT/RFT/DPO/GRPO——读者会发现它们只是同一框架的三件套不同组合。
4. **DAPO 补全工程细节**：老师会强调"论文能复现了吗？"——naive GRPO 只有 30 分。四个技巧逐个讲，每个都配一个直观失败案例：熵崩坍（Clip-Higher）、全对组梯度为零（Dynamic Sampling）、长样本胡话（Token-Level Loss）、截断误罚（Overlong Reward Shaping）。用 AIME 叠加曲线（30→50）让读者看到每个技巧值几个点。最后讲监控：长度、奖励、熵三件套。
5. **收束到 Agent**：这三篇合起来就是"训练期缩放"的完整配方——STaR 的思想（自举）、GRPO 的算法（组内相对）、DAPO 的工程（稳定大规模）。预告 L7：一旦模型会自我改进，就可以让 Agent 设计 Agent；L9 后训练演进会沿用这套 RL 配方。读者容易在这里把"训练期缩放"与 L2 的"测试期缩放"混淆，老师会用一句话区分：测试期缩放改的是**采样分布**，训练期缩放改的是**模型参数**。

## 代码演示点子（4-6 个）

1. **numpy 手算 GRPO 的 advantage 与 loss**：给一个小问题、G=4 个输出及奖励（如 `r = [1, -1, 0, 1]`，对应正确/错误/部分），手算组内 mean、std、归一化 advantage `(r−mean)/std`；再给一个 toy 的重要性比率 `r_ratio = π_θ/π_θold` 列表，实现 `min(ratio·Â, clip(ratio,1−ε,1+ε)·Â)`，打印每个 token 的贡献。期望输出：全对组 advantage 全为 0、梯度为 0（引向 DAPO 的 motivation）。
2. **GRPO vs PPO 公式对比可视化**：用 numpy 实现两个目标函数，横轴画重要性比率，纵轴画目标值，展示 clip 如何在 `r` 偏离 `[1−ε,1+ε]` 时封住目标；并列出 PPO 需要训价值网络 + GAE、GRPO 只需组内统计的组件对比表。期望输出：clip 导致的目标"平台"曲线，直观看到信任域约束。
3. **迷你 STaR 循环（控制流 + 数据构造）**：不训大模型，用小 toy 数据集（如 100 道整数加法/选择题）模拟 STaR 的三步：`generate → filter(D_n) → rationalize(D^rat_n)`，用 llm_client（脚本化 模式确定性输出）或规则假模型给出 (rationale, answer)，按 `ŷ==y` 过滤。重点展示：外层循环数据集如何逐轮膨胀、rationalization 如何补充"生成失败但能反推"的样本、以及"全对就停摆"的平台现象。期望输出：数据集大小的增长曲线 + 正确率提升。
4. **统一范式：梯度系数一张表**：把 SFT/RFT/Online RFT/GRPO 的梯度系数 `GC(q,o,t,π_rf)` 分别实现成一个函数，打印同一批 (question, output) 下各方法的梯度系数，观察 RFT 只奖不罚、GRPO 按幅度奖惩。期望输出：一个 DataFrame，直观看到"为什么 GRPO 比 Online RFT 高效"。
5. **熵崩坍与 Clip-Higher 的数值演示**：给定 `π_θold = 0.01`（探索 token）和 `0.9`（利用 token），分别算 ε=0.2 与 ε=0.28 下的可提升上界，画出"低概率 token 被上界压死、高概率 token 本就不受限"的对比图，复现论文 Fig.2b 的熵下降直觉。期望输出：clip 上界曲线，直观理解"抬 ε_high 为什么保住探索"。
6. **Token-level vs Sample-level loss 的权重对比**：构造一个短样本（|o|=10）和一个长样本（|o|=100），手算两种归一化（`1/G Σ 1/|o_i|` vs `1/Σ|o_i|`）下每个 token 的实际权重，展示长样本中单 token 在样本级 loss 下被稀释。期望输出：每 token 权重的对比数字。

## 作业点子（3 个）

1. **GRPO 组内 advantage 填空**：给定 `rewards = np.array([1.0, -1.0, 0.0, 1.0])`，填空实现 `mean(r)`、`std(r)` 和归一化 advantage `(r - mean) / std`；assert 结果为已知数组；再断言"全对组 advantage 全为 0"。小提示：先算组内均值，再减均值除标准差；注意 np.std 的 ddof 参数与论文默认的一致性。
2. **裁剪目标（clipped surrogate）填空**：给定 `ratio`、`advantage`、`eps`，填空实现 `min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)`，分别用 `A>0、ratio>1+eps`、`A<0、ratio<1-eps`、`ratio∈[1-eps,1+eps]` 三个 case 断言结果；再断言：当 `A<0` 时被 clip 的是下界（惩罚激进增大概率的 token）。小提示：把 `np.clip(ratio, 1-eps, 1+eps)` 和 `np.minimum` 拆开写，先算 ratio 再取 min。
3. **STaR 过滤 + rationalization 数据构造填空**：给定 5 条生成记录（每条含 `rationale`、`answer`、`ground_truth`）和 1 条 rationalization 记录（含 hint 与否标记），填空实现 `filter_correct` 与 `rationalize`，构造出 `D_n` 和 `D^rat_n`；assert 两个数据集的条数、以及"rationalization 样本训练时不含 hint"。小提示：`D^rat_n` 只收"生成错了但反推对了"的样本；可加一个字段 `used_hint` 在入数据集时置 False。

## 参考资料

- STaR: Bootstrapping Reasoning With Reasoning（arXiv:2203.14465）— 本文精读；自举推理循环 + rationalization 的原始论文
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（arXiv:2402.03300）— 本文精读；GRPO 算法提出与统一范式
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale（arXiv:2503.14476）— 本文精读；四个大规模 RL 工程技巧与 verl 开源系统
- Proximal Policy Optimization Algorithms（arXiv:1707.06347，Schulman et al. 2017）— PPO 原始论文，GRPO 的参照
- High-Dimensional Continuous Control Using Generalized Advantage Estimation（arXiv:1506.02438）— GAE 定义，PPO 中 advantage 的来源
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models（arXiv:2201.11903）— CoT few-shot 提示，STaR 的起点
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning（arXiv:2501.12948）— R1 技术报告；DAPO 对照的基线
- Expert Iteration（arXiv:1705.08439，Anthony et al.）— 专家迭代，STaR 的理论近亲
- verl 框架（https://github.com/volcengine/verl）— DAPO 开源的 RL 训练框架
- CS329A 课程大纲（https://cs329a.stanford.edu/）— 本讲在课程中的定位
