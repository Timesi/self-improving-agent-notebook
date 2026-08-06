# Lecture 03 — 答案验证：让模型检查模型 研读笔记

> 本文件是 CS329A 第 03 讲的论文研读笔记，是编写对应 notebook 的素材。
> 来源：https://cs329a.stanford.edu/（Autumn 2025 课程大纲）
>
> 对应 OUTLINE 章节：03-robust-verification.ipynb（生成与验证的差距 / 验证器的训练：Cobbe / ORM 与 PRM / Step-by-step 验证：Lightman / 无人工标注的步级验证：Math-Shepherd / 验证器与采样结合）。

## 课程主题

这一讲要解决的核心问题：**生成容易、验证难**。LLM 经过多步推理时，自回归生成没有任何回头纠错的机制，一个中间错误就足以毁掉整条解；而即使模型能采样出正确解（Pass@K 很高），如何把正确解"挑出来"仍然是瓶颈——这就是生成-验证差距（generation-verification gap）。本讲教 Agent 学会让"模型检查模型"：训练独立的验证器（reward model / verifier）来评判候选解，再在测试时做 best-of-N 搜索。

为什么安排在 L2 之后、L4 之前：上一讲（test-time compute）建立了"多采样 + 投票"的直觉，但 self-consistency 的多数投票会平台期甚至退化，需要一个更强的"选择器"；本讲就把这个选择器做出来。同时验证器（reward model）正是 L6 强化学习（GRPO/PPO）里 reward 信号的核心来源，也是后续 Agent 循环里"critic / 自我评判"的原型。L4 的"代码执行反馈"则代表了另一种验证信号（执行结果而非语言判断），与本讲互为补充。

教学上本讲回答四个递进的问题：验证器是什么、怎么训练（Cobbe）→ 只判结果够不够、要不要判过程（Lightman 的 ORM vs PRM）→ 过程标注太贵，能不能自动（Math-Shepherd）→ 单个验证器还是太弱，怎么把一堆弱验证器合成强验证器（Weaver）。

## 论文精读

### 论文 1：Training Verifiers to Solve Math Word Problems（arXiv:2110.14168，cobbe-verifiers.pdf）
- **核心思想**：OpenAI 提出两条腿走路——发布 GSM8K 数据集，并训练"验证器"（verifier）评判模型生成的候选解，测试时采样多条解、挑验证器打分最高的。动机：多步数学推理对单点错误极度敏感，而自回归模型生成时无法自我纠错；单靠更大模型做 finetuning 的缩放曲线很差（外推需 10^16 参数或再多两个数量级的数据才能到 80%），验证则是一条缩放更友好的路。验证比生成更简单（判别 < 生成），且验证器独立于生成器，天然支持"多采样"的可选项。
- **关键公式/算法**：
  - GSM8K：8.5K 题（7.5K train / 1K test），小学水平、2–8 步、自然语言解题过程、刻意追求语言多样性（估计含破坏性错误的比例 <2%）。
  - 验证器训练 pipeline（三段）：①生成器在训练集上 finetune 2 epochs（超过会多样性塌缩、过度自信）；②对每个训练题从生成器采样 100 条候选解，按"最终答案 == 正确答案"自动打正/负标签；③验证器训 1 epoch。生成器与验证器用**两个独立网络**，防止生成器过拟合到答案。
  - 验证器目标：联合损失 = 语言建模损失 + 验证损失（MSE）。验证头实现为在 LM 最后一层 unembed 输出的一个特殊 token logit 上做 bias+gain（不干扰普通 token 的语言建模）。
  - token-level vs solution-level：solution-level 只在整条解末尾预测一次正确概率；token-level 在**每个 token 后**都预测一次（可视为 token 级 value function），测试时取最后 token 的分数作为整解分数。
  - 计算器注释：训练语料中把算式注入成 `20 + 10 = <<20+10=30>> 30`；测试时遇到 `<<...>>` 用 `eval` 覆盖模型采样，缓解算术错误。
  - 测试：每题采样 100 条（T=0.7）→ 验证器打分 → 取最高分；也可对 top-k 打分最高的解做最终答案多数投票。
- **关键实验结论**：
  - 6B 验证 ≈ finetuned 175B，等价于约 **30x 模型规模提升**；且验证随训练数据增长的曲线明显更陡（在 8K 数据规模下优势最大，小数据下验证不划算）。
  - token-level 最终优于 solution-level 且不易过拟合；联合语言建模目标是严格改进。
  - 生成器大 + 验证器小 ≫ 生成器小 + 验证器大——验证器在利用相对粗的启发式区分解。
  - test@100 随 epoch 急剧下降（模型过度自信 → 采样覆盖差），因此采样用 2-epoch 模型。
  - 每问题候选数增到 400 时性能峰值，再多下降（出现骗过验证器的对抗性解）；100 条候选时 top 3–5 投票最优。
  - dropout 0.2 对 finetuning 与验证都是显著正则化。去掉中间步骤直接输出答案会从 20.6% 跌到 5.2%（中间推理步骤必要）。
- **与课程主题的关系**：本讲的地基。确立"验证 + best-of-N"框架、自动结果标注、token-level 价值函数。本文的 solution-level 验证器 ≈ 后来的 ORM；token-level 验证器已把"分数按位置分布"带进来，是走向过程监督的过渡（每 token 目标仍是整解正确性，注意与 PRM 的区别）。
- **可演示的代码点**：在迷你算术数据集上从零训练一个结果验证器（numpy/torch 分类头），实现 best-of-N 选最高分与 top-k 投票；对比"随机选 / 多数投票 / 验证器选"；复现"候选数先升后降"的曲线。

### 论文 2：Let's Verify Step by Step（arXiv:2305.20050，lightman-verify-step-by-step.pdf）
- **核心思想**：系统比较结果监督（outcome supervision，ORM）与过程监督（process supervision，PRM）。ORM 只用最终结果对错做反馈，PRM 对推理链**每一步**给反馈。在更难的 MATH 数据集上，过程监督显著优于结果监督（PRM best-of-1860 达 78.2%）。发布 PRM800K（80 万条 step 级人工标注）。相比 Uesato et al. 2022（在小学生数学上两者相当），本文用了更强基座、更多人类反馈、更难的数据集。
- **关键公式/算法**：
  - 生成器：base GPT-4 → MathMix（1.5B 数学相关 token 的轻量预训练）→ 用 few-shot 生成 newline 分隔的 step-by-step 解、过滤最终答案正确者，再 finetune 1 epoch 只学"输出格式"（不教新技能）。
  - 数据采集（PRM800K）：人工给每步标 positive / negative / **neutral**；关键设计——**只标注到第一个错误步为止**（既保证 outcome 与 process 的监督信息量可比，也大幅降低标注成本）。主动学习：每次用当前最好 PRM 挑"评分高但答案错"（convincing wrong-answer）的解给标注者，迭代重训 PRM，数据效率提升约 2.6x。
  - ORM：同 Cobbe token-level 验证器，每 token 预测整解正确概率，测试用最后 token 分数；训练集为每题 100 条均匀采样（比 PRM800K 大一个数量级）。
  - PRM：在每个 step 最后一个 token 处预测该步正确的概率（单个特殊 token，最大化 log-likelihood），可以走标准 LM pipeline；测试时**一次前向**即得整条解所有 step 分数。
  - 解分数（solution-level reduction）：最常用 product 规则，$score(S)=\prod_i p(\text{step }i \text{ correct})$，即"所有步都正确"的概率；neutral 按 positive 处理。（Table 4：product+neutral=positive 78.2% 为最优；min 归约 / neutral=negative 各组合 77.4–77.8%，差别不大。）
  - 评测：best-of-N over 生成器均匀采样；测试集 500 道 MATH（因训练扩到 4.5K 道 MATH 测试题，只留 500 道）。
- **关键实验结论**：
  - best-of-1860：**PRM 78.2% vs ORM 72.4% vs majority voting 69.6%**，且差距随 N 增大而拉大（PRM 更擅长在大量候选里搜索）。
  - OOD 泛化（最新 AP Physics/Calculus/Chem + AMC10/12，234 题，best-of-100）：PRM 72.9% vs ORM 63.8% vs majority 61.3%。
  - 小模型合成监督实验（用 PRMlarge 当 oracle 给更小模型打标签，控制变量）：process 监督在所有数据规模下都优于 outcome-from-PRMlarge 与 outcome-from-final-answer 两种结果监督。
  - 信用分配解释：ORM 必须自己推断"错在哪一步"，而难题上大多数生成解都含错，outcome 的负标签边际信息低；PRM 直接给出第一个错步的位置。
  - Alignment：过程监督是"负 alignment tax"——更可解释、更安全，且性能还更高。
- **与课程主题的关系**：把验证从"判结果"推进到"判过程"，确立 ORM vs PRM 的框架与 PRM 的训练/评分细节。PRM 是 L6 强化学习里 step-level reward 的概念雏形，也是 Math-Shepherd（自动标注）、GRPO（DAPO/DeepSeekMath 用 step 奖励）的直接前置。
- **可演示的代码点**：从零实现 PRM 的 step 打分与 product/min 归约；实现 best-of-N 评测；模拟"只标到第一个错误步"的主动学习数据挑选；用 step 分数画每条解的热力图（哪个 step 拉低了分数）。

### 论文 3：Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations（arXiv:2312.08935，math-shepherd.pdf）
- **核心思想**：PRM 强但依赖昂贵的人工 step 标注，本文提出**自动过程标注**：受 MCTS 启发，把"某一步的质量"定义为"从这一步出发推出正确答案的潜力"。给定一个中间 step，用一个 completer LLM 从该步续写 N 条后续推理路径，按这些路径最终答案是否等于 golden answer 来给这一步打标。训练出的 MATH-SHEPHERD 既能做 best-of-N 验证重排，也能做 step-by-step PPO 强化（每个 step 结束给 reward）。
- **关键公式/算法**：
  - completer：从 step $s_i$ 续写 $N$ 条路径 $\{(s_{i+1,j},\dots,s_{K_j,j},a_j)\}_{j=1}^N$，$a_j$ 为第 $j$ 条续写解的最后答案。
  - Hard Estimation（HE）：$y^{HE}_{s_i} = \mathbb{1}[\exists j: a_j = a^*]$（只要有一条续写能到正确答案就算好步）。
  - Soft Estimation（SE）：$y^{SE}_{s_i} = \frac{1}{N}\sum_{j=1}^N \mathbb{1}[a_j = a^*]$（到正确答案的频率）。
  - 训练目标：PRM 二分类交叉熵 $L_{PRM}=\sum_i [y_{s_i}\log r_{s_i}+(1-y_{s_i})\log(1-r_{s_i})]$；对比 ORM 是整解单标签 $L_{ORM}=y_s\log r_s+(1-y_s)\log(1-r_s)$。作者发现二分类与 Lightman 的三分类差别不大。
  - 验证排名：解分数用所有 step 概率的 **min**（沿用 Lightman）；与 self-consistency 组合：$a^*_{sc+rm}=\arg\max_a \sum_i \mathbb{1}[a_i=a]\cdot RM(p,S_i)$。
  - step-by-step PPO：与传统 ORM-PPO 只在末尾给一个 reward 不同，每个推理步结束各给一个 reward。
  - 数据规模：生成器与 completer 在 MetaMATH 上 finetune 3 epochs；7B/13B 生成器在 GSM8K/MATH 训练集各 1 epoch 后每题采样 15 条解、去重；completer 用 LLemma-7B、续写 N=8。得到 GSM8K ~170K 解、MATH ~270K 解——约为 PRM800K 的 4 倍。
- **关键实验结论**：
  - 验证（best-of-256）：DeepSeek-67B + SHEPHERD 在 GSM8K 93.3%、MATH 48.1%；LLaMA2-70B(MetaMATH)：SC 88.0/39.4、ORM 91.8/40.4、SC+ORM 92.0/42.0、**SHEPHERD 93.2/44.5**、SC+SHEPHERD 92.4/45.2。
  - 对开源 LLaMA 模型，**自动标注数据在 MATH 上反超人工 PRM800K**（主因：PRM800K 基于 GPT-4 输出标注，与开源模型的解分布有 gap；自动数据量也大 4 倍）。
  - 标注质量：70B completer、N=4 时 HE 与人工标注一致率 86%；N 继续增大会引入 false positive，SE 的分布更接近人工但最终验证性能与 HE 相当。
  - 对比 NLI/规则自动步标注（Li et al.）：NLI(DeBERTa) 61.3、NLI(LLaMA2-13B) 75.6、Rule 75.0、**SHEPHERD(13B, N=4) 85.0**。
  - step-by-step PPO（Mistral-7B）：GSM8K 77.9→84.1、MATH 28.6→33.0；再叠加验证进一步到 **89.1 / 43.5**。LLaMA2-7B：66.6/19.2 →(+SHEPHERD-PPO) 73.2/21.6。同规模对比：+RFT 79.0/29.9、+ORM-PPO 81.8/31.3（step 监督 > outcome 监督）。
  - 数据效率：10k 训练实例时 PRM 就比 ORM 高约 4pt，且 PRM 上限更高。
  - OOD（匈牙利高考，33 题满分 100）：greedy 46.0、ORM 54.0、**SHEPHERD 63.0**。
  - 规模规律：大验证器校验小生成器显著提升；小验证器校验大生成器反而低于 self-consistency——应让验证器强于生成器。completer 需要"见过"题目分布（剔除相关题的 Weak 训练集表现差）。
- **与课程主题的关系**：把 PRM 从"人工标注垄断"变成"可自动规模化"，打通"验证 → 强化学习"闭环，是 L6 的 DeepSeekMath/GRPO（step-level reward 训练）最直接的前身。自动标注的质量-成本权衡（N、HE vs SE、completer 强弱）是 notebook 可实操的实验。
- **可演示的代码点**：从零实现 HE/SE 自动标注（脚本化 completer）；实现 min 归约 + self-consistency 组合选择；手算一个 step-by-step PPO 的 reward 生成；画"验证器选答案 vs majority vote"随 N 变化的曲线。

### 论文 4：Shrinking the Generation-Verification Gap with Weak Verifiers（Weaver，arXiv:2506.18203，shrinking-gen-verif-gap.pdf）
- **⚠️ 文件问题**：`papers/lecture-03/shrinking-gen-verif-gap.pdf` 里的实际内容不是本文，而是 arXiv:2504.18514 的宇宙学论文 "Inflationary background renormalization"（Kristiano & Yokoyama，与验证无关）。本讲真实论文的 arXiv 编号应为 **2506.18203**（NeurIPS 2025，Stanford + UW-Madison + Together AI，作者 Jon Saad-Falcon 等）。以下笔记基于 2506.18203 的真实内容。写 notebook 时请重新下载正确 PDF。
- **核心思想**：把前两篇建立的思想推到底——**形式化生成-验证差距**（gap = Pass@K − Success Rate），并指出"单个弱验证器不可靠"不是终点：把多个弱验证器（reward model + LM judge，7B–72B）用**弱监督**（不需要人工标签）做加权集成，就能逼近 oracle，甚至让便宜的非推理模型（Llama3.3-70B Instruct + 33 个 ≤72B 验证器）在数学/科学任务上达到 o3-mini 级（87.7% vs 86.7%），把 GPT-4o → o3-mini 的能力跃迁"用测试时采样而非微调实现"。
- **关键公式/算法**：
  - 差距定义：Gap = Pass@K − Success Rate。例：Llama3.3-70B 在 GPQA Diamond 上 Pass@100（oracle）= 82.8%，但 majority vote 只有 45.5%（37.3pt 差距）；单个验证器判对率仅 43–62%。
  - 归一化 + 二值化：各验证器输出格式不一（logits / 二值 / Likert），先归一化再二值化成投票 $\bar{s}_{ijk}\in\{0,1\}$，并过滤低质量/无信息验证器。
  - 隐变量模型（朴素贝叶斯）：解是否正确 $Y\in\{0,1\}$ 是潜在变量，各验证器输出在给定 $Y$ 下条件独立；每个验证器有 TPR $w_{k,1}=P(S_k{=}1|Y{=}1)$ 与 TNR $w_{k,0}=P(S_k{=}0|Y{=}0)$。
  - 矩估计（method of moments，无标签）：匹配可观测矩与模型矩——对（i,j）：$P(S_i,S_j)=\sum_Y P(S_i|Y)P(S_j|Y)P(Y)$，单变量：$P(S_i{=}1)=P(S_i{=}1|Y{=}1)P(Y{=}1)+P(S_i{=}1|Y{=}0)P(Y{=}0)$。目标函数 $\min_\mu \|O_{off\text{-}diag}-(\mu P\mu^\top)_{off\text{-}diag}\|^2+\|\mathrm{diag}(O)-\mu P\mathbb{1}^\top\|^2$，梯度下降求解。
  - 推断：$P(Y{=}1|\bar{s})\propto P(Y{=}1)\prod_k P(S_k{=}\bar{s}_k|Y{=}1)$，每题选后验最大的解。只需极小的 dev 集（约测试集 1%，5–10 条问答）估计任务难度 $P(Y{=}1)$。
  - 蒸馏：用 Weaver 的后验分数做伪标签，蒸馏成 400M ModernBERT-Large cross-encoder。
- **关键实验结论**：
  - 主结果（Llama3.3-70B Instruct 生成、33 验证器、K=100）：**Weaver 87.7%** vs majority vote 72.2、最强单个 RM 72.7、naive(等权) ensemble 69.1、self-verification 66.9、multi-agent verification 71.6；对照 o3-mini 86.7、GPT-4o 69.0、oracle Pass@100 91.9。分数据集：MATH500 93.4、GPQA Diamond 72.1、MMLU-College 94.9、MMLU-Pro 90.2。
  - 相对提升：比等权平均 +17.1%、比多数投票 +13.5%；Pass@1 提升 8B 模型 +17.9、70B 模型 +14.5。
  - 带权集成（oracle）比等权平均最高多 11.2pt；只给 1% 标签时监督学习的权重平均掉 20.1pt → 这正是必须用弱监督的原因。
  - 蒸馏 400M：保留 98.7% 精度、验证算力降 99.97%（35.35→1.01 exaFLOPs/query）；比多数投票高 23.2pt，只增加约 0.57% 推理成本，单卡可跑。
  - 缩放行为：majority vote 在约 2^2–2^3 exaFLOPs/query 处平台期，Weaver 可持续推到 512 exaFLOPs。
  - 弱到强：8B 生成器 + 8B 验证器用 Weaver 达 70.0%，接近 70B 多数投票基线（71.6%）。
  - 失败模式：MiniF2F、AIMO 等正确答案稀少的任务上 Weaver 优势有限（几乎采不到正确解，验证器无从发挥）。
- **与课程主题的关系**：本讲收尾的"验证器与采样结合"。回答了"单个验证器不够强、人工标签太贵"的最终问题：弱监督 + 集成 + 蒸馏是工程上把验证做鲁棒的落地方案；同时把"生成容易验证难"从直觉量化成可测量、可优化的 gap。也预告 L6：reward 信号质量决定 RL 上限（reward hacking / 过度优化正是 L9 与 alignment 讨论的主题）。
- **可演示的代码点**：从零实现 method-of-moments 估计多个模拟弱验证器的 TPR/TNR；实现朴素贝叶斯后验聚合选解；对比 unweighted / weighted / majority 三种聚合在候选数 K 增长时的表现；画 Pass@K 与 Success Rate 的差距曲线。

## 教学主线（想象 Stanford 老师会怎么教）

按"直觉 → 手算 → 代码 → 实验"组织，四篇论文是同一个故事的四个推进：

1. **失败案例建立动机（生成容易验证难）**。先回顾 L2 的 self-consistency 与 best-of-N，再抛一个刺眼的现象：正确答案明明在候选池里（Pass@100 很高），多数投票却选不出来（GPQA 上 82.8% vs 45.5%）。类比：考生能写出正确答案，但阅卷老师（验证器）太弱，把对的全判了低分。自回归生成无法回头纠错，单步错全盘皆输。这里给出本讲的核心定义：生成-验证差距 = Pass@K − Success Rate。
2. **Cobbe：最朴素的验证器**。既然答案可自动判对错（GSM8K 有唯一答案），就按最终答案给生成解打标，训一个"结果验证器"，测试时 best-of-N 选最高分。手算一遍训练 pipeline（生成器 2 epochs → 采样 100 条 → 验证器 1 epoch）。讲两个概念上的洞：①solution-level vs token-level（token-level = 每个 token 一个价值函数，引出"分数可以按位置看"）；②outcome 监督会被"歪打正着"骗（推理错、答案对 = 假阳性）。实验冲击：6B 验证 ≈ 175B finetune（30x）。
3. **Lightman：只看结果不够，要看过程**。提出信用分配问题：ORM 要自己猜错在哪一步，而难题的生成解大多有错，outcome 负标签信息量太低。PRM 让人工标每步（positive/negative/neutral），只标到第一个错步。手算 PRM 的评分：product 规则（所有步都对的概率）、neutral 当 positive。数字对比（78.2 vs 72.4 vs 69.6）并强调"差距随 N 拉大"。**读者容易卡住的地方**：混淆 Cobbe 的 token-level 与 PRM——前者每个 token 预测的是*整解*正确性（同一个 label 刷满全解），后者每步有*各自*的 label。
4. **Math-Shepherd：把人工标注自动掉**。人工 step 标注贵到不现实（PRM800K 是 80 万条人工标签）。关键直觉换法：不问"这一步本身对不对"，而问"从这一步继续，还能不能推出正确答案"——这是 MCTS 式 rollout。手算 HE 与 SE：一条续写中招就算 HE=1，SE 是招中比例。讲清楚三个可调旋钮：completer 强弱、续写数 N、HE vs SE。数字：86% 与人工一致、自动数据反超 PRM800K、step-by-step PPO 让 Mistral-7B 到 84.1/33.0。
5. **Weaver：单验证器弱，就把一堆弱验证器拧成强验证器**。回到第 1 步的 gap 定义收尾：验证器判对率只有 43–62%，但 33 个弱验证器用弱监督加权集成后逼近 oracle（87.7% ≈ o3-mini）。手算矩估计的直觉：两个验证器对同一解的同意程度里藏着它们各自的精度（无标签也能估计 TPR/TNR），再用朴素贝叶斯聚合。讲清为什么"等权平均"差——验证器质量差异很大。
6. **综合与预告**：一条轴串起验证信号来源——真值检查（执行器/最终答案）→ 结果监督 → 过程监督 → 弱验证器集成，成本递增但鲁棒性也递增。预告 L4 的"代码执行反馈"是另一条更硬的验证路径；L6 的 GRPO/PPO 会用本讲的 reward model 做训练信号，reward 质量决定 RL 成败。

关键对照表（可用于 notebook 的总结 cell）：
| 论文 | 监督信号 | 标注来源 | 解分数 | 主要用途 |
|---|---|---|---|---|
| Cobbe | 结果（整解） | 自动（最终答案） | 最后 token 分数 | best-of-N 验证 |
| Lightman | 过程（每步） | 人工（PRM800K） | product 规则 | best-of-N、主动学习 |
| Math-Shepherd | 过程（每步） | 自动（completer+HE/SE） | min 规则 | 验证 + step-by-step PPO |
| Weaver | 多验证器集成 | 无标签（矩估计） | 朴素贝叶斯后验 | 弱验证器聚合 + 蒸馏 |

## 代码演示点子（3-6 个）

1. **迷你数学数据集 + 结果验证器（ORM）**：用 numpy/torch 造一个可控的"答题器"（如规则生成的加减乘除题 + 带随机错误步的解题串），训练一个小验证器（MLP 或线性分类头，输入为解题串特征，标签 = 最终答案是否对）。随后实现 best-of-N：对每题采 N 条候选，比较"随机选 / majority vote（self-consistency）/ 验证器选最高分"三种选择器的正确率。期望输出：验证器选择随 N 增长稳定上升，majority 在某个 N 后平台期；对应 Cobbe 图 7 的直觉。
2. **手算 PRM 的 step reward**：给定一条两/三步解题过程（手工构造，含一个中间错误步），用规则定义每步正确概率，实现 product 与 min 两种归约，并演示 neutral→positive 映射。手算并可视化每个 step 的分数热力图（哪个 step 拉低了整解分数）。期望输出：错误步分数显著低，product 分数明显低于正确解；对比 Lightman Table 4 的四种评分策略排序。
3. **模拟 Math-Shepherd 自动标注（HE vs SE）**：用一个 脚本化/规则的"completer"从中间步续写 N 条路径（用伪随机 + 已知 golden answer 构造可控结果），实现 HE（$\exists$ 中招）与 SE（中招比例）公式，对比两种标签训练出的验证器在排序上的差异；改变 N=1,4,16 观察标注质量。期望输出：HE 在 N 小时与人工标注一致率 86% 的复现性直觉，N 大时 SE 更接近真实分布（对应论文 Figure 4）。
4. **验证器选答案 vs majority vote 对比曲线**：固定一个生成器（脚本化），在 GSM8K 式的小样本上画"选择正确率 vs 候选数 N"曲线：self-consistency、ORM（结果验证器）、PRM（步级、min 归约）三条线。期望输出：PRM 线在 N 增大时斜率更大、与多数投票的差距拉大——复现 Lightman Figure 3 与 Math-Shepherd Figure 3。
5. **生成-验证差距可视化**：定义并计算 Pass@K（候选池里有正确解的比例，oracle 指标）与 Success Rate（选择器实际挑对的概率），对多个选择器画 gap = Pass@K − SuccessRate 随 K 变化的曲线。期望输出：majority vote 的 gap 随 K 增大先降后升（平台期），从而直观展示"正确解生成了却选不出"是本讲的根源。
6. **迷你 Weaver：无标签矩估计 + 弱验证器集成**：构造 3–5 个带不同 TPR/TNR 的"模拟弱验证器"（对一批解给 0/1 投票，含噪声），不提供标签，仅用成对同意率做 method-of-moments 估计各验证器精度，再实现朴素贝叶斯后验选解；对比 majority、等权平均、估计带权三种聚合在候选数 K 增长下的正确率。期望输出：估计带权显著优于等权平均与 majority，且能在无标签下逼近 oracle 精度（对应 Weaver Figure 2/3 与 87.7% 故事）。

## 作业点子（3 个）

1. **PRM 解分数归约**：给一个 step 级概率数组（如 `[0.9, 0.8, 0.2, 0.9]`，第 3 步是错的），填空实现 `score_product` 与 `score_min` 两种归约（并解释 neutral 按 positive 处理）。`assert score_min(probs) == 0.2` 与 `assert abs(score_product(probs) - 0.9*0.8*0.2*0.9) < 1e-9`。小提示：product 会惩罚步数多的解，min 只盯最差一步；两题都正确时哪个归约更稳？
2. **Math-Shepherd 的 HE vs SE**：给定一条续写的 8 条路径最终答案是否正确（如 `answers = [True, True, False, False, True, False, False, False]`），填空实现 HE（任一为 True 则 1）与 SE（True 比例）。`assert hard_estimation(answers) == 1`、`assert abs(soft_estimation(answers) - 3/8) < 1e-9`。小提示：SE 是频率，HE 是存在性；N 越小 SE 噪声越大。
3. **best-of-N 里验证器 vs 多数投票**：给一个构造的候选列表（每项 `(verifier_score, final_answer, is_correct)`，答案串只有两种取值 A/B），填空实现"验证器选最高分"与"majority vote 选多数答案"两种选择器，并统计在给定列表上谁正确率更高。`assert choose_by_verifier(candidates) == "A"`、`assert choose_by_majority(candidates) == "B"`。小提示：多数投票只数答案出现次数、不关心每条的验证分数；当"错误答案占多数但验证分很低"时验证器胜出。

## 参考资料

- Cobbe et al., *Training Verifiers to Solve Math Word Problems*（arXiv:2110.14168）— 验证器 + GSM8K 的奠基论文，确立 best-of-N 与结果验证框架
- Lightman et al., *Let's Verify Step by Step*（arXiv:2305.20050）— ORM vs PRM 系统比较，发布 PRM800K（github.com/openai/prm800k）
- Wang et al., *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations*（arXiv:2312.08935）— 自动过程标注（completer + HE/SE）与 step-by-step PPO
- Saad-Falcon et al., *Shrinking the Generation-Verification Gap with Weak Verifiers*（Weaver，arXiv:2506.18203，NeurIPS 2025）— 弱监督多验证器集成与蒸馏；项目页 scalingintelligence.stanford.edu/pubs/weaver，代码 github.com/HazyResearch/scaling-verification
- Uesato et al., *Solving Math Word Problems with Process- and Outcome-based Feedback*（arXiv:2211.14275）— outcome vs process 的第一篇对比（本文与 Lightman 的基准参照）
- Li et al., *Making Language Models Better Reasoners with Step-Aware Verifier*（DIVERSE，arXiv:2210.01241）— NLI/规则自动步标注方法，Math-Shepherd 的对比基线
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*（arXiv:2203.11171）— majority voting 基线（L2 已介绍）
- Hendrycks et al., *Measuring Mathematical Problem Solving with the MATH Dataset*（arXiv:2103.03874）— MATH 数据集
- Yu et al., *MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models*（arXiv:2309.12284）— Math-Shepherd 的生成器/completer 训练数据
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*（arXiv:2402.03300）— GRPO，step-level reward 在 L6 的延续
