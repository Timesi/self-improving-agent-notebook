# Lecture 02 — Test-time Compute 缩放 研读笔记

> 本文件是 CS329A 第 2 讲（Test-time Compute Scaling）的论文研读笔记，是编写 `notebooks/part1-foundation/02-test-time-compute.ipynb` 的素材。
> 来源：https://cs329a.stanford.edu/（Autumn 2025 课程大纲）

> **文件勘误**：`papers/lecture-02/archon.pdf`（磁盘上标注 2409.07854）与 `monkey-laws.pdf`（标注 2410.05468）内容与标题不符——前者实际是代数几何论文 "2-Gorenstein Stable Surfaces"，后者实际是 NeRF/3DGS 视觉论文 "PH-Dropout"。两篇论文正确的 arXiv ID 分别是 **Archon = 2409.15254**、**Monkey Laws = 2502.17578**（已按此精读并在参考资料给出正确链接）。若重下 PDF，请用正确的 arXiv ID。

## 课程主题

这一讲要回答的核心问题：**模型训练完之后，还能不能通过花更多推理算力（test-time compute）获得更好的能力？如果能，给定一笔固定推理算力预算，怎么花最划算？**

为什么安排在 L1（总览）之后的第 2 讲：它是 Agent 全部后续内容的地基。Agent 的本质是在"一次调用"之外做更多计算（搜索、验证、多轮、多智能体），这些都属于 test-time compute 的变体。这一讲建立统一视角：**proposer（提议者）——verifier（验证者）** 两个旋钮，以及"算力预算如何按题目难度分配"。后面 L3 验证、L4 工具与代码反馈、L5 规划搜索都能回溯到这里的框架。

四篇论文的分工：

| 论文 | 角色 | 回答的问题 |
|---|---|---|
| Large Language Monkeys | 现象 | 最简单的 test-time compute（重复采样）能带来多大收益？coverage vs precision 两轴 |
| Snell Test-Time Compute | 方法/策略 | 给定预算，怎么按题目难度 compute-optimal 地分配（search + revision） |
| Archon | 工程/系统 | 多种技术怎么组合成架构并自动搜索最优组合 |
| Monkey Laws | 理论 | 为什么重复采样会出现幂律缩放？power law 从哪来 |

## 论文精读

### 论文 1：Large Language Monkeys（arxiv:2407.21787，`large-language-monkeys.pdf`）

- **核心思想**：把"推理算力"当作与训练算力平行的缩放轴。方法朴素到极致：对每个问题，用温度采样生成 k 个独立候选解，再用验证器挑一个作为最终答案（即 pass@k）。收益由两个量决定：
  1. **Coverage**（覆盖）：k 个样本里至少有一个是正确的那些问题占比——即"模型到底能不能解出这道题"。
  2. **Precision**（精度）：从一堆候选里能不能挑出正确的那个——即"验证器能不能在干草堆里找针"。
  在有自动验证器（单测、证明检查器）的领域，coverage 直接等于最终成功率；在无验证器的领域（数学应用题），coverage 只是上界，实际成功率取决于验证方法。

- **关键公式/算法**：
  - **pass@k 无偏估计**（来自 Codex/Chen et al.）：每个问题生成 N 个样本，其中 C 个正确，则
    $$\mathrm{pass@k} = \frac{1}{P}\sum_{i=1}^{P}\left[1 - \frac{\binom{N-C_i}{k}}{\binom{N}{k}}\right]$$
    用 Chen et al. 的数值稳定实现（`1 - prod(1 - k/arange(N-C+1, N+1))`）。关键：不是简单算"有正确样本的问题比例"，而是要无偏。
  - **覆盖率幂律（inference-time scaling law）**：$\log(c) \approx a\, k^{b}$，即 $c \approx \exp(a\, k^b)$。拟合示例：Llama-3-8B-Instruct 在 MATH 上 $(a=-1.33, b=-0.43)$；70B-Instruct MATH $(a=-0.75, b=-0.46)$；8B-Instruct CodeContests $(a=-3.88, b=-0.11)$；Gemma-2B CodeContests $(a=-8.54, b=-0.14)$；Pythia-70M MATH $(a=-7.59, b=-0.18)$。拟合方法：对数刻度等距取 40 个点，SciPy `curve_fit`。
  - **推理 FLOPs 估算**（用于成本对比）：$\mathrm{FLOPsPerToken} \approx 2 \times (\text{参数量} + 2 \times \text{Layers} \times \text{TokenDim} \times \text{ContextLen})$。
  - 采样超参：MATH/GSM8K 温度 0.6、每问题 10000 样本；MiniF2F 温度 0.5；CodeContests 温度 0.6 + top-p 0.95；SWE-bench 温度 1.6、250 次尝试。

- **关键实验结论（具体数字）**：
  - **SWE-bench Lite**（真实 GitHub issue，有自动单测）：DeepSeek-Coder-V2-Instruct + Moatless Tools 框架，单次 15.9% → 250 次采样 **56%**，超过单次 SOTA（43%，GPT-4o+Claude 3.5 Sonnet 混合）13 个百分点。
  - **CodeContests**：Gemma-2B pass@1 仅 0.02% → pass@10k 7.1%（**300 倍**）。Pythia 全系在该数据集上 0 覆盖（训练数据缺代码）。
  - **MATH**：Llama-3-8B-Instruct coverage 从 100 样本 82.9% → 10000 样本 98.44%；Pythia-160M 0.27% → 57%。
  - **精度问题（本文最关键的反直觉发现）**：MATH 上 coverage 涨到 95%+，但 majority vote 从 40.50% 只涨到 41.41%，reward model + best-of-N 同样在 ~100 样本处 **plateau**。结论：没有可靠验证器时，重复采样的收益上不去——"coverage 与成功率之间的 gap 随样本数扩大"。
  - **成本对比（Table 1）**：SWE-bench 上 DeepSeek-Coder-V2-Instruct 采 5 次，29.62% 解出率、总成本 $10.8（1x）；GPT-4o 单次 24%、$39（3.6x）；Claude 3.5 Sonnet 单次 26.7%、$51（4.7x）。弱模型+多次采样能比强模型单次更便宜还更好。
  - **验证器瑕疵（两个警示）**：SWE-bench Lite 有 11.3% 问题测试套件 flaky（34 个问题，其中 30 个连 gold solution 都时而判错）；CodeContests 122 个 Python3 问题中 35 个存在"gold solution 也过不了测试"的假阴性。
  - **CoT 忠实度（Table 2）**：人工标注 105 条正确样本的思维链，>90% 是忠实的推理（说明验证器有信号可利用，问题是验证器不够好）。
  - 顺带发现 GSM8K 第 1042 题 ground truth 是错的（答案是 3.5 不是 3）。

- **与课程主题的关系**：这一篇定义了 test-time compute 缩放的两个基本量 coverage/precision，给出最简单的缩放方式与幂律证据，并指出"无验证器就卡住"这一核心瓶颈——直接把问题抛给后面的讲次（L3 验证器、L5 搜索）。

- **可演示的代码点**：
  - 从零实现数值稳定的 pass@k 无偏估计器，用小例子手算验证。
  - 用合成数据（给定每个问题的 pass@1）模拟 pass@k 曲线，拟合 $\exp(ak^b)$ 并画 log-log。
  - 对比"majority vote / best-of-N（oracle）"的收益曲线，重现 plateau vs 持续上升。

### 论文 2：Scaling LLM Test-Time Compute Optimally（arxiv:2408.03314，`snell-test-time-compute.pdf`）

- **核心思想**：给定固定推理算力预算，**不同的 test-time 方法在不同难度的题目上效率不同**。容易题适合"顺序修订"（revision，局部修正），难题适合"并行重采样 / 对验证器做树搜索"（全局探索）。因此最优策略是按题目难度自适应分配——**compute-optimal scaling**，可在同预算下把效率提升 2–4 倍。

- **关键公式/算法**：
  - **统一视角（proposer-verifier）**：test-time compute 只有两个旋钮——改**提议分布**（改输入 token，例如微调一个能自我修订的模型）或**后处理输出**（用 verifier 对候选打分/选择，例如 best-of-N、对 PRM 做搜索）。
  - **难度定义**：对每个问题用 base LLM 采 2048 样本估 pass@1，按 5 分位数分成 5 个难度 bin（比 MATH 自带的手标难度更有效）。有 oracle（用 ground truth 判对错）与 model-predicted（用 PRM 打分均值）两种；两者曲线基本重合。
  - **PRM 训练**：无人工标注，用 Monte-Carlo rollout 给每个 step 打软标签（reward-to-go 值），二元交叉熵训练；步内聚合用**最后一步的分数**（比 min/prod 更好，虽然 PRM 由此近似 ORM 但仍显著优于 ORM）。
  - **搜索方法**：best-of-N weighted（对同一最终答案的分数做边缘化求和）／beam search（束宽 $M$，选 top-$N/M$ 每步扩展）／lookahead search（beam 每步 rollout $k$ 步、温度 0，用 rollout 末尾 PRM 值打分；$k=0$ 即 beam）。lookahead 成本记为 $N\times(k+1)$。
  - **顺序 vs 并行（revisions）**：微调修订模型（每个正确答案配上至多 4 个按编辑距离相关挑选的错误答案做 SFT），推理时顺序生成一串修订；预算按 sequential/parallel 比例 $\in\{2^{-7},\dots,2^7\}$ 分配。
  - **FLOPs 兑换率**（与训练算力对比）：$X = 6ND_{\text{train}}$，$Y = 2ND_{\text{infer}}$；把模型参数放大 $M$ 倍使总 FLOPs 变 $M(X+Y)$。用小模型+test-time 匹配这笔预算，需把推理算力放大 $M + 3(D_{\text{train}}/D_{\text{infer}})(M-1)$ 倍，所以结论依赖 $R = D_{\text{infer}}/D_{\text{train}}$。

- **关键实验结论（具体数字）**：
  - **compute-optimal 比 best-of-N 少花约 4 倍算力**达到同等精度：search 场景 16 vs 64 代、revisions 场景 64 vs 256 代。
  - 搜索方面：低预算时 beam search 明显优于 best-of-N，但预算增大后优势消失甚至反超；lookahead 在同预算下普遍更差（过度优化验证器产生 1–2 步的"作弊"解）。按难度分 bin 后：容易题（bin 1–2）beam search 会随预算上升而**掉点**（over-optimization/验证器信号被放大），中难题（bin 3–4）beam 稳定优于 best-of-N，最难题（bin 5）所有方法都无大进展。
  - revisions 方面：顺序采样在 verifier 与 majority 两种选择下都窄幅优于并行；约 **38% 的正确答案会在下一轮修订里被改错**，所以必须用 verifier/majority 从整条链里挑；容易题全顺序最优，难题存在最优 sequential/parallel 比例（Fig.7）。
  - **与 14 倍大模型 FLOPs-matched 对比**：容易/中等题（bin 1–3，有时 4）或推理负载低（$R\ll1$，self-improvement 场景）时，test-time compute 比多训参数更划算；难题（bin 4/5）或推理负载高（$R\gg1$，大规模部署）时，扩训练更划算。结论：**test-time 与 pretraining 算力不是 1-to-1 可兑换的**。
  - 修订模型的 pass@1 随修订步数逐步上升（能学到从错误中修正）；微调细节：AdamW lr 1e-5，ReSTEM 在线式修订训练反而变差。

- **与课程主题的关系**：在 LLM Monkeys 的"更多采样=更好"之上加了**预算约束下的最优分配**。这是"compute-optimal"这个 Agent 设计准则的源头：为不同难度的输入选不同策略、不同预算。

- **可演示的代码点**：
  - 用合成数据模拟 best-of-N vs 不同 sequential/parallel 比例，画"预算→精度"曲线族。
  - 难度分 bin：对每个问题按 pass@1 排序分 5 档，展示每档最优策略不同。
  - 实现简版 beam search（PRM 用 mock 打分器），对比 best-of-N。
  - FLOPs-matched 可视化：画 14x 大模型"星标"在 R=0.16/0.79/22 三条 x 轴位置上的高低。

### 论文 3：Archon: An Architecture Search Framework for Inference-Time Techniques（arxiv:2409.15254，`archon.pdf`，注意正确 ID）

- **核心思想**：单一 test-time 技术（重复采样、revision、验证、融合……）各有适用场景，**没有哪个架构在全部任务上最优**。Archon 把多种技术组织成"分层 LLM 系统"（类比神经网络：层内并行、层间顺序），然后用**贝叶斯优化自动搜索**最优组合，把"搭系统"变成"调超参"。

- **关键公式/算法**：
  - **LLM 组件（全是 text-to-text 操作，无权重可学）**：
    - Generator：接受指令输出候选，支持多模型并行（ensembling）与多次采样（temperature 0.7）。
    - Fuser：把多个候选合并成更高质量输出（平均 +8.9%）。
    - Ranker：两两比较排序，取 top-K（指令跟随任务 +10.8%，距 oracle 只差 2.7%）。
    - Critic：给每个候选列优缺点，喂给 ranker/fuser（+11.5 个点）。
    - Verifier：两阶段（先给"为何正确"的推理，再给 [Correct]/[Incorrect] 结论），只放行验证过的候选（推理任务 +8.4%）。
    - Unit Test Generator + Evaluator：LLM 生成 5–10 条测试语句，Evaluator 按通过数打分（CodeContests 上 Pass@1 17.9%→29.3%，**+56%**）。
  - **结构规则**：Generator 只能在第一层；每层只放一类组件；Critic 必须在 Ranker/Fuser 之前；Unit Test 生成器在评估器之前；每层的输入输出始终是字符串列表，最后一层取第一个字符串。
  - **搜索空间（6 轴）**：top-K 生成器（1–10）、每个生成器采样数（1–5，CodeContests 可到 1000）、fusion 层数（1–4，末层恒为单 Fuser）、每层 Fuser 数（2–10 步长 2）、每层前是否加 Critic/Ranker、末层前是否加 Verifier/Unit-Test。剔除无效配置后共 **9576** 个。
  - **搜索方法**：20% 数据做 dev 集；随机采一批架构校准高斯过程代理，然后 "suggest → evaluate → refine" 循环直到预算用完；超预算的架构直接从搜索空间剔除（用于 compute-matched 对比）。可优化 accuracy/latency/cost 任意目标。

- **关键实验结论（具体数字）**：
  - 最佳架构**平均超 frontier 模型（o1、GPT-4o、Claude 3.5 Sonnet）15.1%**；只用开源模型时也平均超开源 SOTA **11.2%**；比 MoA/ADAS/AFlow 平均高 8.4–8.6%。
  - 效率：比替代框架**少 20.0% 推理调用、15.1% 输入 token、13.5% 输出 token**；给定相同 token 预算时高出 12.4%。best targeted 架构平均超 o1 8.1%、超 o1-mini 9.7%。
  - **贝叶斯优化有效性**：96.0% 的搜索找到最好架构；比 greedy 少 88.5% 次评估、比 random 少 90.4%。预算 <20 次调用时 BO 优势消失，greedy 可比。校准样本约 230–240 个后收益递减。
  - **任务差异（motivation 所在）**：指令跟随/推理 → 多生成器多样性 + 多层 fusion 深度；推理子任务各自需独立架构（MixEval +3.7、MixEval-Hard +8.9 点）；代码 → 单测 + 高采样数（用 5 条单测 + 大采样，GPT-4o Pass@1 从 40/140 提到 58/140，+44.3%）。
  - **泛化**：general-purpose 架构在未见过的 GPQA/MMLU/MMLU Pro 上保持 task-specific 的 **91.7–94.2%** 性能（ADAS/AFlow 只有 66–74%）。
  - 7B 开源模型可把性能提升 7.5%，但适合做 ranker、不适合 critic/fusion；全量 70B+ 效果最好。架构 ~5x 于单次调用的延迟与成本。
  - 最佳 general-purpose all-source 架构：10 个最强 generator 起步 → 4 层 critique+fusion（Fuser 数 8→6→4 漏斗式收窄）→ 单 Fuser 输出。

- **与课程主题的关系**：把本讲前面所有散点技术（采样、修订、验证、投票）**统一进一个可搜索的系统空间**，并用自动搜索替代手工搭系统。它是"test-time compute 组合拳"的工程化终点，也为 Agent 的"自动设计架构"提供了模板。

- **可演示的代码点**：
  - 用 `llm_client`（mock 模式）手动搭一个 Generator(3)→Critic→Ranker(top-2)→Fuser 的小流水线，对比单次调用。
  - 在简化搜索空间（比如只调 top-K 与 fusion 层数两个超参）上实现"随机搜索 vs 网格搜索"，展示 Archon 的搜索思想。
  - 做一个"单位测试筛选器"：mock LLM 生成测试语句 + 打分，观察通过率对最终答案的影响。

### 论文 4：How Do Large Language Monkeys Get Their Power (Laws)?（arxiv:2502.17578，`monkey-laws.pdf`，注意正确 ID）

- **核心思想**：回答"为什么重复采样会出现幂律（power law）"。看似矛盾：对**单个问题**，失败率应随尝试次数**指数**下降（因为 i.i.d.）；但**聚合**到整个 benchmark，失败率却按**幂律**下降。答案在于：单次成功率 $p_i@1$ 在问题间的分布是**重尾的**——少量"几乎解不出"的题把聚合曲线拖成了幂律。幂律的"力量"来自任务难度的重尾分布，而不是每个问题本身的缩放行为。

- **关键公式/算法**：
  - **每问题指数衰减**：$p_i@k = 1 - (1 - p_i@1)^k$；对大 $k$，$-\log(p_i@k) \approx (1-p_i@1)^k$（指数衰减，已被 Pythia on MATH 与 HarmBench jailbreak 数据逐题证实）。
  - **聚合定义**（复合二项分布）：$\mathrm{pass}_D@k = 1 - \int_0^1 (1-p)^k\, p_D(p)\, dp$，其中 $p_D(p)$ 是 $p_i@1$ 在问题间的密度。
  - **定理 3.1（充分性）**：若 $p_D(p) = C\, p^{\,b-1} + O(p^{\,b-1+\theta})$（$p\to0^+$），则 $-\log(\mathrm{pass}_D@k) \sim C\,\Gamma(b)\,k^{-b}$。
  - **定理 3.2（必要性）**：若 $-\log(\mathrm{pass}_D@k) \sim A\,k^{-b}$，则在温和正则条件下 $p_D(p) \sim \frac{A}{\Gamma(b)}\,p^{\,b-1}$。即"聚合是幂律" $\iff$ "$p_i@1$ 分布左尾是幂律"。
  - **分布族**：$p_i@1$ 分布用**带尺度参数的三参数 Kumaraswamy / Beta-Binomial** 拟合（$p \in (0,c)$，$c\approx 0.01\sim0.1$，因为大多数单次成功率远小于 1）。MLE 在按 $1/N$ 离散化的分辨率下做，把落在 $(0,1/N)$ 左尾桶的问题数也算进似然。
  - **新估计器**：先拟合 $\hat p_D(p_i@1)$，再按上式模拟 $\mathrm{pass}_D@k$ 外推到任意 $k$，在 log-log 空间回归出幂律指数。

- **关键实验结论（具体数字）**：
  - 数据：LLM Monkeys 的 Pythia 70M–12B 在 MATH 128 题，以及 Best-of-N jailbreaking 的 frontier 模型（Claude/GPT-4o/Gemini/Llama 3 8B IT）在 HarmBench 159 条 prompt。
  - 逐题确认指数衰减、聚合确认幂律（LLM Monkeys 的 $\approx a k^{-b}$ 与 Hughes et al. 的 $\approx a k^{-b}$，见正文式 (2)/(4)）。
  - **解释反例**：Llama 3 8B IT 在 jailbreaking 下**没有**聚合幂律——因为每条 prompt 都能在预算内被 jailbreak，$p_i@1$ 分布没有重左尾（Fig.4 可见其分布不平滑）。
  - 简单分布族缩放指数：Uniform → $k^{-1}$；Beta(α,β) → $k^{-\alpha}$；Kumaraswamy(α,β) → $k^{-\alpha}$；连续 Bernoulli(λ<1/2) → $k^{-1}$；Reciprocal(α,β) → $(1-\alpha)^k/k$（**不是幂律**）。
  - **新估计器**：比最小二乘回归的**相对误差小约一个数量级**（$|\hat b - b|/b$），等价于少花 **2–4 个数量级**的推理算力；在分布失配下也表现稳健。

- **与课程主题的关系**：为论文 1 的经验幂律提供了严格的理论解释，并把"覆盖率的缩放指数 $b$"连接到"题目难度分布"这一可测对象。它让"重复采样能不能继续涨"这个问题变成可预测的（先看 $p_i@1$ 有没有重左尾）。

- **可演示的代码点**：
  - 用合成 $p_i@1$ 分布（比如 Kumaraswamy）同时画：单问题指数衰减曲线 + 聚合幂律曲线，直观看到"指数 vs 幂律"的冲突与统一（重现 Fig.2 示意）。
  - 从 $p_i@1$ 分布**模拟** $\mathrm{pass}_D@k$，对比"分布估计器 vs 最小二乘"拟合出的指数 $b$。
  - 检查"没有重尾"的分布（如均匀远离 0）是否会失去幂律（对照 Llama 3 8B IT 反例）。

## 教学主线（想象 Stanford 老师会怎么教）

老师会把这节课组织成"给一笔推理算力预算 → 怎么花"的一条决策链：

1. **先建直觉：训练完还能花钱**。人是"难题想更久"，LLM 目前是"每题只试一次"。抛出一个看似天真的问题：多试几次会怎样？→ 引出最简单的 test-time compute = **重复采样**。

2. **引入 Large Language Monkeys，建立两个量：coverage 与 precision**。大量实证数字（SWE-bench 15.9%→56% 超单次 SOTA；Gemma-2B pass@1 0.02%→pass@10k 7.1%）证明"采样次数能换覆盖率"，且呈幂律 $\exp(ak^b)$。这里强调两件事：(a) pass@k 无偏估计器的细节；(b) **验证器决定成败**——MATH 上 coverage 到 98% 而 majority vote 只有 41%，gap 越拉越大。这是第一个"卡住读者"的点：为什么"多采样"和"最终答对"不是一回事。

3. **换问题：不是采多少样，而是预算怎么分配 → Snell 的 compute-optimal**。给出 proposer-verifier 统一视角（两个旋钮）。指出"同一方法在不同难度题目上效率不同"（beam 对难题好、对容易题会 over-optimization），于是按 pass@1 分难度 bin，每 bin 选最优策略。关键数字：同精度少 4 倍算力；与 14x 大模型 FLOPs-matched 对比中，容易/中等题 test-time 更划算、难题 pretraining 更划算。这里要解释清楚 FLOPs 兑换的坑（$R = D_{infer}/D_{train}$ 决定结论）。

4. **理论反躬：为什么是幂律 → Monkey Laws**。摆出矛盾：单题应是指数衰减（$1-(1-p_i@1)^k$），聚合却呈幂律。答案：$p_i@1$ 的重尾分布。用定理 3.1/3.2 讲清"左尾幂律 $\iff$ 聚合幂律"。用一个反例（Llama 3 8B IT 无重尾→无幂律）帮读者真正理解。同时给出工程副产品：分布估计器能省 2–4 个数量级算力来预测缩放指数。

5. **工程终点：把技术组合成系统并自动搜索 → Archon**。既然单点技术各有所长、且"最优架构随任务而变"，就搭一个可搜索的系统空间（Generator/Fuser/Ranker/Critic/Verifier/Unit-Test），用贝叶斯优化搜出 9576 个配置里的最优。关键数字：平均超 frontier 15.1%、少 20% 调用。这里呼应本讲开头："多花算力"不是无脑多采样，而是**聪明地组装 + 按题分配**。

读者最容易卡住的三个地方：
- **pass@k 无偏估计**：为什么不能直接算"有正确样本的问题占比"（会高估，尤其 k 接近 N 时）。需要一个手算例子。
- **指数 vs 幂律**：log-log 图上"直线"是幂律，单题是 log-log 上向下凹的指数——两者叠加出聚合幂律。
- **best-of-N vs majority vote 的区别**：前者靠验证器打分挑一个（需要 precision），后者靠众数（对罕见正确解免疫）；所以多数表决必然 plateau。

## 代码演示点子（3–6 个）

1. **pass@k 无偏估计器（从零实现 + 手算验证）**：实现 `estimate_pass_at_k(num_samples, num_correct, k)`（数值稳定版 `1 - prod(1 - k/arange(N-C+1, N+1))`）。用一个小例子（N=10, C=3）分别算 pass@1/pass@5/pass@10，并与朴素估计"有正确样本比例"对比，画误差。**关键观察**：无偏估计与朴素估计在 k 接近 N 时差异明显。

2. **coverage 幂律拟合（合成数据，可离线复现）**：给每个问题赋一个 $p_i@1$（采样自 Kumaraswamy(0.3,1)），模拟 k 次采样算 pass@k，对 40 个对数刻度点用 `curve_fit` 拟合 $\exp(ak^b)$，在 log-log 图叠加真实曲线与拟合。**关键观察**：拟合误差很小，重现 LLM Monkeys 的 Fig.5（拟合出负的 a、绝对值小的负 b）。

3. **"单题指数 vs 聚合幂律"冲突演示（Monkey Laws Fig.2 重现实质）**：同一批 $p_i@1$，分别画 (a) 某一难题的 $-\log(p_i@k)$ 随 k 指数下降；(b) 聚合 $-\log(\mathrm{pass}_D@k)$ 呈直线（幂律）。再换一个"无重尾"分布（如 $p_i@1$ 集中在 0.2–0.4）观察聚合曲线从幂律变快衰减。**关键观察**：重尾分布决定幂律；对应 Llama 3 8B IT 反例。

4. **验证器决定收益：oracle / majority / best-of-N 三条曲线**：合成 $p_i@1$ 分布，模拟三种选择器在 k=1..1000 下的成功率。oracle（永远选到对的）持续上涨到接近 1；majority vote 和"弱验证器"在 ~100 处 plateau。**关键观察**：重现 LLM Monkeys Fig.7 的 gap 扩大。

5. **Snell 的 compute-optimal 分配可视化**：把预算 k 拆成"并行 N_par + 顺序 N_seq"（比例从全并行到全顺序），用合成数据模拟每档难度的最优比例热力图（x=难度 bin，y=比例，颜色=成功率）。**关键观察**：容易题最优偏全顺序、难题最优偏平衡——复现 Snell Fig.7 右图。

6. **FLOPs-matched 大小模型对比（简化）**：用 `FLOPs ≈ 2N·tokens` 画"小模型+test-time compute" vs "14x 大模型 greedy"，x 轴按 $R = D_{infer}/D_{train}$ 的三种取值（0.16/0.79/22）放"大模型星标"，看星标落在线上（test-time 更优）还是线下。**关键观察**：R 小（self-improvement 场景）test-time 更划算，R 大（部署场景）pretraining 更划算。

> 演示实现建议：以上 1–5 全部可用 numpy 合成数据**离线完整执行**（mock 模式兼容，符合 CLAUDE.md 要求）；若想用真实 LLM，可用 `llm_client.get_llm()` 在 GSM8K 小样本上做真采样（real 模式），并把 mock 输出标为占位。演示 4、5 不依赖真实 LLM 即可给出与论文一致的趋势。

## 作业点子（3 个）

1. **实现 pass@k 无偏估计**：填空完成 `estimate_pass_at_k(N, C, k)`，用 (N=20, C=5) 验证 pass@5、pass@20；`assert` 通过后 `print` 出"无偏估计与朴素估计的差随 k 增大而增大"。小提示：`np.prod(1 - k/np.arange(N-C+1, N+1))` 只在 C<k 时成立，C>=k 时直接返回 1.0。

2. **从重尾分布推出幂律指数**：给定合成 $p_i@1$（Kumaraswamy(α=0.4, β=1)），模拟 $\mathrm{pass}_D@k$ 并在 log-log 上拟合出 $b$；再按定理 3.1 用左尾密度幂 $b-1$ 做理论预期，`assert abs(fit_b - theoretical_b) < 0.1`。小提示：Kumaraswamy 密度在 0 附近 $f(p)\propto p^{\alpha-1}$，故理论指数 $b=\alpha$。

3. **majority vote 的 plateau**：合成数据上实现 oracle / majority 两种选择器，扫描 k 到 10000，`assert` majority 在 k=100 之后的提升 < 1 个百分点而 oracle 提升 > 10 个百分点；再交换成"无重尾分布"验证 majority 的 plateau 变弱。小提示：majority 只对"多数派答案"敏感，罕见正确解改变不了众数。

## 参考资料

- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787)（Brown et al., 2024）— 重复采样覆盖率的实证与幂律；coverage/precision 两轴。代码 https://github.com/ScalingIntelligence/large_language_monkeys
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)（Snell et al., 2024）— compute-optimal 缩放：按难度分配 test-time 算力，revision + PRM 搜索。
- [Archon: An Architecture Search Framework for Inference-Time Techniques](https://arxiv.org/abs/2409.15254)（Saad-Falcon et al., ICML 2025）— 分层 LLM 系统 + 贝叶斯优化架构搜索。注意正确 ID 是 2409.15254（仓库内 PDF 标错）。代码 https://github.com/ScalingIntelligence/Archon
- [How Do Large Language Monkeys Get Their Power (Laws)?](https://arxiv.org/abs/2502.17578)（Schaeffer et al., ICML 2025 oral）— 单题指数衰减 + 重尾 pass@1 分布 ⟹ 聚合幂律的理论解释。注意正确 ID 是 2502.17578（仓库内 PDF 标错）。
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)（Chen et al., 2021）— pass@k 无偏估计器的出处。
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)（Wang et al., 2023）— majority vote / self-consistency，与 LLM Monkeys 的 precision 讨论直接相关。
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)（Lightman et al., 2023）— PRM 训练与 MATH 难度分 bin 的源头；Snell 论文沿用其 12k/500 split。
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)（Cobbe et al., 2021）— GSM8K 与最早的验证器训练。
- [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814)（Li et al., 2022）— 百万级重复采样的先驱；CodeContests 数据集出处。
- [Beyond Chinchilla-Optimal: Accounting for Inference in LM Scaling Laws](https://arxiv.org/abs/2401.00448)（Sardana & Frankle, 2023）— 推理 FLOPs 计入的缩放定律，Snell 论文 FLOPs 公式的依据。
