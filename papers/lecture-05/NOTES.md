# Lecture 05 — 多步推理与规划（Multi-step Reasoning / Planning）研读笔记

> 本文件是 CS329A 第 05 讲的论文研读笔记，是编写对应 notebook（`05-multi-step-planning.ipynb`）的素材。
> 来源：https://cs329a.stanford.edu/（Autumn 2025 课程大纲）
>
> 对应论文：LATS（2310.04406）、ADaPT（2311.05772）、SPRINT（2506.05745）、
> Wider or Deeper（2503.04412）、SWiRL（2504.04736）。

## 课程主题

这一讲解决的核心问题是：**单个 LLM 调用只能完成一步，如何让 Agent 完成需要多步、有分支、可能出错并需要回退的任务？**

它放在 Part 1 的最后一个位置，承接前三讲的两条线：

- 第 2 讲讲了"test-time compute"（推理时计算）为什么有效——把计算花在推理阶段而不是训练阶段；
- 第 3 讲讲了验证器（verifier）——用标量分数给一条解打分；第 4 讲讲了工具与执行反馈——让模型与环境交互、拿到观察。

第 5 讲把这三样东西**拼成一个"多步推理/规划"系统**：验证器分数成为搜索时的价值信号，工具执行成为动作，test-time compute 被花在"搜索多条路径"而不是"只往前走一条路"。同时它也埋下伏笔：搜索要不要训练？（SWiRL 的 step-wise RL 是一个"把规划能力直接训练进模型"的对照方案，通向 Part 2 的 RL 缩放课程。）

本讲的五篇论文正好构成"如何规划"的四个互补视角：

1. 分解（ADaPT）——把任务拆成子任务，只在需要时拆；
2. 树搜索（LATS）——在状态树上同时保留多条候选路径，用 MCTS 决定往哪走；
3. 并行执行（SPRINT）——把互相独立的推理步骤打包并行执行，降低延迟；
4. 自适应分支（Wider or Deeper）——回答"该加宽（多采样）还是加深（细化）"；
5. 训练视角（SWiRL）——不靠推理时搜索，而把"什么时候分解、什么时候调工具"直接训练进模型。

## 论文精读

### 论文 1：Language Agent Tree Search（LATS，arxiv:2310.04406，lats.pdf）

- **核心思想**：把 ReAct 式"一步步往前走"的 Agent 升级成一次**树搜索**。作者观察到很多 LLM 任务满足 MCTS 需要的"可回退"性质：想回到任何历史状态，只要把此前文本重新作为输入复制粘贴即可，不需要世界模型。于是把蒙特卡洛树搜索（MCTS）套在语言 Agent 上，让同一个 LLM 扮演三个角色——Agent（采样动作）、价值函数（评估状态）、反思器（从失败中总结教训）。这是第一个把 reasoning（推理）、acting（行动）、planning（规划）三者统一进一个框架的工作。

- **关键算法/公式**：
  - 六个操作循环执行：**selection → expansion → evaluation → simulation → backpropagation → reflection**，直到成功或耗尽预算 k 条轨迹。
  - **Selection**：从根出发，每层选 UCT 最大的子节点，直到叶节点。UCT 公式：
    $$
    UCT(s) = V(s) + w\sqrt{\frac{\ln N(p)}{N(s)}}
    $$
    其中 $V(s)$ 是子节点价值，$N(s)$ 是访问次数，$N(p)$ 是父节点访问次数，$w$ 是探索权重。
  - **Expansion**：在当前状态用 $p_\theta$ 采样 $n$ 个动作，每个动作交给环境执行、拿到观察，生成 $n$ 个新子节点。
  - **Evaluation**：价值函数是两项的加权平均
    $$
    V(s) = \lambda \cdot LM(s) + (1-\lambda)\cdot SC(s)
    $$
    $LM(s)$ 是让 LLM 对状态打个分（ToT 式启发式，但放在拿到环境反馈之后，更准）；$SC(s)$ 是 self-consistency 分数——同一个状态下采样多次都出现的动作更可靠。
  - **Backpropagation**：沿根到叶的路径更新每个节点：
    $N(s_i)=N(s_{i-1})+1$，$V(s_i)=\dfrac{V(s_{i-1})N(s_{i-1})+r}{N(s_i)}$，$r$ 是最终奖励（成功=1，失败=0）。
  - **Reflection**：失败后让 LLM 用自然语言写反思（哪里错了、下一步换什么策略），存进记忆，作为后续迭代的上下文——这是比标量 $r$ 更丰富的"语义梯度"，且不需要训练。
  - 关键超参数：expansion 采样 $n=5$，轨迹预算 $k=50$（编程任务 $k=8$）。

- **关键实验结论**：
  - HotPotQA（GPT-3.5）：LATS（CoT+ReAct）EM=0.71，是 ReAct（0.32）的两倍多；纯 acting 版 LATS（ReAct）0.63 > RAP(ReAct) 0.54 > Reflexion 0.51 > ToT(ReAct) 0.39。
  - HumanEval pass@1：GPT-4 下 LATS=92.7（当时 SOTA），高于 Base LM 80.1 与 Reflexion 91.0；GPT-3.5 下 83.8 vs Reflexion 68.1。MBPP（GPT-3.5）81.1 vs RAP 71.4。
  - WebShop（GPT-3.5）：LATS 分数 75.9 / 成功率 38.0，超过 ReAct（53.8/28.0）、Reflexion（64.2/35.0），甚至超过需要训练的 IL+RL（62.4/28.7）。
  - Game of 24：LATS(CoT) 0.44 > RAP 0.40 > ToT 0.20。
  - 消融（HotPotQA）：去掉 LM 打分掉到 0.37，换成 DFS 掉到 0.42，去掉 reflection 掉到 0.58，完整版 0.63——每个组件都不可缺。
  - 成本：同为树搜索方法，LATS 成功时平均展开节点数最少（k=50 时 66.65 vs RAP 70.60 vs ToT 84.05），token 开销低于 ToT/RAP。

- **与课程主题的关系**：这篇是"树搜索"思路的旗舰代表，直接回答"为什么要规划"——greedy 解码只看一条路，错误会沿着路径累积且无法回头；树搜索保留多条候选、用 UCT 平衡探索与利用。它同时示范了验证器（L3）如何变成搜索的价值函数、工具反馈（L4）如何变成 search 的观察信号。

- **可演示的代码点**：
  - 从零实现 `Node`（state/visits/value/children）+ UCT 选择 + backprop，不依赖任何 Agent 框架。
  - 在一个可判定的环境（如 Game of 24：给 4 个数用 +−×÷ 凑 24）上跑简化 LATS：环境反馈本身就给出 reward，mock LLM 只负责生成候选算式。
  - 可视化搜索树（每轮选哪个节点、奖励如何回传），直观看到探索与利用的交替。

### 论文 2：ADaPT — As-Needed Decomposition and Planning（arxiv:2311.05772，adapt.pdf）

- **核心思想**：规划最朴素的做法是"先列计划再执行"（plan-and-execute），但固定计划无法应对"某个子任务其实做不了"的情况。ADaPT 提出**按需分解**：先让执行器（executor）直接尝试整个任务，只有执行器自己报告失败时，才调用规划器（planner）把任务拆成 3–5 个子任务，然后对每个子任务递归调用同样的流程。分解的深度由任务难度和执行器能力共同决定，而不是预先拍死。

- **关键算法/公式**：
  - 三个模块：**executor**（LLM，与环境交互，输出动作，最后必须输出"task completed"或"task failed"作为成功启发式）、**planner**（LLM，把任务拆成 3–5 个抽象子任务，并用 AND / OR 逻辑运算符组合）、**controller**（一个固定的递归 LLM 程序，把前两者串起来）。
  - 递归式：`ADaPT(Task, k)`：若 $k > d_{max}$ 只跑 executor；否则先 `Executor(Task)`，成功则结束；失败则 `Planner(Task)` 得到 `step1 AND step2 ...`，对每个 step 递归 `ADaPT(step, k+1)`，最后按逻辑组合子结果。
  - AND 表示子任务必须顺序执行；OR 表示探索类情况（如"在台面上找马克杯 OR 在柜子里找马克杯"），任一成功即可。
  - 关键超参数：$d_{max}=3$（ALFWorld/WebShop），TextCraft 因为配方深度更深用 $d_{max}=4$；计划保持 3–5 步的短抽象计划，允许后续按需再拆。

- **关键实验结论**（GPT-3.5，成功率）：
  - ALFWorld：ADaPT=71.6%，比 ReAct（43.3）与 Plan-and-Execute（43.3）高 28.3 个点，比 Reflexion（57.5）高 14.1 个点。难点子类 pick2（要组合两次取物、动作历史长）：ADaPT=52.9%，ReAct 系基线都低于 12%（约 4 倍提升）。
  - WebShop：ADaPT=44.0，高于 LATS（38.0）、Reflexion（35.0）、ReAct（32.0）、Plan-and-Execute（17.0）。
  - TextCraft（论文新提出的 Minecraft 配方文字游戏，天然可分解、配方深度 2–4）：ADaPT=52.0 vs Reflexion 32.0 vs Plan-and-Execute 27.0 vs ReAct 19.0，高 33 个点。
  - 分析：成功率随 $d_{max}$ 单调上升；ADaPT 会随 executor 换用更弱的模型（LLaMA-2、Lemur）自动拆得更细——说明它确实在"按需"适应。

- **与课程主题的关系**：这篇代表"分解"思路。它和 LATS 是互补的两种规划策略：LATS 在"水平方向"展开（同一状态保留多个候选动作），ADaPT 在"垂直方向"展开（把任务切细）。ADaPT 的关键洞见是**失败信号来自 executor 自评而不是外部奖励**——这和 LATS 的 reflection、验证器打分是同一主题：让 LLM 自己判断"走到哪一步了、要不要回头"。

- **可演示的代码点**：
  - 写一个可配置深度的合成任务（模拟 TextCraft 式配方树），对比三种策略的展开：ReAct（一条长路径）、Plan-and-Execute（一次性全拆）、ADaPT（失败才拆）。
  - 实现递归 controller：解析 mock LLM 输出的 AND/OR 计划，按逻辑组合子任务结果。
  - 画"调用树"：展示 ADaPT 只在某条子路径上深入展开，而 plan-and-execute 全程统一深度。

### 论文 3：SPRINT — Interleaved Planning and Parallelized Execution（arxiv:2506.05745，sprint.pdf）

- **核心思想**：长 CoT 推理模型（如 DeepSeek-R1）准确率高，但输出一串很长的顺序 token，延迟高。作者分析 R1 的推理轨迹，发现大量步骤（反思、任务分解、试错探索、多个独立子计算）彼此独立，完全可以并行。SPRINT 分两步：**训练时**用一条数据整理流水线把原始顺序轨迹重排成"多轮计划 + 并行执行"的结构化数据，微调模型学会自主发现并行机会；**推理时**模型交替扮演 planner 与 executor，planner 生成一批独立子任务，executor 并行执行、同步回主上下文，形成 plan→execute→sync→replan 的滚动循环。

- **关键算法/公式**：
  - 数据流水线四步：(1) **step extraction**——用 GPT-4o 把每条轨迹拆成若干 step，每步含 planning 阶段 $P_i$ 和 execution 阶段 $E_i$（纯计划的 step 记为 $E_i=\varnothing$）；(2) **DAG creation**——用 GPT-4o-mini 判定 step 之间的依赖 $D=\{(S_i,S_j)\mid S_j \text{ 依赖 } S_i\}$；(3) **packing**——按依赖算每个 step 的阶段号
    $$
    \sigma(S_i)=\begin{cases}1, & S_i \text{ 无父节点}\\
    \max_{S_p\in Parents(S_i)}\big(\sigma(S_p)+\mathbf{1}[E_p\neq\varnothing]\big), & \text{否则}\end{cases}
    $$
    同一阶段的步骤打包并行；优化：若父节点是 plan-only（$E_p=\varnothing$），子节点可以并入父节点同一阶段；(4) **filter + SFT**——只保留 parallelization ratio = 步数/阶段数 ≥ 1.5 的轨迹（约 6000 条 → 1700 条），重排成 `<Plan_i>`/`<prompt_i.j>`/`<execution_i.j>` 标签格式后微调 DeepSeek-R1-Distill-Qwen-7B（5 epochs，lr 1e-5，8×A100）。
  - 推理时的顺序 token 度量（近似 wall-clock 延迟）：$\text{seq tokens}=\sum_{i}\max_k(P_{i.k}+E_{i.k})$，即每一阶段取其最长那一路的 token 数。

- **关键实验结论**（DeepSeek-R1-Distill-Qwen-7B，MATH-500 上训练）：
  - MATH-500：SPRINT 准确率 92.5%，比同数据微调的 RFT（91.0%）和基座（89.1%）都高，且平均顺序 token 少 440（约 15%）。在需要 >8000 token 的长轨迹上顺序 token 最高降 39%。
  - 泛化（未在 OOD 数据上训练）：Countdown 准确率 85.9%（>RFT 84.9%），顺序 token 降 53.5%（2284 vs 4917），长轨迹上最高降 65%；GPQA-Diamond 准确率 51.0%（最高），顺序 token 降 10.8%，长轨迹最高降 45%。
  - 难度越高并行收益越大：短轨迹（<2000 token）反而有约 5% 的顺序 token 开销（标签与额外 prompt 的代价）。
  - 运行时估算：MATH-500 每题 36.92s vs RFT 40.57s（快 9%）；长轨迹 74.47s vs 120.54s（快 38%）。
  - 对比 SoT（单轮计划）多出近 3 倍 token——说明多轮 plan→sync→replan 的交错执行是关键，一轮计划放不下互相依赖的子任务。

- **与课程主题的关系**：这是"并行执行"思路。前两篇（ADaPT、LATS）关注的是**怎么把任务做对**（正确性），SPRINT 关注的是**怎么做得快**（延迟）——把推理轨迹里天然存在的并行性找出来。它也是本讲里唯一把"计划与执行交错"和"训练"结合的工作：不靠手工定义搜索结构，而是用数据教会模型在推理时自主 split 任务。与 SWiRL 一样，代表"能力可以训练进模型"这条线。

- **可演示的代码点**：
  - 手写一张 step 依赖表（或从脚本化轨迹解析），实现 $\sigma(S_i)$ 的阶段划分，把同阶段步骤并行化，计算顺序 token 与串行的对比。
  - 用 networkx 画依赖 DAG，展示 plan-only 父节点优化如何把子节点并入同一阶段。
  - 用 mock LLM 模拟一次"plan→并行执行→sync"的滚动循环，数每个阶段的最长路径 token，直观看出延迟来源。

### 论文 4：Wider or Deeper? — Adaptive Branching Tree Search（AB-MCTS，arxiv:2503.04412，wider-or-deeper.pdf）

- **核心思想**：推理时缩放（inference-time scaling）最朴素有效的手段是 repeated sampling（best-of-n、majority voting）——它只"加宽"（多采样新答案），不利用外部反馈；而顺序细化（sequential refinement）只"加深"（拿着反馈改进已有答案）。标准 MCTS 想兼顾两者，但分支宽度是固定超参数，限制了扩展。AB-MCTS 的关键洞见是：**分支应该无界自适应**——在每个节点动态决定"加宽"（从当前节点生成一个全新候选答案，记作 GEN 动作）还是"加深"（细化某个已有答案）。

- **关键算法/公式**：
  - 每个节点 $N$ 都有一个特殊的 **GEN 子节点**，代表"从这里再生成一个新答案"。选到 GEN 节点就加宽；选到已有子节点就加深。与标准 MCTS 不同，已展开过的节点可以再次展开，分支数理论上无界。
  - 选择策略用 **Thompson sampling** 而不是 UCT：UCT 假设臂（分支）是静态的，但 GEN 节点会动态生成新臂，不适用。做法：对每个候选动作 $a_j$ 计算分数 $r$ 的后验预测分布 $P_N(r\mid a_j)$，从每个分布采样一个分数，取最大者。
  - 两种实现：
    - **AB-MCTS-M**（混合模型）：每个子树是一个"组"，用混合贝叶斯模型 $r_{N_{new},a_j}=\alpha_j+\sigma_y\epsilon$（组级截距 + 实例噪声），用 MCMC 拟合共享超参数；GEN 动作的 $\alpha_0$ 从共享后验推断。
    - **AB-MCTS-A**（节点聚合）：引入 CONT 节点聚合所有"细化"子节点，用指数族 + 共轭先验做解析更新——Gaussian 变体（normal-inverse-χ²，用于无界分数）、Beta 变体（用于 [0,1] 分数）。更轻量，类似标准 UCT 的哲学。
  - 生成预算 = LLM 调用次数上限，实验用 $2^7=128$。

- **关键实验结论**（GPT-4o / DeepSeek-V3，LiveCodeBench、CodeContest、ARC-AGI、MLE-Bench）：
  - 平均排名：AB-MCTS-M 2.3、AB-MCTS-A（Gaussian）2.7、AB-MCTS-A（Beta）2.7，均优于 repeated sampling（3.5）、标准 MCTS（4.2）、sequential refinement（5.5）。
  - LiveCodeBench（GPT-4o）：AB-MCTS-M 38.9 vs repeated sampling 37.8 vs 标准 MCTS 36.7；CodeContest：40.6 vs 37.9 vs 37.5。
  - ARC-AGI：repeated sampling 强（需要广泛探索），AB-MCTS 与之相当；把预算扩到 $2^9=512$ 时 repeated sampling 趋平，而 AB-MCTS 继续上升——大预算下自适应分支更会"把钱花在刀刃上"。
  - MLE-Bench（GPT-4o）：AB-MCTS-M 平均排名 1.3，最佳基线（sequential refinement）2.3。
  - 树形分析：AB-MCTS 生成的树比标准 MCTS 更宽（因为它允许从任意节点加宽），同时在有潜力的分支上也会加深——两个方向都自适应。

- **与课程主题的关系**：这篇回答整讲最关键的"如何选择"问题——到底该加宽还是加深？答案不是固定策略，而是一个**每节点在线决策**（贝叶斯后验采样）。它是 LATS 的"下一代"：同样做 MCTS，但把 LATS 里"展开 $n$ 个孩子"的固定超参数换成自适应分支。对本讲教学而言，它是从"会搜索"到"会自适应地搜索"的一步，也点出 UCT 公式在 LLM 场景（臂动态生成）下的局限。

- **可演示的代码点**：
  - 实现一个 mini 版自适应分支：两个动作（GEN / 细化），每个动作维护 Beta(α, β) 后验，用 Thompson sampling 选动作，在 mock 打分器上跑若干轮，画出选择轨迹。
  - 对比三种策略的树形（纯加宽 / 纯加深 / 自适应）在同一个合成打分环境下的表现曲线。
  - 手算演示"为什么 UCT 不适用"：构造一个动态生成新臂的场景，比较 UCB 与 Thompson sampling 的行为。

### 论文 5：SWiRL — Synthetic Data Generation & Multi-Step RL（arxiv:2504.04736，swirl.pdf）

- **核心思想**：前面四篇都在推理时（test-time）想办法；SWiRL 换一条路——**把"怎么分步、什么时候调工具、什么时候收尾"训练进模型**。传统 RLHF/RLAIF 是单步优化（只看最终回答），而多步任务里中间错一步会连锁带偏结尾。SWiRL 提出**逐步强化学习（Step-Wise RL）**：把一条多步轨迹按每个动作切成多条子轨迹（前缀），用生成式奖励模型对每一步单独打分做 RL，全程不需要 golden label 或人工标注。

- **关键算法/公式**：
  - **Stage 1 数据生成**：开源的 Gemma 2 配上工具（搜索引擎 / 计算器），迭代生成多步轨迹；每步模型可自由输出 CoT、调工具或给最终答案（用 `<search_query>` / `<math_exp>` / `<answer>` 标签标识）。生成 50,000 条 HotPotQA 轨迹（1 万题 × 5 条）、37,500 条 GSM8K 轨迹。每条 k 步轨迹拆成 k 条子轨迹（前缀 + 当前动作）。
  - **过滤策略**：四种对比——不过滤、process 过滤（用 Gemini 1.5 Pro Thinking 判断"每一步是否合理"）、outcome 过滤（最终答案是否匹配 golden）、process+outcome。**只用 process 过滤效果最好**。
  - **Stage 2 逐步 RL**：优化目标
    $$
    J(\theta)=\mathbb{E}_{s\sim T,\,a\sim\pi_\theta(s)}\big[R(a\mid s)\big]
    $$
    其中奖励 $R(a\mid s)$ 由生成式奖励模型（Gemini 1.5 Pro）对"给定前文 $s$ 的动作 $a$"打分，不用 golden label；优化器沿用 Gemma 2 微调用过的 policy gradient。
  - 推理时：迭代让模型"调工具 or 给答案"，检索用 Gecko 向量库近邻、计算用 SymPy，最多 5 次查询（QA）/ 10 次（数学）。

- **关键实验结论**（Gemma-2-27b 微调）：
  - 相对准确率提升：GSM8K +21.5%、HotPotQA +12.3%、CofCA +14.8%、MuSiQue +11.1%、BeerQA +15.3%（平均约 15%）。
  - 跨任务泛化：只在 HotPotQA（多跳问答）上训练，零样本提升 GSM8K（数学）16.9%；反向（GSM8K → HotPotQA）提升 9.2%——学到的是一般性的"如何推理"，不是任务特化。
  - **process 过滤 > outcome 过滤**：RL 能从最终答案错误但步骤合理的轨迹里学到东西（与 SFT 相反——SFT 需要正确结果，否则甚至比基座更差；这呼应"SFT 记忆、RL 泛化"）。SWiRL 最受益于含正反例混合的 process 过滤数据。
  - 规模：1000 条轨迹就有明显收益；Gemma-2-27b 有跨任务泛化，2b/9b 只有域内收益。
  - 机制：SWiRL 提升了平均 process label（HotPotQA 82.5%→91.0%，GSM8K 87.5%→91.6%），说明最终准确率提升确实来自多步推理质量变好，而不只是蒸馏奖励模型。

- **与课程主题的关系**：这篇把"规划"从推理时搬进训练时，是整讲的收束与转折。LATS/ADaPT 用搜索和分解帮"推理时的 LLM"规划，SWiRL 用 RL 让模型"自己学会"规划——两条路殊途同归（SPRINT 的 SFT 训练也属于第二条路）。它的 step-wise 分解思想（把轨迹切成前缀）与 ADaPT 的按需分解、LATS 的价值函数（用 LLM 打分）一脉相承，但用在训练信号上。SWiRL 和 L6 之后的"RL 缩放"、"后训练演进"直接衔接，是 Part 1 通向 Part 2 的桥梁。

- **可演示的代码点**：
  - 用 mock LLM 生成若干条脚本化多步轨迹，实现"子轨迹分解"（k 步 → k 条前缀），再套一个 mock 的 process 打分器，复现"process 过滤 vs outcome 过滤"在数据混合上的差异。
  - 用 toy 策略（小 softmax 分类器）演示"逐步奖励 vs 只给最后一步奖励"带来的梯度差异（可简化成加权更新）。
  - 统计"SWiRL 在只有正确 / 只有错误 / 混合结局数据上的表现"，直观看到 RL 不排斥错误结局样本。

## 教学主线（想象 Stanford 老师会怎么教）

老师的叙事顺序（对应 notebook 的章节顺序）：

1. **建立动机：单步推理的局限。** 抛一个具体失败案例：让 Agent 直接回答"谁更年长：Glenn Hughes 还是 Ross Lynch？"或者玩 Game of 24。指出 greedy 解码的三个毛病——错误沿路径累积（error propagation）、错了无法回头、不会利用环境反馈。于是自然引出问题：多步任务里，"想清楚下一步"已经不够，需要"规划"。

2. **思路 A：分解——把大任务切成小任务（ADaPT）。** 直觉类比：做一道没做过的菜，遇到不会的步骤才翻菜谱，而不是一开始就把十个步骤全背下来。核心教义是"按需"：executor 先试，失败了才让 planner 拆，拆完递归。这里要讲清楚 AND/OR 逻辑，以及成功启发式（executor 自报 completed/failed）为什么是可行的替代奖励。读者容易卡住的地方：为什么不能一次性拆完？——因为无法预知哪个子任务难，拆太细反而引入错误假设。

3. **思路 B：树搜索——同时探索多条路（LATS）。** 引入 MCTS：节点=状态，边=动作，UCT 平衡探索与利用。先手算 UCT（用一张小表），再讲六个操作，强调"LLM 任务可回退"这个前提让 MCTS 免训练可用。对比分解与搜索：ADaPT 是"垂直切片"（降低单步难度），LATS 是"水平展开"（保留多种可能）；一个处理"任务本身难"，一个处理"路径不确定"。读者容易卡住的地方：UCT 里 $\ln N(p)$ 和 $N(s)$ 为什么一个在对数里一个在外面？——未探索的孩子访问数小，$\ln N(p)/N(s)$ 大，会被优先试。

4. **思路 C：并行——把独立步骤同时做（SPRINT）。** 从"做对"转向"做快"。用一个具体例子：反思、分解、试错这几类推理步骤其实互不依赖。讲数据流水线（拆 step → 建 DAG → 打包 stage → 过滤微调），再讲推理时的 planner/executor 滚动循环。读者容易卡住的地方：为什么不直接让模型一次生成一个并行计划？——SoT 就是这么干的但会生成互相依赖的子任务，必须多轮 sync。

5. **思路 D：自适应——到底加宽还是加深（Wider or Deeper）。** 把问题升级：repeated sampling 只加宽、sequential refinement 只加深、标准 MCTS 宽度固定。AB-MCTS 让每个节点自己决定。讲 GEN 节点、Thompson sampling（为什么 UCT 失效），以及两个变体 M / A 的分工。这是把前面所有"搜索超参数"问题收束成一个"在线决策"问题。

6. **对照与收束：训练视角（SWiRL）+ 如何选择。** 指出前面都是推理时方法，SWiRL 把规划能力训练进模型，为 Part 2 铺垫。最后给一个选择决策表：反馈是否可得（决定能不能用验证器/搜索）、错误能否回退（决定 MCTS 是否适用）、延迟是否敏感（决定要不要并行）、预算大小（决定分支宽度）。给读者的心智模型：**分解解决任务难度，搜索解决路径不确定性，并行解决延迟，训练解决"每次都要现搜"的成本。**

## 代码演示点子（4-6 个）

所有演示遵守项目约定：统一走 `llm_client.py` 的 `get_llm()`，mock 模式下 LLM 输出是脚本化轨迹，算法逻辑（树、分解、打包、采样）仍必须完整跑通；核心算法用 numpy 从零实现。

1. **手算 UCT 选择公式（numpy，无需 LLM）**：给一棵手写的小树（根节点带 4 个子节点，各有 N 和 V），实现 `uct = V + w*sqrt(ln(N_parent)/N_child)`，打印每个子节点的 UCT，并选出下一个要展开的节点。改变探索权重 w 观察选择变化。期望输出：一张 UCT 数值表 + 被选节点。

2. **从零实现简化 LATS（mock LLM 兼容）**：实现 `Node`（state、visits、value、children）与六个操作的最小版——UCT 选择、用 `get_llm()` 采样 n 个候选动作（mock 模式返回脚本化候选）、环境判 reward（如 Game of 24 判定算式是否等于 24）、backprop 更新、失败时存一条 reflection 文本。跑 20 轮后输出：最优路径、树规模、每轮选中的节点序列。关键设计：reward 来自环境判定而非 LLM 打分，保证 mock 也能跑；LLM 只负责"出候选动作"，解析时对 mock 的脚本化输出宽容。

3. **ADaPT 按需分解的可视化对比**：做一个可配置深度的合成配方环境（目标物品→子物品→再子物品，某个分支设成"executor 必失败"）。用 mock LLM 作为 executor（返回 completed/failed）与 planner（返回 AND/OR 计划），分别跑 ReAct（一条长路径）、Plan-and-Execute（一次全拆）、ADaPT（失败才拆，dmax=3）。输出：三种策略的调用树（每层谁调了谁）+ 各自动作步数，直观看到 ADaPT 只在坏分支上深入。

4. **SPRINT 的 DAG 打包与顺序 token 计算**：给一张手写的步骤依赖表（如"算 A 子式"、"验算"、"反思"等，含 plan-only 步骤），实现 $\sigma(S_i)$ 阶段号公式，把同阶段步骤打包；假设每个步骤有给定的 token 数，分别按串行与并行计算顺序 token（并行 = 每阶段取最长那路）。用 networkx 画 DAG，标注阶段划分。期望输出：阶段分组 + 串行/并行 token 对比条形图，演示 plan-only 父节点把子节点并入同一阶段的优化。

5. **自适应深宽选择（Thompson sampling 迷你版，Beta 共轭）**：一个简化 AB-MCTS——每个节点只有两个动作：GEN（生成新答案）和 Refine（细化已有答案）。给每个动作维护 Beta(α, β) 后验，从各动作后验采样分数、取最大者执行；mock 打分器返回 0/1 分数并回传更新后验。跑 50 轮，画出"加宽次数 vs 加深次数"以及选择轨迹，对比"纯加宽（repeated sampling）"和"纯加深（sequential refinement）"在合成环境上的表现曲线。

6. **SWiRL 风格的数据分解与过滤**：用 mock LLM 生成若干条脚本化多步轨迹（含 search/answer 动作标签），实现"k 步轨迹 → k 条前缀子轨迹"的分解；写一个 mock 的 process 打分器（对"每步是否合理"给 0/1）与 outcome 判定（末步是否匹配 golden），统计四种过滤策略（不过滤/process/outcome/process+outcome）各自保留的样本数与标签构成。期望输出：过滤对比表，说明"process-only 保留正反例混合"，呼应论文"RL 能从错误结局样本学习"。

## 作业点子（3 个）

1. **补全 UCT 选择（填空 + assert）**：给一棵树（子节点的 N、V、父节点 N），在空位补 `uct = ...`，`assert` 选出的节点索引与手算一致。小提示：$N(s)$ 是子节点访问次数、$N(p)$ 是父节点访问次数，先想哪个节点该被优先尝试。

2. **补全 LATS backprop（填空 + assert）**：给一条从根到叶的节点路径（各节点的 N_old、V_old）和终点奖励 r，在空位补更新式，`assert` 更新后根节点的 N 与 V 等于手算值。小提示：$V(s_i)=\dfrac{V(s_{i-1})N(s_{i-1})+r}{N(s_i)}$，先算 $N$ 再算 $V$。

3. **补全 SPRINT 阶段号计算（填空 + assert）**：给一张步骤依赖表（含一个 plan-only 父节点），在空位补 $\sigma(S_i)$ 的实现，`assert` 各步骤的阶段号与手算一致，并 `assert` 出 plan-only 父节点的子节点被并入同一阶段。小提示：只有 $E_p\neq\varnothing$ 才把父节点的阶段号 +1；父节点是 plan-only 时子节点与父节点同阶段。

## 参考资料

- LATS: Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models（arxiv.org/abs/2310.04406，github.com/lapisrocks/LanguageAgentTreeSearch）— 本讲树搜索主线；MCTS 与 LLM 结合的第一篇通用框架。
- ADaPT: As-Needed Decomposition and Planning with Language Models（arxiv.org/abs/2311.05772，allenai.github.io/adaptllm）— 分解思路；按需递归分解与 AND/OR 计划。
- SPRINT: Enabling Interleaved Planning and Parallelized Execution in Reasoning Models（arxiv.org/abs/2506.05745，github.com/stanford-futuredata/Sprint）— 并行执行；DAG 打包 + planner/executor 滚动循环。
- Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search（arxiv.org/abs/2503.04412，github.com/SakanaAI/treequest）— 自适应分支；GEN 节点 + Thompson sampling，回答"加宽还是加深"。
- SWiRL: Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use（arxiv.org/abs/2504.04736）— 训练视角；逐步 RL、process/outcome 过滤对比、跨任务泛化。
- 前置概念（第 2–4 讲已涉及，可复习）：Chain-of-Thought（Wei et al., 2022）；Self-Consistency（Wang et al., 2022）；ReAct（Yao et al., 2023b）；Tree-of-Thought（Yao et al., 2023a）；Reflexion（Shinn et al., 2023）；RAP: Reasoning via Planning（Hao et al., 2023）；UCT（Kocsis & Szepesvári, 2006）；AlphaGo 的 MCTS 应用（Silver et al., 2016）。
