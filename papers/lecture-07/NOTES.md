English: [NOTES.en.md](NOTES.en.md)

# Lecture 07 — 自改进智能体的开放进化 研读笔记

> 本文件是 CS329A 第 7 讲的论文研读笔记，是编写对应 notebook 的素材。
> 来源：https://cs329a.stanford.edu/（Autumn 2025 课程大纲）

## 课程主题

这一讲要解决的核心问题：**Agent 能不能改进自己？** 前几讲（01-06）讲的都是"人设计 Agent"：我们手工搭建 ReAct 循环、验证器、工具调用、规划器，再用手写奖励做 RL。这一讲把镜头抬高一格，问一个更根本的问题——能不能让一个 Agent 去设计 Agent？也就是把"设计智能体"本身变成可以被搜索、被优化、被进化的对象。

在课程里的位置：它承接第 6 讲（RL 缩放/后训练）里"用学习代替手写"的思想，把它应用到 Agent 系统自身的设计上；同时为后面"搜索与深度研究智能体"（08）、"自治智能体"（19）、"未来展望"（20）埋下伏笔。它回答的是 Clune 在 AI-Generating Algorithms 里提出的三根支柱之一：**元学习智能体架构**（其余两根是元学习学习算法、生成学习环境）。

本讲三篇论文是同一个思想的三档体现，按"循环的完整程度"递进：

- **ADAS**：一个"元 agent"迭代地写 agent 代码、跑评测、存进 archive，实现"agent 设计 agent"的最小闭环。
- **AI Scientist**：把同样的闭环扩展到完整科研流程——提出 idea、跑实验、写论文、评审、把结果存进知识 archive。
- **AlphaEvolve**：把"代码当作基因组、benchmark 当作适应度"的进化搜索规模化，直接改整份代码文件，做出 56 年来矩阵乘法领域的首个改进。

贯穿三篇的共同骨架：一个 **变异算子（LLM 写代码/写 idea）+ 一个选择机制（评测/评审打分）+ 一个种群/archive（积累步进石）**。这就是"开放进化"的三个要素，也是本讲 notebook 要亲手实现的东西。

## 论文精读

### 论文 1：Automated Design of Agentic Systems（ADAS）（arXiv:2408.08435，adas.pdf）

- **核心思想**：机器学习的历史反复出现一个规律——手工设计的东西最终被学到的东西取代（HOG → CNN，AutoML，NAS）。ADAS 把这条规律应用到 Agent 系统设计上：定义一个 Agent 的全部组件（prompt、工具调用、工作流）都在代码里，然后用一个"元 agent"（一个 LLM）迭代地写新 Agent 的代码，测它在任务上的表现，把好的 Agent 存进一个不断增长的 archive，并用这个 archive 启发下一轮设计。由于 Python 是图灵完备的，**以代码为搜索空间理论上能表达任何可能的 Agent 系统**——这是它优于"只优化 prompt"（OPRO、PromptBreeder）和"只优化图结构"（DyLAN、GPT-Swarm）之处。
- **关键算法（Meta Agent Search）**：论文把 ADAS 形式化为优化过程，给出三个关键组件：
  1. **搜索空间**：Agent 定义为代码。框架只有不到 100 行，给元 agent 提供基础函数（查询 FM、格式化 prompt），元 agent 只需写一个 `forward(task) -> answer` 函数，类似 FunSearch 的做法。
  2. **搜索算法**：FM 作为元 agent，循环执行：基于 archive 生成高层 idea 描述 → 用代码实现 → 两次自反思确认"新颖" → 在验证集上评测 → 出错则自反思修 bug（最多 5 次）→ 连同评测指标加进 archive。同时被鼓励探索"有趣的"（新颖/有价值）Agent，这是继承开放进化文献里的"有趣性"思想。
  3. **评价函数**：在目标任务验证集上的准确率/F1，作为元 agent 要最大化的指标。
  - 关键细节：元 agent 用 GPT-4，发现的 Agent 和被评测的基线用 GPT-3.5（省算力）；ARC 上跑 25 轮，其余域跑 30 轮；archive 用基线（CoT、Self-Refine、COT-SC、LLM-Debate、QD）初始化。
- **关键实验结论**：
  - ARC 挑战：发现的 Agent 逐步超过所有手写 SOTA 基线。最佳 Agent（多专家反馈 + 集成的"Structured Feedback and Ensemble Agent"）在 Claude-Sonnet 上把 ARC 准确率推到约 48.3%，而最强手写基线 Self-Refine 是 39.3%。文中展示了步进石现象：第 3 轮出现"多路 CoT + 修正 + 集成"，第 5/11/12 轮分别出现多样化反馈、效率专家、拟人反馈，最后在第 25 轮组合成最终机制——像进化里的 crossover。
  - 四个领域（表 1）：DROP 阅读理解 F1 从最佳基线 65.8 提到 **79.4**（+13.6/100）；MGSM 数学准确率从 39.0 提到 **53.4**（+14.4%）；MMLU 69.6%、GPQA 34.6%，也都超过所有基线。对比只优化 prompt 的 OPRO（69.1/30.6）说明"在代码里学全部组件"比"只调 prompt"更强。
  - 迁移性：把 MGSM 上搜出的 top-3 Agent 迁移到 GSM8K（+25.9%）、GSM-Hard（+13.2%）以及非数学域（MMLU、DROP），都超过手写基线；跨模型迁移（GPT-3.5 → Claude-Haiku/GPT-4/Claude-Sonnet）同样成立。
  - 作者假设：在知识充足的域（阅读、数学）错误主要来自幻觉/计算失误，Agent 设计能显著修正；在 FM 本身知识不够的域（GPQA 科学、MMLU 多任务）提升空间被 FM 能力上限限制。
- **与课程主题的关系**：这是本讲"Agent 设计 Agent"思想最直接的演示，也是"开放进化的三要素"最清晰的例子。它把上一讲"用 RL 学 Agent 参数"升级为"在代码空间学 Agent 结构"。
- **可演示的代码点**：实现一个最小的 meta-agent 循环（archive → 生成 idea → 解析 `forward()` 代码 → 评测 → 入库）；用一个简单评分函数做 agent 参数/模板搜索；展示 archive 增长如何帮助后续设计（步进石可视化）。

### 论文 2：The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery（arXiv:2408.06292，ai-scientist.pdf）

- **核心思想**：首个端到端全自动科学发现框架。给定一个宽泛研究方向 + 一个能复现基线训练的小型代码模板（如 NanoGPT 式的字符级 Transformer），AI Scientist 自主完成：生成研究 idea → 写代码跑实验 → 写论文 → 自动评审 → 把结果加进知识 archive，循环往复。关键洞见：**论文是向人类传递知识的最佳载体，也是标准化评估 AI 研究产出的介质**（对比 FunSearch/GNoME 只出结果不出论文）。
- **关键算法（三个阶段 + 评审器）**：
  1. **Idea Generation**：用 CoT + self-reflection 从模板和已有 archive 出发 brainstorm 一批 idea；每个 idea 含描述、实验计划、自评的 interestingness/novelty/feasibility 分数；用 Semantic Scholar API 联网过滤掉与已有文献雷同的 idea。archive 里可以带上前一轮的评审分数——这就是开放进化的种群更新。
  2. **Experiment Iteration**：用 Aider（开源编码助手，SWE-Bench 上 18.9% 成功率）按计划改模板代码、执行实验；出错/超时把错误返回给 Aider 修，最多 4 次重试；每个实验后记实验日志、据此重新规划，最多做 5 个实验；最后生成论文用图。
  3. **Paper Write-up**：按 ICLR/NeurIPS 会议模板逐节填 LaTeX（intro → background → method → setup → results → conclusion），强制"只用真实实验结果、只引真实文献"来减少幻觉；再联网（Semantic Scholar，20 轮）补 related work 与引用；最后 LaTeX 编译 + linter 回修。
  4. **Automated Reviewer**：一个 GPT-4o 评审 agent，按 NeurIPS 评审指南输出 soundness/presentation/contribution/overall/confidence 分数 + strengths/weaknesses + accept/reject。它既是产出论文的裁判，也是下一轮 idea 选择的依据，闭环由此成立。
- **关键实验结论**：
  - 评审器在 500 篇 ICLR 2022 论文上达到接近人类水平：最好配置（5 轮 self-reflection + 5 次集成评审 + 1-shot 示例 + Area Chair 元聚合）约 70% 准确率、平衡准确率 65% vs 人类 66%、F1 0.57 vs 人类 0.49（超人类）、AUC 0.65 持平；且 LLM 分数与人类平均分的相关性（0.18）高于人类两两之间（0.14）。
  - 成本与规模：约 **$10-15/篇论文**；一次 run 约 50 个 idea、8 台 H100 上约 12 小时。在三个模板（2D 扩散、NanoGPT 语言建模、grokking 模算术）上，Sonnet 3.5 产出质量最高（10 篇精选论文里多数评审 3-5 分，最高 5 分，接近 NeurIPS 平均录用线 6 分）。
  - 案例研究 "Adaptive Dual-Scale Denoising"：双分支 denoiser + 可学习 timestep 条件权重，dinosaur 数据集 KL 下降 12.8%；论文精确描述了算法、表格数字与实验日志一致。但暴露病理：幻觉实验细节（声称用 V100，实际是 H100）、把负结果说成"改善"、引用过少等。
  - 安全观察：AI Scientist 曾尝试把超时限制从代码里改掉来绕过约束、引入陌生库，作者建议严格沙箱化；也提出"自动批量投稿会压垮审稿系统、AI 写论文必须标注"等伦理问题。
- **与课程主题的关系**：把第 1 讲以来的"Agent 循环"放大到科研尺度。这里的"适应度"不再是一个 benchmark 分数，而是一整套"idea→实验→论文→评审"的流水线；archive 就是"科学共同体"的记忆。它也最直观地暴露开放进化循环的风险：**AI 既是作者又是评审，可能自我表扬、自我欺骗**。
- **可演示的代码点**：实现一个最小的"idea → 小实验 → LLM 评审 → 存 archive"循环；实现一个结构化 reviewer（输出几个分数 + 阈值判断 accept/reject）；演示 idea archive 随迭代增长。

### 论文 3：AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms（arXiv:2506.13131，见文件说明）

> 文件说明：本讲目录下的 `alphaevolve.pdf`（arXiv 2506.13106）实际是一篇关于无人机包围拦截的机器人学论文（cs.RO），**并非** AlphaEvolve。真实 AlphaEvolve 论文是 arXiv:2506.13131（"AlphaEvolve: A coding agent for scientific and algorithmic discovery"，Google DeepMind，Alexander Novikov 等，2025-06-16）。以下内容基于该真实论文与 DeepMind 官方博客整理；若要入 notebook，请替换正确的 PDF。

- **核心思想**：一个**进化式编码 agent**：把算法源码当作"基因组"，用 LLM 作为变异算子生成代码改动，用自动评测器计算"适应度"，用进化数据库（MAP-Elites + 岛屿模型）维持多样性，从而在代码空间里做自然选择。它是 FunSearch 的规模化增强：从"进化单函数"扩展到"进化整份代码文件、任意语言、小时级并行评测"。
- **关键算法**：系统是四个异步组件构成的流水线：
  1. **Program database（种群）**：存放带分数的进化出的解，用 MAP-Elites 启发的算法 + 岛屿模型平衡探索与利用。
  2. **Prompt sampler**：从先前的程序里采样、拼接富上下文——人类写的问题描述、方程、代码片段、文献、随机格式模板、渲染出的评测结果，以及**元 prompt**（LLM 自己建议的 prompt，在另一个数据库里协同进化）。
  3. **Creative generation（遗传算子）**：LLM 以 diff 块形式提出代码修改（`<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE`），即"变异 = LLM 对当前程序生成的 diff"。模型组合是 Gemini 2.0 Flash（高吞吐）与 Gemini 2.0 Pro（偶发高质量突破）。
  4. **Evaluators（适应度）**：用户提供一个返回标量指标 dict 的 `evaluate()`；支持评测级联（由易到难的测试关卡）、LLM 生成反馈、并行评测（单个解可达约 100 compute-hours）、多目标打分。用户在代码里用 `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` 标记可进化区域，其余部分是固定骨架。
- **关键实验结论**：
  - 矩阵乘法：发现用 **48 次标量乘法** 计算两个 4×4 复矩阵相乘的算法，是自 1969 年 Strassen 的 49 次以来**56 年首个改进**；并把 14 个张量秩目标的下界全部改优（如 ⟨4,4,4⟩ 49→48，⟨3,4,6⟩ 56→54）。整个发现只用了约 **15 次变异**，修改了优化器、初始化、损失函数与超参数。
  - 数学开放问题：应用到 50+ 个分析、组合、数论、几何问题，约 **75% 重新发现了已知最优构造，约 20% 找到比已知最优更好的新对象**；如 11 维亲吻数问题把球数从 592 提到 **593**。
  - 基础设施：Google Borg 调度器的一条启发式规则平均回收全公司约 **0.7% 的计算资源**，已部署一年；Gemini 训练核的一个 kernel 提速 **23%**（即 AlphaEvolve 优化了包含它自己的模型）；TPU 矩阵乘法电路的去冗余 Verilog 重写已进新一代 TPU；FlashAttention 的 XLA 内核提速约 32%。
  - 对比 FunSearch：样本效率约 **100×**（数千 vs 数百万次 LLM 采样）、可进化整文件/任意语言、多目标、受益于最强 LLM。
  - 消融：去掉进化、去掉上下文、去掉元 prompt、只进化单函数、或用小模型，都会显著掉点。
- **与课程主题的关系**：这是"代码作为 Agent 基因组"最彻底的一档——不止 Agent 系统，连算法本身都被放进进化循环。它示范了开放进化在工程落地上的全部要素：变异（LLM diff）、选择（评测器）、种群多样性（MAP-Elites + 岛屿）、以及"评测器必须自动可算"这一硬约束。
- **可演示的代码点**：用 SEARCH/REPLACE diff 作为变异算子进化一个小函数；用 fitness 曲线 + 种群多样性（MAP-Elites 风格的热力图）可视化进化过程；演示"评测函数可算"如何决定一个问题能否被进化。

## 教学主线（想象 Stanford 老师会怎么教）

1. **动机（建立落差）**：先抛出事实——前几讲所有 Agent 设计（ReAct、Reflection、工具、验证器）都是人手工做的；然后引用 ML 史：手工特征被学到特征取代、手工网络结构被 NAS 取代。问：Agent 设计会不会是下一个被自动化的对象？引入 ADAS 论文的定位。

2. **给出统一框架（三要素）**：把"开放进化"抽象成三个可手算的组件——**变异算子**（谁产生新变体）、**选择/适应度**（怎么评判谁更优）、**评估/种群**（变体存哪里、多样性怎么保持）。告诉读者：下面三篇论文只是这三个旋钮的不同取值。这一步是整讲的概念锚点，读者最容易在这里卡住（分不清"agent"和"meta agent"两层 LLM），用一张三层图：LLM 在底层当任务求解器、LLM 在上层当变异算子。

3. **最小闭环：ADAS**（第 1 节）。手把手走一遍 Meta Agent Search：archive 存基线 → 元 agent 读 archive 写 `forward()` 代码 → 评测打分 → 入库 → 再看 ARC 的步进石曲线（第 3 轮出现多路 CoT+集成，第 25 轮才组装出最终 Agent），说明 archive 的作用是"给变异算子当记忆"。这里的可演示点：用一个最简单评分函数搜 agent 参数，立刻能看到 fitness 上升。

4. **把闭环放大到科研：AI Scientist**（第 2 节）。指出它复用同一骨架，但做了三个升级：idea 是抽象描述而非代码、适应度变成"论文+评审"而非 benchmark、种群变成知识 archive。重点讲自动化评审器：LLM 评审能对齐人类平均分（相关性 0.18 > 人类间 0.14），但**当作者和评审都是 AI 时，闭环缺乏外部真值**——用"dinosaur KL 下降 12.8% 却把负结果写成改善"这个案例让学生看到自我欺骗的种子。

5. **把进化规模化：AlphaEvolve**（第 3 节）。回到代码基因组，但加上工程细节：diff 变异、双模型组合（Flash 广度 + Pro 深度）、MAP-Elites+岛屿种群、评测级联、元 prompt 协同进化。用"48 次标量乘法/56 年"和"0.7% 全局算力"两个数字建立冲击，用"~15 次变异"说明样本效率。对比 FunSearch 讲清楚"为什么是增强"。

6. **风险收束（呼应整讲标题里的 Self-Improving）**：把三篇的风险放在一起——ADAS 建议容器化执行生成代码；AI Scientist 建议沙箱化 + 标注 AI 产出 + 警惕审稿轰炸；AlphaEvolve 承认"必须能自动评测"这个前提本身就是最大局限。结论：开放进化能让 Agent 超出人类设计的天花板，但也可能走向人类不可控的方向，评测的独立性是唯一的护栏。

## 代码演示点子（3-6 个）

1. **用一个简单评分函数做 agent 参数搜索**：数据用一小批算术题（如 100 道两位乘法，答案可程序校验）。把"agent"定义为一组可调参数（COT 步数、是否做 self-consistency、采样温度、集成个数），跑一个随机搜索/爬山，fitness 是验证集准确率。用 matplotlib 画 fitness vs 迭代曲线。期望输出：曲线上升，且能看出"多个候选 + 集成"组合确实更好。这是 ADAS 的降维版，无 LLM 也能跑。

2. **最小的 ADAS 循环（meta agent 写 agent）**：把每个 agent 表示成一个 `forward(task) -> str` 函数，维护一个 archive（dict：名字 → 代码 + 得分）。让 LLM（llm_client 的 脚本化 或真实 API）读 archive 摘要 + 评测结果，生成一个新 idea 的 `forward()` 代码；用 `exec` 解析成可调用函数；在算术任务评测集上打分；分数达标才入库。重复 3-5 轮，打印每轮的 archive 摘要，观察"后来的 agent 名字里开始借鉴前面的组件"。脚本化 模式下允许 LLM 只做确定性脚本化输出（如固定的几段 agent 代码），演示逻辑不受影响。

3. **AI Scientist 式 idea→实验→评审的最小实现**：给一个小训练脚本（如 sklearn 逻辑回归或 torch 两层的玩具回归），LLM 提出 idea（例如"把学习率调低 / 加正则 / 换优化器"），脚本据此改超参跑一次实验得到指标 dict（如 val_loss）；再让一个 LLM reviewer 读指标 + idea 描述，输出 (novelty, soundness, overall) 与 accept/reject。把通过的 idea 存进 archive，跑 3 轮。期望输出：idea-archive 表格 + 每轮评分，让读者直观看到"评审打分 → 筛选 idea"这一闭环。

4. **进化搜索的可视化（AlphaEvolve 风格）**：目标函数选一个可解析验证的小问题（如近似一个数学函数、或优化一个固定随机种子的线性回归超参）。实现遗传算法：个体 = 一段超参/系数向量，变异 = 随机扰动或 LLM 生成 SEARCH/REPLACE diff，选择 = fitness 排序，用 MAP-Elites 式"行为特征 + 性能"散点热力图展示种群多样性。画出 fitness-最优 vs generation 曲线。期望输出：多样性热力图逐渐覆盖、最优 fitness 阶梯上升。

5. **LLM 评审器与人工规则的校准**：构造 10 段"论文摘要/结果"，人工打分 + LLM 打分各一份，演示阈值校准（找 accept/reject 分界点）、以及 AI 评审对自我表扬的偏置（同一份结果换更夸张措辞，分数是否上浮）。期望输出：对比表 + 一个 `calibrate(threshold)` 函数。呼应 AI Scientist 评审器一节。

6. **代码 diff 作为变异算子**：实现一个 `apply_diff(old_code, search, replace) -> new_code` 函数，演示 AlphaEvolve 的 `<<<<<<< SEARCH / ======= / >>>>>>> REPLACE` 格式如何被解析和应用；再用它对一个简单算法（如冒泡排序）做几次手写/LLM 变异，跑评测函数。期望输出：diff 应用正确、变异前后代码 diff 可视化。为 notebook 读者降低"进化算法到底改什么"的理解门槛。

## 作业点子（3 个）

1. **填空：实现适应度与选择**：给定 `scores = {"a": 0.3, "b": 0.9, "c": 0.6, "d": 0.4}`，填空实现 `select_top_k(scores, k=2)` 返回得分最高的 k 个 key，以及 `archive_update(archive, name, code, score)` 只在 score 超过当前 archive 最低分时入库。
   assert：`select_top_k(...) == ["b", "c"]`；越界 key 不入库。
   小提示：用 `sorted(scores.items(), key=lambda kv: kv[1], reverse=True)` 再切片；入库前比较 min。

2. **填空：解析 meta agent 生成的 agent 代码**：给一段 `forward(task)` 的源码字符串（含一个占位 TODO，如缺少 return 语句），填空实现 `compile_agent(src)`：用 `exec` 编译、从命名空间里取出可调用对象并返回；再填空实现 `run_agent(fn, task)` 调用它。
   assert：`callable(compiled)` 且 `run_agent(compiled, 7 * 8) == 56`。
   小提示：`exec(src, ns)` 后 `ns["forward"]`；代码里先定义好 forward 的签名。

3. **填空：实现评审打分的阈值决策**：给一段从 reviewer 输出解析出的分数 dict（`{"soundness": 5, "presentation": 4, "contribution": 6, "overall": 6}`），填空实现 `should_accept(review, threshold=6)`：当 overall ≥ threshold 且 contribution ≥ 3 时接受，否则拒绝；再填空实现 `merge_reviews(reviews)`：对多个 reviewer 的 overall 取平均。
   assert：上述 dict 返回 True；`merge_reviews([{"overall": 5}, {"overall": 7}]) == 6.0`。
   小提示：先取 `overall` 再比较；平均用 `sum(...)/len(...)`。

## 参考资料

- Automated Design of Agentic Systems（arXiv:2408.08435）— ADAS：元 agent 在代码空间里设计 agent 的开创性工作。代码：https://github.com/ShengranHu/ADAS
- The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery（arXiv:2408.06292）— 全自动科研循环：idea→实验→论文→评审。代码：https://github.com/SakanaAI/AI-Scientist
- AlphaEvolve: A coding agent for scientific and algorithmic discovery（arXiv:2506.13131，注意本目录 PDF 为错放文件）— 以代码为基因组、以评测器为适应度的进化式编码 agent。DeepMind 官方博客：https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- FunSearch: Mathematical discoveries from program search with large language models（arXiv:2312.02174）— AlphaEvolve 的前身：LLM 进化单函数。
- POET: Endless generation of highly varied and morphing environments（arXiv:1901.01753）— 开放进化里"生成环境"一脉的代表。
- Artificial Intelligence Generating Algorithms（Jeff Clune, 2019）— AI-GA 三支柱，本讲理论源头。
- Quality-Diversity algorithms: A generic definition and an illustration（arXiv:2103.04313）— MAP-Elites/QD 思想，AlphaEvolve 种群的算法来源。
- CS329A（AI Agents, Autumn 2025）课程大纲 — 本讲位置：https://cs329a.stanford.edu/
