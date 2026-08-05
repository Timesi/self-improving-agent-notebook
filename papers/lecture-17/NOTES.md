# Lecture 17 — Agent 评测与长程任务 研读笔记

> 本文件是 CS329A 第 17 讲的论文研读笔记，是编写对应 notebook 的素材。
> 来源：https://cs329a.stanford.edu/（Autumn 2025 课程大纲）

## 课程主题

这一讲回答的问题：**怎样评测一个 Agent 才算靠谱？** 前面各讲教了怎么把 LLM 做成 Agent（循环、工具、规划、记忆、SWE 智能体），这一讲转向"度量学"：Agent 做的是多步、开放、时长不一的任务，传统 benchmark 的"选择题分数"既会快速饱和、又无法换算成对人类的价值。核心矛盾是：任务越真实，就越难自动打分；打分越省事（LLM judge），就越不可靠。

全讲按三条线索层层递进：
1. **从能力到任务时长**（METR）：用"人类完成该任务所需的时间"来给 Agent 能力标定标尺，得到可跨模型、跨年份比较的连续指标（time horizon）。
2. **从任务时长到经济价值**（GDPval）：时长还不够，还要看"做的活儿值多少钱"——用真实职场任务、盲评 win rate 对比行业专家。
3. **自动评测的可靠性与漏洞**（DeepScholar-Bench）：大规模评测必须自动打分（LLM judge），但 LLM judge 有位置偏差、自我偏好等漏洞，需要和人类打分校准。

三篇合在一起回答"评测 Agent 为什么难"：因为任务分布不再固定、成功判据不唯一、自动评分器的可靠性本身就是待测对象。

## 论文精读

### 论文 1：Measuring AI Ability to Complete Long Software Tasks（arxiv:2503.14499，measuring-long-tasks.pdf）
- **核心思想**：现有 benchmark 只能给出"在某个固定题库上的得分"，无法回答"这个模型到底能做多复杂的真实工作"。METR 提出把 AI 能力翻译成人类时间单位：**X% 任务完成时间视野（time horizon）**——"AI 能以 X% 成功率完成、而人类专家通常需要 t 小时的任务"的 t。测量方法是：构造一套覆盖极宽难度（秒级到 8 小时）的 170 个软件/ML 任务 → 让人类专家和 AI Agent 分别做 → 记录人类耗时与 AI 成功率 → 拟合出每个模型对应 50% 成功率的时间视野 → 再按模型发布日期画成长曲线。核心发现：**前沿模型的 50% 时间视野自 2019 年起约每 7 个月翻一倍**（GPT-2 只有 2 秒，o3 约 110 分钟），且 2024 年后可能还在加速。按此趋势外推，AI 在 2028 年中到 2030 年中之间可能达到"一个月时长任务"的 50% 时间视野。
- **关键公式/算法**：
  - 成功概率的 logistic 模型（思想来自心理测量学 Item Response Theory 的 2PL，但难度直接用人类基线时长而非学出来的参数）：
    $$p_{\text{success}}(\text{agent},\text{task}) = \sigma\!\big((\log h_{\text{agent}} - \log t_{\text{task}})\cdot \beta_{\text{agent}}\big)$$
    其中 $t_{\text{task}}$ 为成功人类基线时长的几何平均，$h_{\text{agent}}$ 即 50% 时间视野，$\beta_{\text{agent}}$ 为斜率（每个 agent 一个参数）。对每个模型单独做 logistic 回归，任务按所属 family 用 $1/\sqrt{n}$ 加权（family 内任务相关性强，需降权保证多样性）。
  - 趋势线：对 $\log(\text{horizon})$ 与 release date 做 OLS，斜率换算成翻倍天数；误差用三层层次 bootstrap（family→task→run，10000 次）估计。
  - 任务套件：HCAST 97 个任务（1 分钟到 30 小时，46 个 family）+ RE-Bench 7 个 8 小时 ML 研究工程任务 + 新增 SW AA（Software Atomic Actions）66 个 1~30 秒的单步小任务（文件选择、告警分诊、请求路由、代码补全、心算），用于测 GPT-2/GPT-3 这类旧模型。
- **关键实验结论**：
  - 12 个前沿 + 4 个近前沿模型（2019–2025），每个 agent/task 组合跑约 8 次。50% 时间视野翻倍时间 = 207 天（95% CI 166–240 天，约 ±19%）；80% 时间视野约短 4–6 倍，翻倍时间相近（204 天）——说明模型"偶尔成功长任务"和"稳定胜任中长任务"差距还很大。
  - o3 高于长期趋势线（p=0.006），提示 2024 年后增速更快；单用 2024–2025 数据外推，一个月视野最早可到 2027 年初。
  - 外部效度验证：SWE-bench Verified 上同样的指数趋势成立（翻倍时间约 70 天，比主套件预测的 143 天更陡）；内部真实 PR 实验发现模型表现与"低上下文外包人员"耗时一致、与"仓库维护者"耗时不一致（维护者比外包人员快 5–18 倍）。
  - 失败模式归因：GPT-4 的失败 12/31 是"重复失败动作"，o1 只有 2/32；o1 的失败一半是"过早放弃任务"——能力提升主要来自更会工具使用、更能从错误中恢复、逻辑更强。
  - "messiness"：给任务打 16 个"真实世界的脏度"因子（动态环境、不可逆失误、资源受限、需要主动搜寻信息等），均值 3.2/16；控制时长后，messiness 每 +1，成功率约降 8.1%。但高/低 messiness 子集的时间趋势接近——脏任务绝对分低，但进步速度没有变慢。
  - 成本：按 $143.61/小时（Google L4 工程师平均时薪折算）对比，超过 80% 的成功 run 成本不到人类做同样任务的 10%。
- **与课程主题的关系**：这一篇是全讲的"标尺"。它给出把任意 benchmark 结果换算成"人类时间"的方法论（human baseline + logistic 拟合），让 GPT-2 和 o3 能放在同一条曲线上比较，也是后续 GDPval（用时间×时薪换算经济价值）和 DeepScholar（长任务自动评分）的动机来源。
- **可演示的代码点**：
  - 从零实现 time horizon：造一个"任务时长 vs 成功率"的合成数据集，numpy 手写 logistic 回归（或用 scipy.optimize）拟合 $h_{\text{agent}}$ 和 $\beta_{\text{agent}}$，验证 $h_{\text{agent}}=t_{\text{task}}$ 时成功率恰为 0.5。
  - 画"时间视野-发布年份"对数图：对 $\log h$ 做 OLS，输出翻倍天数，复现"每 7 个月翻倍"。
  - 模拟层次 bootstrap 误差条（对 family/task/run 三层重采样）。
  - 演示 50% vs 80% 视野的差距（不同成功率阈值下拟合）。

### 论文 2：GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks（arxiv:2510.04374，gdpval.pdf）
- **核心思想**：METR 用"时长"标度能力，但时长不等于经济价值——一句"7 小时的工作"没说明这工作值不值钱、做得好不好。GDPval（OpenAI）直接测**真实经济价值任务**：从美国对 GDP 贡献最大的 9 大行业中，选 44 个高薪、以数字化为主的职业，让平均 14 年经验的行业专家把他们真实的工作产物改造成任务（每个任务 = 一份 request + 一份 deliverable）。主指标是**盲评 win rate**：由该职业的专家成对比较"模型产出 vs 人类专家产出"，判定谁更好/打平。核心结论：**前沿模型质量已逼近行业专家**（最优模型 Claude Opus 4.1 有 47.6% 的产出达到"优于或打平人类"），且 OpenAI 模型性能随时间**近似线性**提升；但若把专家审阅/返工的时间算进去，纯速度优势大幅缩水。开源 220 个 gold 任务并给出实验性自动评分服务（evals.openai.com）。
- **关键公式/算法**：
  - win rate：盲评成对比较，评分取值 {0, 0.5, 1}（模型赢/平/输）。
  - 自动化评分一致性（也是"评测评测者"的公式）：
    - 人-机一致率 $A^{\text{HA}}_s = E[\,1 - |H - A|\,]$；人类间一致率 $A^{\text{HH}}_s = E[\,1 - |H_1 - H_2|\,]$（对同一样本的两份人类评分取均值）。
  - 经济价值换算：任务美元价值 = 平均完成时长 × 该职业时薪中位数（OEWS 数据）。
  - 速度/成本收益模型（"先用模型，不满意自己修"）：设 $w_i$ 为模型在任务 $i$ 上的 win rate，$M_T/M_C$ 为模型耗时/成本，$R_T/R_C$ 为专家审阅耗时/成本，$H_T/H_C$ 为人类完成耗时/成本：
    - try-1 次：$E[T_{1,i}] = M_{T,i} + R_{T,i} + (1-w_i)\,H_{T,i}$
    - try-n 次：$E[T_{n,i}] = (M_{T,i}+R_{T,i})\,\frac{1-(1-w_i)^n}{w_i} + (1-w_i)^n\,H_{T,i}$；当 $n\to\infty$ 时人类时间被 $(M_T+R_T)/w$ 取代。
  - 任务构建流水线：从 9 大行业（占 GDP>5%，Q2 2024）选每个行业贡献薪酬最高的 5 个数字化职业（用 GPT-4o 按 O*NET 任务标注"是否数字化"，阈值 60%）；专家平均 14 年经验、需面试+背调+培训+测验，入选率 <10%；每任务平均 5 轮人类评审。
- **关键实验结论**：
  - 规模：full set 1,320 个任务（每职业 30 个），gold 子集 220 个（每职业 5 个，69% 任务需操作参考文件，最多 17 个参考文件）。gold 任务平均耗时 9.49 小时（中位 5 小时），平均美元价值 $398；任务几乎都是多模态交付物（CAD、PDF、PPTX、表格、音视频）。
  - 头部 win rate：gpt-4o 12.5% → o4-mini 29.1% → o3 35.2% → gpt-5 39.0%，Claude Opus 4.1 最优约 47.6%（win+平手）——质量逼近但未超过人类。
  - 速度/成本：naive 比值高达 90–327 倍；计入"审阅+失败返工"后（try-1），gpt-4o 反而只有 0.87 倍（更慢）、gpt-5 约 1.12 倍；try-n 后 gpt-5 约 1.39 倍。**结论：报告里 327x 的 naive 加速是被高估的，真实增益主要来自"人在回路"。**
  - 自动评分器（GPT-5-high）与人类一致率 66%，人类间一致率 71%——只差 5 个百分点；但对强模型的产出一致率更低（模型偏好自己的产出）。12/220 个任务被标记为"不可自动评分"（需联网、需非 Python 运行、字体渲染、语音转写等）。
  - 提示工程+脚手架（best-of-4 + GPT-5 judge、鼓励多模态自查）：win rate +5pp，PPT 格式错误 86%→64%，多模态自检率 15%→97%；推理 effort 越高越好。
  - 弱上下文版（prompt 缩短到 42% 长度）：模型明显变差——真实工作是"弄清要做什么"，缺上下文是最难的部分。
- **与课程主题的关系**：从 METR 的"时长标尺"升级为"经济价值 + 质量"双标尺，并示范了自动评测的边界：真专家盲评贵（单次对比 >1 小时）但可靠，自动评分便宜但只到 66% 一致率。它同时是"LLM judge 可靠性"讨论的靶子（第 3 篇论文继续深化）。
- **可演示的代码点**：
  - 从零实现 win rate 统计与一致率：给定合成评分数据，算 win rate、$A^{\text{HA}}$、$A^{\text{HH}}$，比较"自动评分器"和"人类"。
  - 实现 try-n-then-fix 的期望成本公式并画曲线：看 win rate 多高时模型协助才划算（交叉点分析）。
  - 演示盲评打分流程：用 mock LLM 生成两个交付物描述，让"人类 judge"（脚本或 mock）成对打分，理解 0/0.5/1 三值评分。

### 论文 3：DeepScholar-Bench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis（arxiv:2508.20033，deepscholar-bench.pdf）
- **核心思想**：评测"深度研究/生成式研究综述"这类长程 Agent 的新 benchmark。真实研究综述任务是"开放式的、正确性没有唯一标准、依赖实时网络检索"的，而既有 QA benchmark 只测短答案、专家手工标注集又会过时和被污染。DeepScholar-Bench 用**持续更新的数据管道**从近期高质量 arXiv 论文自动生成 query（任务：给定论文标题+摘要，生成 related work 一节，人类作者写的原文就是 exemplar），并提出**三维 7 指标的全自动评测**：知识综合（organization & coherency、nugget coverage）、检索质量（relevance rate、reference coverage、document importance）、可验证性（citation precision、claim coverage）。全部指标用 LLM judge 实现，并与人类评分校准。结论：**所有系统几何平均都低于 31%，benchmark 远未饱和**，连 OpenAI DeepResearch 的最佳指标（nugget coverage 39.2%）离饱和也很远。
- **关键公式/算法**：
  - 任务形式化：给定论文描述 $d$，检索源集合 $S$，生成综述 $W$，与人类 exemplar 对比。
  - Relevance Rate：LLM judge 给每个被引源打相关性 $Rel(s)\in\{0,1,2\}$，$RR(S)=\frac{1}{2|S|}\sum_{s\in S}Rel(s)$。
  - Reference Coverage：先对 exemplar 的引文标"important/not"，$RC(S,E)=\frac{1}{|E|}\sum_{s\in S}\mathbb{I}[s\in E]$（对重要引文的召回率）。
  - Document Importance：$DI(S,S^*)=\min\!\big(\frac{\text{median cites}(S)}{\text{median cites}(S^*)},1\big)$，衡量引文"含金量"（被引次数中位数对比人类 exemplar）。
  - Nugget Coverage：从人类 exemplar 抽取信息 nugget（essential fact），算生成报告覆盖了多少比例（沿用 Great Nugget Recall 的 LLM 方法）。
  - Organization & Coherency：LLM-as-judge 成对比较生成报告 vs 人类 exemplar，交换顺序两次取平均以消除位置偏差，报告 win rate。
  - Verifiability：Citation Precision（句内引文是否真的支撑该句某条主张）、Claim Coverage（句子主张是否被其引文/滑窗 $w$ 内引文完全支撑）。
  - 数据管道：限定 arXiv 发表日期区间（避开 Llama-4 训练截止 2025-04-05 以防污染）、只保留 v1、只取会议已接收论文、有显式 Related Work 节和 .bib。
- **关键实验结论**：
  - DeepScholar-June-2025：63 篇论文、18 个 arXiv 领域，平均每篇 related work 有 23 条不重复引文，63% 引文在 arXiv 上。DeepScholar-Nov-2025 扩展到 200 个 query、75+ 学科。
  - 主结果（几何平均）：OpenAI DeepResearch 最高 0.309；OpenAI o3 search agent 0.287；DeepScholar-ref (GPT-4.1, Claude) 0.286；所有系统 <0.31。DeepResearch 在 Nugget Coverage（.392）、Reference Coverage（.187）、Document Importance（.124）上都仍 <0.40——"能组织好文笔（Org .857），但漏掉关键事实、引不到重要文献"。
  - DeepScholar-ref（LOTUS 语义算子：semantic filter → semantic top-k → semantic aggregation）：开源强基线，可验证性比 DeepResearch 高最多 6.3 倍、便宜 4.3 倍、快 2.28 倍。
  - Oracle 消融：喂给系统"正确答案应引的重要文献"（oracle retriever）后 Reference Coverage 到 1.0，但 Nugget Coverage 也只到 ~0.49——**瓶颈主要不在检索，而在"把好材料合成成关键事实"**。
  - LLM judge 校准（11 名 CS 博士、300+ 条标注）：Organization 成对比较与人类一致率 71.43%，nugget 标注 83.33%，reference importance 65.9%——LLM judge 在长文档任务上可用但仍非完美。
- **与课程主题的关系**：这是"自动评测可靠性与漏洞"的正面示范：用 LLM judge 支撑 7 个指标，却必须一一与人类校准、考虑位置偏差；同时"live benchmark + 防污染"是针对"benchmark 饱和/数据泄漏"问题的工程化答案，呼应 METR 对 benchmark 饱和的批评。它也是第 5 节"构建自己的评测 harness"的模板。
- **可演示的代码点**：
  - 从零实现几个指标：给定一组 mock 引文（含被引次数、相关度标签、claims），算 RR、RC、DI，理解各指标对"好的综述"不同侧面的刻画。
  - 实现 nugget coverage：用 mock LLM 从 exemplar 提取 nugget，再判断生成报告是否覆盖。
  - LLM judge 位置偏差演示：同一对答案正反两序各评一次，统计不一致率，再看"swap 平均"如何消除偏差。
  - 几何平均聚合：解释为什么"用几何平均而不用算术平均"（任何维度得 0 都会拖垮总分，不允许单项摆烂）。

## 教学主线（想象 Stanford 老师会怎么教）

1. **先立动机：现有 benchmark 到底在骗你什么。** 抛出例子：同一个模型在 MMLU 接近满分、在 SWE-bench 却很吃力；HellaSwag、Humanity's Last Exam 是"反向筛选"出来专难模型的题。给读者一个直觉锚点——**"在选择题上 90 分"无法翻译成"能做完几小时的工作"**。由此引入三篇共同的敌人：benchmark 饱和、无跨模型统一尺度、任务不真实。
2. **第一篇 METR 建立"标尺"：把能力换成人类时间。** 老师会先讲一个直观类比：就像用"人类跑完要多久"来度量赛道的难度。然后拆解三步：构造秒级到 8 小时的任务套件 → 找 800+ 人类基线测时长（几何平均）→ 对每个模型做 logistic 拟合得到 50% time horizon。亲手带读者算一遍 logistic：为什么 $h_{\text{agent}}=t_{\text{task}}$ 时成功率是 0.5。最后亮出那张"每 7 个月翻一倍"的对数曲线，并强调**斜率比单个模型的绝对高度更可信**（误差高度相关）。卡点提示：读者容易把"任务时长"误当"模型运行时长"；要强调时长是人做的、成功率是 AI 做的，两者通过模型拼起来。
3. **从时间到钱：GDPval 换一个标尺。** 承接："METR 告诉你活儿有多长，没告诉你活儿值多少钱、做得像不像样。" 讲 GDPval 如何自顶向下选职业（9 大行业 × 每行业薪酬前 5 的职业）、如何由专家造任务并给任务标"美元价值 = 时长 × 时薪"。核心演示是**盲评 win rate** 的机制：成对、匿名、专家判 0/0.5/1。在这里老师会埋一个伏笔：真专家打分太贵（一次对比 >1 小时），于是他们训练了一个自动评分器——只有 66% 和人类一致，这自然引出第三篇。
4. **自动评测的可靠性与漏洞：DeepScholar-Bench 收尾。** 讲"既然人太贵，就让 LLM 打分"，但三个坑逐个示范：LLM judge 有位置偏差（所以要成对交换顺序取平均）、模型偏好自己的产出（GDPval 里自动评分器对强模型一致率更低）、评测本身的"正确性"需要再评测（DeepScholar 拿 300+ 条人类标注校准 71%/83%/66% 的一致率）。最后用 DeepScholar 的三维 7 指标把"好的深度研究"拆解，用 oracle 实验点出"瓶颈在综合而非检索"。**收束全讲的一句话**：评测长程 Agent 难在四个环节各有一层误差——任务分布是否真实、成功判据是否唯一、自动评分器是否可靠、以及这些误差本身是否被校准过。
5. **落到动手：让读者自己写一个 eval harness。** 用 mock 环境把上面每个指标跑一遍（时间视野拟合、win rate、一致率、nugget coverage），体会"评测一个评测"和"评测一个 Agent"是同一件事。

## 代码演示点子（3-6 个）

1. **从零实现一个 Agent 评测 harness**：定义一个小任务池（如"改写函数 + 单测自动打分"），用 `llm_client` 的 mock 模式跑 N 次，统计成功率、逐步构造"任务池定义 → 运行 → 二元化 → 聚合"的最小闭环。期望输出：每个任务的 success/fail 表与平均成功率，让读者看到"评测 harness = 环境 + 评分函数 + 聚合统计"。
2. **长任务的时间-完成率曲线与 time horizon 拟合**：构造合成数据 `tasks = [(log_time, true_rate)]`，用 `scipy.optimize` 或手写梯度下降拟合 $p=\sigma((\log h-\log t)\cdot\beta)$，输出拟合的 $h$（50% 时间视野），并画出数据点+拟合曲线，标注"该模型 50% 成功率对应的人类时长"。再对多个"模型"（不同 $h$）做 OLS，画 $\log h$ vs 时间的直线，报翻倍天数。
3. **Win rate 与评分一致率计算**：给几组 {model, human, tie} 三值评分模拟数据，计算模型 win rate，再实现 $A^{\text{HA}}_s=E[1-|H-A|]$ 与 $A^{\text{HH}}_s=E[1-|H_1-H_2|]$，对比"自动评分器 vs 人类"和"人类 vs 人类"两条基线，复现 GDPval 的"66% vs 71%"结构。
4. **LLM judge 偏差演示**：让 mock/真实 LLM 对同一对答案做正反两序成对比较，统计位置偏差（正反不一致的比例）；再演示"swap 平均"如何把偏差对 win rate 的影响消掉，以及"自我偏好"（judge 更容易选与自己风格/模型同源的输出）。
5. **Try-n-then-fix 的成本-收益曲线**：用公式 $E[T_n]=(M_T+R_T)\frac{1-(1-w)^n}{w}+(1-w)^n H_T$，固定 $M_T,R_T,H_T$，扫 $w$（模型质量）和 $n$（尝试次数），画"总时间 vs win rate"曲线，找模型 win rate 超过多少、模型协助才比纯人工划算（交叉点），直观展示"naive 327x 为什么失真"。
6. **检索质量三维指标 + oracle 消融**：给一组 mock 引文（每条带 related/important/被引次数/对应 claim），实现 RR、RC、DI；再模拟"把 oracle 的重要文献直接喂给系统"，看 RC 拉满而 nugget coverage 没跟着涨，复现"瓶颈在综合而非检索"的结论。

## 作业点子（3 个）

1. **拟合 time horizon**：给一张"任务人类时长 + 各模型每任务成功率"的小表，要求用 scipy 拟合 logistic 求出某模型的 $h$ 和 $\beta$，再断言 `abs(h_est - 真值) < 容差`、以及 $h=t_{\text{task}}$ 时预测成功率约等于 0.5。小提示：先对时长取 log 再做 logistic 回归；目标函数是负对数似然。
2. **实现 win rate 与一致率**：给一份 {model_grade, human1_grade, human2_grade} 的评分表，要求计算该模型的 win rate、$A^{\text{HA}}$、$A^{\text{HH}}$，断言自动评分器"没人类彼此之间一致"且"对弱模型更一致"。小提示：$|H-A|$ 只在 H 与 A 都是 0/0.5/1 的三值时才直接算，先想清楚 tie 该记成什么。
3. **实现 nugget coverage**：给一个人类 exemplar 的 n 个 nugget 和一个 mock 生成报告的句子列表，用简单关键词/嵌入相似度或 mock LLM 判断每个 nugget 是否出现，算覆盖比例，断言"报告 B 覆盖更全所以分更高"。小提示：nugget 是"essential fact"级别的原子事实，别把整句话当 nugget。

## 参考资料

- METR. *Measuring AI Ability to Complete Long Software Tasks*（arxiv:2503.14499）— 提出"50% 时间视野"指标，发现前沿 Agent 长任务能力每 7 个月翻倍。
- OpenAI. *GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks*（arxiv:2510.04374）— 44 个职业 9 大行业的真实经济价值任务，盲评 win rate 逼近人类专家。
- Patel et al. *DeepScholar-Bench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis*（arxiv:2508.20033）— 三维 7 指标自动评测深度研究 Agent，无系统几何平均超 31%。
- Ngo, Richard. *Clarifying and Predicting AGI*（LessWrong，2023）— METR 外推所采用的"1 个月 AGI（167 小时）"定义来源。
- OpenAI. *Introducing SWE-bench Verified*（2024）— METR 用来做外部效度验证的行业标准软件工程 benchmark。
- OpenAI. *evals.openai.com* — GDPval gold 子集（220 任务）与实验性自动评分服务入口。
- DeepScholar-Bench 官方仓库：https://github.com/guestrin-lab/deepscholar-bench — live 数据管道与评测代码。
- Rein et al. *HCAST: Human-Calibrated Autonomy Software Tasks*（forthcoming，2025）— METR 任务套件的三分之一来源。
- Wijk et al. *RE-Bench: Evaluating Frontier AI R&D Capabilities of Language Model Agents Against Human Experts*（arxiv:2411.15114）— 8 小时 ML 研究工程任务与人类专家基线。
- Panickssery, Bowman, Feng. *LLM Evaluators Recognize and Favor Their Own Generations*（arxiv:2404.13076）— LLM judge 自我偏好现象，GDPval 自动评分器的解释引用。
