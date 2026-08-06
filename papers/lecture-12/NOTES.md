# Lecture 12 — LLM 推理（LLM Reasoning）研读笔记

> 本文件是 CS329A 第 12 讲的论文研读笔记，是编写对应 notebook 的素材。
> 嘉宾：Denny Zhou（Google DeepMind，三篇论文的共同作者）。
> 来源：https://cs329a.stanford.edu/（Autumn 2025 课程大纲）

## 课程主题

这一讲要回答的核心问题：**LLM 的"推理能力"从哪里来？如何在不改模型、不微调的前提下把它激发出来？**

三篇论文给出了一条自洽的答案链：
1. **能力存在**：大模型在足够规模下会"涌现"出小模型没有的推理能力（emergent abilities）。
2. **能力可激发**：靠 few-shot 提示格式的极小改动（把 exemplar 从 ⟨问题,答案⟩ 换成 ⟨问题,推理步骤,答案⟩），就能用提示词把这种能力"钩"出来（chain-of-thought prompting）。
3. **能力可聚合**：单次贪婪解码只走一条推理路径，容易局部最优；改为"多次采样 + 对最终答案多数投票"，用"自我集成"（self-consistency）进一步提升，还顺带得到不确定性估计。

为什么放在 Agent 课程的这个位置：Agent 的一切都建立在模型的能力上——多步任务分解、推理中间步骤生成、对自己答案的置信度判断（"know when it doesn't know"）。这一讲是在为后面的 Agent 构建（planning、tool use、反思、自校正）打下"推理引擎"的理论与直觉基础。三篇论文的主角（Denny Zhou）也把推理与程序合成、Least-to-most、思维树（ToT）等后续工作直接串起来。

## 论文精读

### 论文 1：Chain-of-Thought Prompting Elicits Reasoning in Large Language Models（arxiv:2201.11903，cot.pdf）
- **核心思想**：给大模型 few-shot 的 exemplar 时，把"输入→输出"对改成"输入→**中间推理步骤**（chain of thought）→输出"三元组，模型就能解决它原来解不出的多步推理任务。核心观点有三层：(1) 中间步骤让模型把多步问题分解，把"计算"分配到更难的题上；(2) 推理步骤提供了可解释窗口，便于排查错在哪一步；(3) **这是提示格式的功劳而非训练**——全程没有微调任何模型。论文自称"标准 prompting 只是模型能力的下界"。
- **关键公式/算法**：核心是提示格式，无训练目标。格式为 `Q: <题目>\nA: <分步推理…>。The answer is <答案>.`；用贪婪解码。算术任务共用 8 条手工写的 CoT exemplar（AQuA 因为是选择题用 4 条）；常识任务每条 4–7 条；符号任务每条手工示范。评估 5 个模型族：GPT-3（text-ada/babbage/curie/davinci-002 ≈ 350M/1.3B/6.7B/175B）、LaMDA（0.4B–137B）、PaLM（8B/62B/540B）、UL2-20B、Codex。三类基准：算术（GSM8K、SVAMP、ASDiv、AQuA、MAWPS）、常识（CSQA、StrategyQA、Date/Sports Understanding、SayCan）、符号（last letter concatenation、coin flip）。
- **关键实验结论**：
  - **GSM8K 显著提升**：PaLM 540B + 8 条 CoT exemplar 达到 57% 解题率，远超标准 prompting（18%）与微调 GPT-3 175B + verifier（33%），刷新当时 SOTA（图 2：33 / 55 / 18 / 57）。
  - **CoT 是规模上的涌现能力**：小模型（<~100B）不仅没收益，还常产生"流畅但不合逻辑"的推理链，性能反而低于标准 prompting；到 ~100B 才出现明显增益。越难的题增益越大（GSM8K 上性能翻倍以上；单步题 SingleOp 无增益）。
  - **消融隔离出"自然语言中间步骤"是关键**：只给方程（equation only）、只给一串点模拟"可变计算量"（variable compute）、把推理步骤放在答案之后（reasoning after answer），三者都与基线持平 → 增益来自"作为顺序推理的自然语言步骤"，不是方程、不是算力、不是知识激活。
  - **鲁棒性**：不同标注者（A/B/C）、不同 exemplar 集（含 GSM8K 训练集抽样）、不同顺序、不同数量，CoT 始终大幅优于基线（虽然 prompt 工程仍会引入方差，如 coin flip 从 99.6% 到 71.4%）。
  - **符号推理 + 长度外推（OOD）**：PaLM 540B 在 in-domain 接近 100%；标准 prompting 在更长序列上完全失败，而 CoT 让长度泛化（2 词名字→4 词、2 次翻币→4 次）成为可能。
  - **错误分析**：50 个正确样本中 48 个推理链逻辑正确；50 个错误样本中 46% 是"几乎正确"（计算器/符号映射/漏一步），54% 是语义理解或连贯性问题；把 PaLM 从 62B 扩到 540B 主要修好了"漏一步"（18 个修好 12 个）和语义理解错误（20 个修好 6 个）。
- **与课程主题的关系**：这一篇是整个第 12 讲的锚点——证明"推理可以被提示激发、且只在足够大模型上被激发"，把"涌现"与"提示"两个概念焊在一起。也是后文 self-consistency 的直接基础（论文自己也提到多数投票是改进方向）。
- **可演示的代码点**：手写标准 vs CoT 两组 few-shot 模板并对比输出；实现 "The answer is X." 的答案解析；复现三种消融（equation only / dots / answer-first）；用规则或小模型复现 last-letter 与 coin flip 玩具任务及 OOD 长度外推。

### 论文 2：Self-Consistency Improves Chain of Thought Reasoning（arxiv:2203.11171，self-consistency.pdf）
- **核心思想**：CoT 的贪婪解码只走一条路径，容易掉进局部最优或单步错误。自洽性把解码换成"**sample-and-marginalize**"：先采样出一组**多样的推理路径**，再对它们各自的最终答案做**多数投票**。直觉：复杂的推理题通常存在多条通向唯一正确答案的路径；正确的路径即使多样也会在最终答案上"汇合"，而错误的路径很少巧合到给出同一个答案。这类似人的经验——多种思路得到同一个答案时信心更足。方法完全无监督、无需 verifier / reranker / 微调，是单模型上的"self-ensemble"。
- **关键公式/算法**：三步骤：(1) CoT 提示；(2) 从解码器采样 m 条路径（temperature + top-k）；(3) 对最终答案集合取 argmax 多数投票 `argmax_a Σ_{i=1..m} 𝟙(a_i = a)`。可选加权：用长度归一化的条件对数概率 `P(r_i,a_i|prompt,q) = exp((1/K)·Σ_k log P(t_k | …))`。**关键发现：无加权多数投票 ≈ 归一化加权求和（74.4 vs 74.1），且明显优于未归一化加权**——因为模型各条路径的概率差不多（模型校准差），因此简单多数投票就够了。采样配置：UL2/LaMDA T=0.5、top-k=40；PaLM T=0.7、top-k=40；GPT-3 T=0.7 无 top-k；默认采样 40 条、10 次取均值。
- **关键实验结论**：
  - **普遍大增益**：PaLM-540B 上 GSM8K 56.5→74.4（**+17.9%**）、SVAMP +7.6%、AQuA +12.5%、StrategyQA 75.3→81.6（+6.4%）、ARC-c 85.2→88.7（+3.9%）。摘要口径：GSM8K +17.9%、SVAMP +11.0%、AQuA +12.2%、StrategyQA +6.4%、ARC-c +3.9%。
  - **增益随规模增大**：UL2-20B 只 +3–6%，LaMDA-137B 与 GPT-3 达 +9–23%；对已很强的模型仍显著（AQuA/GSM8K +12–18%）。在几乎所有任务上超越有监督 SOTA，且完全无监督。
  - **优于其他解码/集成方法**：同样样本数下显著优于 sample-and-rank；优于 beam search（beam 多样性低，反而更差）；优于 prompt 打乱集成（40 次置换 19.2、3 套 prompt 18.6 vs self-consistency 27.7，LaMDA GSM8K）；多模型集成不如单模型自洽（PaLM 540B self-consistency 74.4 vs 模型集成最高 36.9）。
  - **鲁棒且通用**：对采样策略/超参（T、top-k、nucleus p）稳健；能修复不完美提示（17.1→14.9，+SC→23.4）、方程提示、以及 zero-shot CoT（PaLM 零样本 43.0→69.2，+26.2%）；在 CoT 有害的普通 NLP 任务上也能反超标准 prompting。
  - **一致性即不确定性**：样本间一致性（最大票占比）与正确率高度相关 → 可以当"知道自己不知道"的置信度信号（图 5）。
  - **路径数-精度饱和曲线**：路径越多越好但很快饱和，实践中 5–10 条即可拿到大部分收益。
- **与课程主题的关系**：CoT 的直接续作（同为 Denny Zhou 工作），把"推理"从一次性输出推进到"多路径决策 + 聚合"，并顺手给了 Agent 需要的**不确定性估计**与校准视角。它示范了"不需要更多数据/训练，只用解码侧小改动"就能放大推理能力——这是提示/解码技巧这类"推理工程"的典范。
- **可演示的代码点**：numpy 实现多数投票与编辑距离（对自由文本答案）自洽聚合；复现 Table 1 四种聚合策略对比；模拟"错误路径不汇合"的玩具概率模型；画"路径数 vs 精度"饱和曲线；画"一致性 vs 正确率"图复现不确定性信号。

### 论文 3：Emergent Abilities of Large Language Models（arxiv:2206.07682，emergent-abilities.pdf）
- **核心思想**：这是一篇综述/概念论文（Jason Wei 等，Denny Zhou 参与），把"涌现能力"形式化：**某个能力如果小模型没有、大模型才有，且无法靠外推小模型性能预测出来，就是涌现的**。定义根植于 Philip Anderson 1972 年 "More Is Different"：系统里的**量的变化**导致**质的行为变化**。画成 scaling curve（x 轴训练 FLOPs），涌现表现为：性能在小规模段贴近随机（flat），到达某个临界规模后**跳**到明显高于随机——一种"相变"形态。论文强调涌现是"许多相关变量的函数"（算力、参数量、数据、甚至 WikiText103 困惑度），临界规模不是能力的内在属性，会随数据质量、架构、训练方法变化。
- **关键公式/算法**：无方法，主要是**分类学 + 证据汇总**。两类涌现场景：(a) few-shot prompting 能力本身（图 2 的 8 个例子）；(b) **增强型提示/微调技巧只有在大模型上才有效**（图 3）——如果一个技巧在小模型上无效甚至有害、到足够规模才开始生效，也视为涌现能力。给出的机制直觉：多步推理需要 O(l) 层深度的顺序计算；闭卷 QA 需要足够参数"压缩知识库"。
- **关键实验结论**：
  - 涌现的 few-shot 任务（图 2）：模算术（GPT-3 于 2×10²² FLOPs/13B、LaMDA 于 10²³/68B 跳出随机）、IPA 转写、字母重排、波斯语 QA、TruthfulQA（Gopher 280B/5×10²³ FLOPs 跳到高于随机 20%+）、grounded mappings、MMLU（≤~10B 全部接近随机，70B–280B 才显著超过随机）、WiC。
  - 涌现的增强技巧（图 3 + 表 1）：**CoT 于 ~10²³ FLOPs（~100B）**；指令微调 FLAN 于 68B；**scratchpad（8 位加法）仅需 40M 参数/8.9×10¹⁹ FLOPs 就涌现**（阈值可以很低）；P(True) 校准于 52B。表 1 明确把 **self-consistency 也列为涌现技巧**（1.3×10²³ FLOPs / 68B LaMDA），形成三篇论文的闭环。
  - **历史案例 WiC**：GPT-3/Chinchilla 到最大规模仍不高于随机，Brown et al. 曾归咎于架构/自回归目标；结果 PaLM 540B（2.5×10²⁴ FLOPs）靠继续 scaling 就突破——说明"负结果"也可能只是规模没到。
  - **涌现 ≠ 只有规模**：Sanh 等人用 encoder-decoder T0 在 11B 上实现了指令跟随；InstructGPT 用 RLHF 让 1.3B 胜过更大模型；PaLM 62B 能在 14 个 LaMDA/GPT-3 到最大都接近随机的 BIG-Bench 任务上超随机（数据质量与架构也起作用）。**能力可先在小模型上出现，再随训练方法进步被"解锁"到更小规模**。
  - **度量视角的自我批判（§5.1）**：exact match / 只判最终答案对错会掩盖递增的局部改进，可能把平滑进步伪装成涌现；但用交叉熵损失分析六个涌现任务，发现小规模时 CE 确实在稳步下降（Outcome 2），且分类任务也涌现、换 BLEU/ROUGE 等部分评分指标依然有涌现形态 → 度量只能部分解释，不能完全归因于度量。
  - **BIG-Bench 关键词分析**：涌现占比最高的是类比推理、词义消歧、真实性、社会推理、情绪理解；数学/算术占比反而低；视觉、非语言、context-length 类多为 flat 曲线。没有清晰的"哪些任务会涌现"规律。
- **与课程主题的关系**：给整讲提供了概念伞：CoT 论文的"CoT 是规模涌现"、self-consistency 论文的"自洽解码也是涌现技巧"都挂在它上面。同时它引入了**能力 vs 度量**的诚实反思，是教学上"批判性转向"的关键文本。可与 Yi et al. (2204.07646，"涌现是大模型的幻觉吗？") 对比：Yi 等人用逐 token 精度、Brier score、部分评分等连续/平滑度量重画同一批曲线，发现曲线变平滑可预测，认为"涌现"主要是**不连续度量（exact match、acc）造成的假象**；本论文则用 CE 分析与分类任务上的涌现反驳"纯粹是度量"。两边本质分歧在于：底层能力是平滑增强的（Yi 方）还是存在真正的质变门槛（Wei 方）。
- **可演示的代码点**：可视化"度量制造涌现"——对一条平滑的底层能力曲线分别用 exact match 与逐 token 精度度量；多步任务上把平滑的单步正确率 p 变形成最终准确率 p^L 的 compounding 曲线，展示跳变；用合成数据复现 Yi et al. 的核心论点。

## 教学主线（想象 Stanford 老师会怎么教）

1. **失败案例建立动机**：先展示 GSM8K 一道多步应用题在标准 few-shot prompting 下的失败输出——模型直接给一个错答案，且无论怎么加大模型，scaling curve 都接近水平。提出疑问：为什么"算力上去了，推理却上不去"？
2. **CoT 登场**：只把 exemplar 从 ⟨Q,A⟩ 换成 ⟨Q,推理步骤,A⟩，PaLM 540B 的 GSM8K 从 18%→57%，超过微调 + verifier 的 GPT-3。强调两个"竟然"：竟然不用训练；竟然只在 ~100B 以上才有效。用 Figure 1 的对比图讲清提示格式，带学生手写一遍 exemplar。
3. **用消融建立直觉**：逐一"拆掉" CoT 的成分——方程 only（说明不是会算方程）、dots only（说明不是可变算力）、答案后的推理（说明不是知识激活）→ 剩下的是"用自然语言做顺序推理"本身。再讲错误分析（62B→540B 修好了什么）建立"规模给模型买来了什么"的直觉。
4. **上升到涌现框架**：引出 emergent abilities 论文，定义 + 相变曲线 + BIG-Bench/MMLU 例子，把 CoT/self-consistency 放进表 1 的涌现技巧清单。
5. **批判性转折（学生容易卡住/想反驳的点）**：停下来讨论——这些"跳变"会不会只是 exact match 度量造成的假象？引入 Yi et al. 的"幻觉"论证，现场画两条曲线：同一底层平滑能力，exact match 呈现跳变、逐 token 精度呈现平滑。澄清"度量诚实"是推理研究的必要纪律（呼应论文 §5.1 的自我批判）。
6. **self-consistency 收尾**：回到解码侧——贪婪只有一条路，容易局部最优。直觉类比：让多个"思考者"各自解题再投票，正确的会汇合、错误的分道扬镳。展示 sample-and-marginalize、聚合策略对比（多数投票≈归一化加权）、路径数-精度饱和曲线、一致性-正确率相关（不确定性信号）。
7. **串成一句话**：推理能力 = 预训练给潜力 + 规模触发涌现 + 提示格式激发 + 解码聚合放大 + 度量诚实验证。这五个词正好预演后面 Agent 需要的四件事：prompt 设计、多路径思考、置信度估计、可验证评估。

## 代码演示点子（3-6 个）

1. **手写 few-shot CoT 提示模板**：用一个小模型或 OpenAI 兼容 API，对 5–10 道 GSM8K 风格数学题分别构造 standard（⟨Q,A⟩）与 CoT（⟨Q,步骤,A⟩）两组 prompt（各 8 条 exemplar，抄论文格式 `…The answer is X.`）。关键代码：模板字符串 + 答案正则解析 + 逐题输出对比表。期望输出：CoT 明显更高正确率，且可打印出模型生成的中间步骤观察"推理链"。

2. **numpy 实现多数投票与编辑距离自洽聚合**：给定对一个问题的 N 条"路径+答案"采样（来自 API 或预先存好的文本），实现四种聚合：多数投票、归一化 log 概率加权、未归一化加权、按编辑距离聚类后的中心答案。在 100 题上对比四种策略的正确率，复现论文 Table 1 的直觉（多数投票≈归一化加权>未归一化）。期望输出：一张聚合策略×准确率的条形图。

3. **"错误路径很少汇合"的玩具概率模型**：设单条路径以 p=0.5–0.9 给出正确答案、以 (1-p) 均匀随机抛向 k 个错误答案，用 numpy 模拟 K=1..40 条路径的投票正确率。画"投票正确率 vs 路径数×p"热图。期望输出：直观看到只要 p>1/k，投票正确率随 K 迅速逼近 1——把 self-consistency 的成功归因讲清楚。

4. **涌现曲线的度量视角可视化（复现 Yi et al. 论证）**：构造一个"底层每 token 正确率随规模平滑提升"的合成模型，用 (a) exact match / 只判最终答案、(b) 逐 token 精度、(c) 多步任务最终准确率 = p^L 三种度量画曲线。期望输出：同一底层能力在 exact match 下呈"相变/涌现"、在逐 token 精度下平滑，直观演示"涌现可能是度量制造的"，并讨论与 Wei 方 CE 分析的立场差异。

5. **符号推理 + 长度外推**：实现 last letter concatenation 与 coin flip 两个玩具任务，few-shot exemplar 只含 2 词名字/2 次翻币，测试 3–4 词/4 次翻币的 OOD 样本；用"伪 CoT"（规则模拟分步输出）对比直接给答案。期望输出：直接给答案 OOD 近乎全错，分步输出正确率随规模/规则可解释地上升。

6. **自洽性作为不确定性估计**：对一批题各采样 N=20 条路径，计算一致性（最大票占比），画出"一致性 vs 该题是否正确"的分箱散点图。期望输出：明显正相关（复现图 5），并演示把它当置信度阈值来"拒绝回答"的简单策略——直接接到 Agent 的"知道自己不知道"。

## 作业点子（3 个）

1. **手写 CoT 并做消融**：给定 8 条 GSM8K 训练样本，让学生自己写 CoT 注释，然后跑三种提示（standard / CoT / equation-only）各 30 题。填空：补全 exemplar 模板字符串、补全 `parse_answer(text)` 正则；assert：CoT 正确率 > standard 且 > equation-only，并写一句话解释为什么 equation-only 在 GSM8K 上失效。

2. **自洽性采样与聚合**：对同一道题用 temperature 采样 K=20 条路径，补全 `majority_vote(answers)` 与 `normalized_logprob_weighted`（实现论文式 (1) 的长度归一化）；回答：为什么未归一化加权比多数投票差？assert：多数投票 ≥ 贪婪解码答案的正确性，且 40 条 ≥ 5 条的准确率（不降反升）。

3. **涌现的度量可检验性**：给定合成 scaling 数据（底层能力平滑增长），补全两个度量函数（exact-match 与 per-token accuracy）并画图；写一个简单的"涌现检测器"（如阈值两侧平均斜率变化），判断给定曲线是否"涌现"。assert：exact-match 曲线被判为涌现、per-token 曲线被判为平滑；再用 1–2 句话反思：论文的 CE 分析（Outcome 2）如何反驳"纯粹是度量"这一观点。

## 参考资料

- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models（arxiv:2201.11903）— 本讲锚点：few-shot CoT 提示格式与涌现规模
- Self-Consistency Improves Chain of Thought Reasoning（arxiv:2203.11171）— sample-and-marginalize 解码：采样+多数投票
- Emergent Abilities of Large Language Models（arxiv:2206.07682）— 涌现能力的定义、分类学与证据汇总
- Are Emergent Abilities of Large Language Models a Mirage?（Yi et al., arxiv:2204.07646）— 反方观点：不连续度量制造的"涌现假象"，与本讲第三篇对照
- Training Verifiers to Solve Math Word Problems（Cobbe et al., arxiv:2110.14168）— GSM8K 数据集与"微调+verifier"基线
- Large Language Models are Zero-Shot Reasoners（Kojima et al., arxiv:2205.11916）— 零样本 CoT："Let's think step by step"（self-consistency 论文也验证了与之兼容）
- Least-to-Most Prompting Enables Complex Reasoning（Zhou et al., arxiv:2205.10625）— CoT 的进阶分解策略（同为 Denny Zhou 团队）
- Show Your Work: Scratchpads for Intermediate Computation（Nye et al., arxiv:2112.00114）— 中间计算预测，涌现于 40M 参数，支撑涌现论文证据
- Finetuned Language Models are Zero-Shot Learners / FLAN（Wei et al., arxiv:2109.01652）— 指令微调的涌现（图 3B）
- BIG-bench: Beyond the Imitation Game（arxiv:2206.04615）— 涌现证据的主要来源（200+ 任务）
