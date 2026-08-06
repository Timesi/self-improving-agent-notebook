# Self-Improving Agent Notebook

<p align="center">
  <strong>用 17 篇可运行 Jupyter Notebook，从一次 LLM 调用出发，一路走到会自我进化的 Agent。</strong>
</p>

<p align="center">
  <a href="README.md"><strong>English</strong></a>
  ·
  <a href="README-CN.md"><strong>中文文档</strong></a>
  ·
  <a href="https://walkinglabs.github.io/self-improving-agent-notebook/"><strong>在线阅读</strong></a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C">
  <img alt="Notebooks" src="https://img.shields.io/badge/Notebooks-17-orange">
  <img alt="Language" src="https://img.shields.io/badge/Language-Chinese-2ea44f">
  <img alt="LLM" src="https://img.shields.io/badge/LLM-OpenAI%20Compatible%20%2B%20Mock-4b32c3">
</p>

<p align="center">
  <a href="#这是一门课">这是一门课</a> ·
  <a href="#学习路线">学习路线</a> ·
  <a href="#你会亲手实现什么">你会实现什么</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#项目状态">项目状态</a> ·
  <a href="#课程地图">课程地图</a> ·
  <a href="#notebook-索引">Notebook 索引</a> ·
  <a href="#质量标准">质量标准</a> ·
  <a href="#课程来源">课程来源</a>
</p>

---

## 这是一门课

Self-Improving Agent Notebook 是一门 Agent 实践课。它不是框架调用教程——每一讲都从论文出发，
用手算和从零实现把 Agent 的核心算法做出来。

一句话概括这门课的主线：

> **一次 LLM 调用只会"说"不会"做"。我们一步步把它变成能思考、能动手、能验证、能被训练、
> 最终能自我改进的 Agent。**

每个 notebook 遵循同一个学习契约：

```text
直觉理解 -> 手算验证 -> 代码实现 -> 实验观察
```

你不只会"跑通"代码，而是能追着具体数字把它手算一遍，再亲手实现它，最后解释为什么行为会这样出现。

## 学习路线

这门课按认知难度排成一条渐进的路。**每一步都以上一步为基础，加一块新能力；每一步结束时，
你都能做一件之前做不到的事。** 想跳着读也欢迎，但按顺序读，每本 notebook 都会接住上一本留下的问题。

### 第 1 步 · 看清一次调用能做什么、不能做什么（L1）

起点先不急着写代码，先看清 LLM 应用的四条硬边界：只会"说"不会"做"、上下文窗口有限、
错了不会自我纠正、知识停在训练截止日。这一讲动手实现**第一个最小 Agent 循环**——
这正是整门课反复出现的骨架。

### 第 2 步 · 推理时多花算力，能不能变强？（L2）

模型答得不够好，但训练太贵。换个思路：**同一个问题多问几次，再投票**。这一讲实现重复采样、
self-consistency、best-of-n，并手算"采样越多、覆盖越好"的曲线。关键收获是——**生成只是第一步，
"挑出好的"才是难点**，这直接引出下一步。

### 第 3 步 · 谁来挑？让机器检查机器（L3）

"挑出好的"不能靠猜。这一讲训练**验证器**：结果验证器（ORM）只看最终答案对不对，
过程验证器（PRM）检查每一步。你会看到同一个错误答案，两种验证器给出的分数完全不同。
收获：**生成 + 验证是一对搭档**，验证器会在后面的强化学习里反复出现。

### 第 4 步 · 让 Agent 动手：工具与环境（L4）

光想不做没有意义。这一讲实现 **ReAct 循环**：模型每轮输出"思考 + 动作"，动作在工具里执行，
观察结果喂回去。你会亲手给 Agent 接上搜索、计算器等工具，看着它"边想边做"。

### 第 5 步 · 让 Agent 规划：一步之外的事（L5）

任务太长、一步做不完怎么办？这一讲学**规划**：把任务拆成子目标（ADaPT）、
在动作空间里做树搜索（LATS 的 UCT 选择）、把可以并行的步骤打包（SPRINT）。
你会发现"保留多条候选路径"比"贪心走一步"强得多——这和第 2 步的投票是同一个思想。

### 第 6 步 · 把能力训练进模型：强化学习（L6）

推理时花算力很贵，能不能**把会推理的能力装进模型**？这一讲实现 STaR 自举
（让模型用自己答对的题继续学）和 GRPO（组内相对强化学习）。你会手算一组样本的 advantage，
看清"为什么减去均值、除以标准差"。**第 3 步的验证器，在这里变成了训练时的奖励信号。**

### 第 7 步 · 让 Agent 改进 Agent：开放进化（L7）

如果 Agent 本身可以被改进，那改进它的也是 Agent 呢？这一讲实现**进化循环**：
一个"设计 Agent"生成 Agent 的代码 → 在任务上评分 → 入库 → 下一轮基于评分改进。
还会看到一个危险的失败模式——**奖励黑客**：Agent 学会"刷分"而不是"做成事"。

### 第 8 步 · 让 Agent 自己搜索、自己做研究（L8）

把第 2 步的"多采样"放大一万倍，就是代码合成和深度研究的思路。这一讲实现**采样-过滤-聚类**
管线（AlphaCode）、按需检索的推理（Search-o1）、以及一个迷你深度研究工作流。

### 第 9 步 · 回看全景：一个模型从 Chatbot 到 Agent 的一生（L9）

把前八步串起来，看**后训练**的完整演进：SFT（模仿示范）→ RLHF（学习偏好）→
RLVR（学习可验证的对错）→ Agent 化训练（学习环境反馈）。核心线索是**信号来源的迁移**。

### 第 10 步 · 工程：让它真的能用（L13 / L14 / L17）

前面学的是"能力"，这里学"落地"：让 Agent **修代码**（软件工程智能体）、**记住事**（记忆系统）、
**被公正地评价**（评测 Harness 与长时程任务）。评测里你会学到时间视野（time horizon）——
一个任务人类要多久、Agent 要多久，能力曲线怎么拟合。

### 第 11 步 · 前沿：边界在哪里（L15 / L16 / L18-L20）

最后走向 Agent 的前沿：**推理的极限**（CoT 与自洽）、**数学证明**（AlphaGeometry 与 AlphaProof
的神经-符号结合）、**自治与监督**、**机器人**（视觉-语言-动作模型）、以及**还没解决的开放问题**。

### 一条概念的接力

各讲之间不是并列的，很多概念是"接力"的：

```text
L2 投票 → L3 验证器帮你"挑" → L5 树搜索里也靠打分选节点
L3 验证器 → L6 GRPO 把它当训练奖励 → L7 进化里它当适应度
L2 多采样 → L8 AlphaCode 放大到百万候选
L4 ReAct 循环 → L5 被装进搜索树 → L13 在代码任务里被反复调用
L7 奖励黑客 → L17 评测要防的就是它 → L20 Goodhart 定律
```

按顺序读，每一本都会接住上一本留下的问题；跳着读，也能在"概念的接力"里找到位置。

## 你会亲手实现什么

| 你会亲手实现 | 一句话说明（不堆术语） |
|:---|:---|
| 一个 Agent 循环 | 模型每轮输出"思考 + 动作"，动作在工具里执行，结果喂回去，直到任务完成 |
| 重复采样与投票 | 同一个问题让模型答 N 次，多数票获胜；看清"生成容易、挑难" |
| 一个验证器 | 训练一个小模型给答案打分：结果级只看最后对错，过程级检查每一步 |
| 一个 ReAct Agent | 给模型接上搜索、计算器等工具，看着它"边想边做"地完成任务 |
| 一棵搜索树 | 把任务当成树：选择节点 → 扩展 → 评估 → 把结果回传给祖先（手算 UCT 与回传） |
| 一个 GRPO 更新 | 手算一组样本的 advantage，跑一轮"组内相对强化学习"更新 |
| 一个进化循环 | 让一个"设计 Agent"生成 Agent 代码、评分、入库、再改进；并复现奖励黑客 |
| 一个分层记忆 | 主上下文装不下时，把旧信息换出、需要时召回；理解 KV 缓存复用 |
| 一个评测 Harness | 定义任务、跑 Agent、汇总通过率；用时间视野拟合"能力-时长"曲线 |
| 一个证明搜索器 | 让模型出证明步骤、验证器逐条把关，理解 AlphaProof 的思路 |

## 快速开始

### Python notebooks

```bash
git clone https://github.com/walkinglabs/self-improving-agent-notebook.git
cd self-improving-agent-notebook

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user \
  --name self-improving-agent \
  --display-name "Python (self-improving-agent)"

jupyter notebook notebooks/part1-foundation/01-course-overview.ipynb
```

如果提示 `jupyter: command not found`，说明虚拟环境没有激活，运行 `source .venv/bin/activate`，
或直接用 `.venv/bin/jupyter notebook ...`。

推荐环境：Python 3.10+、PyTorch 2.0+、NumPy、Matplotlib、Jupyter、16GB 内存。
大多数 notebook 在 CPU 上即可运行。

### Mock 模式（无需 API key）

每一本 notebook 都保证所有依赖 LLM 的 cell 在确定性 mock 下正常工作：

```bash
export LLM_MOCK=1
jupyter nbconvert --to notebook --execute \
  notebooks/part1-foundation/02-test-time-compute.ipynb --output /tmp/out.ipynb
```

### 真实 LLM（可选）

想看真实模型行为，把客户端指向任意 OpenAI 兼容端点：

```bash
export AGENT_LLM_BASE_URL="https://api.deepseek.com"
export AGENT_LLM_API_KEY="sk-..."
export AGENT_LLM_MODEL="deepseek-v4-flash"
```

没有 key 时，`llm_client.py` 的 `get_llm()` 自动返回 mock，notebook 依旧完整执行。

## 项目状态

| 领域 | 状态 |
|:---|:---|
| Notebook | 完成 17/17，分布在 4 个 Part |
| 研读笔记 | 17/17 讲，每讲一份 `papers/lecture-XX/NOTES.md` |
| 论文 | 下载并精读 39 篇；`scripts/download_papers.py` 可复现下载 |
| 讲解深度 | 每本补齐"直觉 + 具体例子 + 手算 + 为什么"，面向大一读者 |
| 格式 | 全部通过 `nbformat.validate`，cell id 规范化 |
| 执行 | mock 模式 17/17 零报错；真实 API 抽查通过 |
| 语言 | 中文 notebook，中英双语 README |

### 近期路线图

1. 打磨行文，让讲解从直觉到代码更自然。
2. 扩大真实 API 验证范围。
3. 增加 Agent 可靠性与可扩展监督的深度内容。
4. 考虑补一份英文 notebook 镜像。

## 课程地图

```text
Self-Improving Agent Notebook
│
├── Part 1 · 基础与方法（L1-L5）     # 循环、test-time compute、验证、工具、规划
│   ├── 01 课程总览
│   ├── 02 Test-time Compute 缩放
│   ├── 03 鲁棒验证
│   ├── 04 工具使用与代码反馈
│   └── 05 多步推理与规划
│
├── Part 2 · 训练与进化（L6-L9）     # 训练进模型、进化、搜索、后训练全景
│   ├── 06 训练期缩放与强化学习
│   ├── 07 自改进智能体的开放进化
│   ├── 08 搜索与深度研究智能体
│   └── 09 后训练演进：从 Chatbot 到 Agent
│
├── Part 3 · 智能体工程（L13/L14/L17）# 让它真的能用
│   ├── 13 软件工程智能体
│   ├── 14 智能体记忆
│   └── 17 Agent 评测与长程任务
│
└── Part 4 · 前沿（L15-L20）         # 边界在哪里
    ├── 15 LLM 推理
    ├── 16 数学推理：AlphaProof 与 AlphaGeometry
    ├── 18 构建自治智能体
    ├── 19 多模态机器人智能体
    └── 20 未来研究方向
```

每本 notebook 自包含，可独立运行。原始课程中的 L10-12 是中期展示，没有 notebook。

## Notebook 索引

### Part 1 · 基础与方法

| # | Notebook | 核心问题 | 实现重点 |
|:---:|:---|:---|:---|
| 01 | [课程总览](notebooks/part1-foundation/01-course-overview.ipynb) | 什么是 Agent，为什么它是 Chatbot 的下一步？ | 最小循环骨架、动作解析器 |
| 02 | [Test-time Compute 缩放](notebooks/part1-foundation/02-test-time-compute.ipynb) | 推理期多花算力为什么有用？ | 重复采样、self-consistency、best-of-n |
| 03 | [鲁棒验证](notebooks/part1-foundation/03-robust-verification.ipynb) | 如何检查模型生成出的答案？ | ORM vs PRM、验证器训练、步级奖励 |
| 04 | [工具使用与代码反馈](notebooks/part1-foundation/04-tool-code-feedback.ipynb) | Agent 如何在环境中行动？ | ReAct 循环、工具注册表、执行奖励 |
| 05 | [多步推理与规划](notebooks/part1-foundation/05-multi-step-planning.ipynb) | Agent 如何规划一步之外？ | 任务分解、UCT 树搜索、并行执行 |

### Part 2 · 训练与进化

| # | Notebook | 核心问题 | 实现重点 |
|:---:|:---|:---|:---|
| 06 | [训练期缩放与强化学习](notebooks/part2-training/06-train-time-scaling-rl.ipynb) | 如何把推理能力训练进模型？ | STaR 自举、GRPO advantage 与 loss |
| 07 | [开放进化](notebooks/part2-training/07-open-ended-evolution.ipynb) | Agent 如何设计更好的 Agent？ | 进化循环、奖励黑客 |
| 08 | [搜索与深度研究](notebooks/part2-training/08-search-deep-research.ipynb) | Agent 如何搜索程序与知识？ | 采样-过滤-聚类、按需检索 |
| 09 | [后训练演进](notebooks/part2-training/09-post-training-evolution.ipynb) | 后训练如何从 Chatbot 演进到 Agent？ | SFT vs RLHF vs RLVR、信号来源 |

### Part 3 · 智能体工程

| # | Notebook | 核心问题 | 实现重点 |
|:---:|:---|:---|:---|
| 13 | [软件工程智能体](notebooks/part3-engineering/13-swe-agents.ipynb) | Agent 如何修复真实代码？ | 覆盖-选择、串行修复循环 |
| 14 | [智能体记忆](notebooks/part3-engineering/14-agent-memory.ipynb) | Agent 如何记住看到过的东西？ | 分层上下文、逐出、KV 复用 |
| 17 | [Agent 评测](notebooks/part3-engineering/17-agent-evaluation.ipynb) | 如何度量长时程 Agent？ | 评测 Harness、胜率、时间视野 |

### Part 4 · 前沿

| # | Notebook | 核心问题 | 实现重点 |
|:---:|:---|:---|:---|
| 15 | [LLM 推理](notebooks/part4-frontiers/15-llm-reasoning.ipynb) | 推理能力从哪里来？ | CoT、自洽投票、涌现度量 |
| 16 | [数学推理](notebooks/part4-frontiers/16-alphaproof-math.ipynb) | 验证器 + 搜索如何证明定理？ | 符号引擎、证明验证器、搜索 |
| 18 | [自治系统](notebooks/part4-frontiers/18-autonomy-agents.ipynb) | 从演示到自治之间缺什么？ | 可靠性、自检重试、监督 |
| 19 | [多模态机器人](notebooks/part4-frontiers/19-multimodal-robotics.ipynb) | VLA 模型如何工作？ | 动作离散化、词汇表屏蔽 |
| 20 | [未来研究方向](notebooks/part4-frontiers/20-future-research.ipynb) | 还有什么没解决？ | 开放问题、Goodhart、协调 |

## 质量标准

为了让 notebook 真正可用作学习材料，本仓库遵循几条标准：

- 概念先讲动机，再给记号。
- 新术语先定义，再大量使用。
- 核心算法至少包含一次具体的手算或 toy 示例。
- 代码 cell 短小、可观察。
- 随机实验使用固定种子。
- 每个 notebook 自包含，mock 模式零报错运行。
- 讲解面向有耐心的初学者，代码保持贴近真实算法结构。

## 论文与系统

本课程把实现细节与有影响力的论文和系统连接起来：

| 论文或系统 | 覆盖的概念 |
|:---|:---|
| Large Language Monkeys / Snell et al. | 重复采样、compute-optimal 测试期缩放 |
| Cobbe 验证器 / Lightman / Math-Shepherd | 结果奖励模型 vs 过程奖励模型 |
| ReAct / RLEF / Constitutional AI | 工具使用、执行反馈、AI 反馈 |
| LATS / ADaPT / SPRINT | 树搜索、按需分解、并行执行 |
| STaR / DeepSeekMath-GRPO / DAPO | 推理自举、组内相对 RL |
| ADAS / AI Scientist / AlphaEvolve | 自动化 Agent 设计、开放进化 |
| AlphaCode / Search-o1 | 程序采样-过滤-聚类、Agentic 检索 |
| MemGPT / Cartridges / CacheBlend | 分层记忆、可学习压缩、KV 复用 |
| CodeMonkeys / KernelBench | 软件工程中的测试期算力 |
| METR / GDPval / DeepScholar-Bench | 长时程与经济价值评测 |
| CoT / Self-Consistency / Emergent Abilities | 推理链、采样并边缘化 |
| AlphaGeometry / AlphaProof | 神经-符号证明、形式化 + RL |
| RT-2 / OpenVLA | 视觉-语言-动作模型 |

## 课程来源

课程大纲复刻自 Stanford 的 [CS329A: AI Agents](https://cs329a.stanford.edu/)（Autumn 2025）。
原课程页只列出每讲的论文、不展开内容。本仓库下载并精读这些论文，为每一讲写一份研读笔记
（`papers/lecture-XX/NOTES.md`），再把每一讲做成一本可运行的 notebook，还原讲师会怎么教这节课。

没有指定论文的嘉宾讲座（后训练演进、LLM 推理、AlphaProof、自治、机器人等），按嘉宾的
研究方向与相关论文补全内容。研读笔记见 [papers/](papers/)，用
`scripts/download_papers.py` 可复现论文下载。教学契约与仓库风格参考
[Modern LLM Notebook](https://github.com/walkinglabs/modern-llm-notebook)。

## 仓库结构

```text
self-improving-agent-notebook/
├── notebooks/                    # 中文 notebook（17 本）
│   ├── part1-foundation/         # 01-05
│   ├── part2-training/           # 06-09
│   ├── part3-engineering/        # 13, 14, 17
│   └── part4-frontiers/          # 15, 16, 18, 19, 20
├── papers/                       # 每讲研读笔记（NOTES.md）；论文 PDF 可复现
├── scripts/download_papers.py    # 从 arXiv 解析并下载全部课程论文
├── llm_client.py                 # 统一 LLM 客户端（OpenAI 兼容 + 确定性 mock）
├── web/                          # React/Vite 在线阅读器（部署到 GitHub Pages）
├── .claude/CLAUDE.md             # Notebook 写作规范
├── OUTLINE.md                    # 完整课程大纲
└── README.md / README-CN.md
```

## 贡献

欢迎能提升清晰度、正确性或覆盖面的贡献：修正错误讲解、改进手算与可视化、增加带断言的小练习、
为重要的 Agent 主题提议新 notebook、帮助在真实 LLM 端点上验证 notebook。

## 引用

如果 Self-Improving Agent Notebook 对你的研究或工作有帮助，请引用：

```bibtex
@misc{self-improving-agent-notebook,
  title   = {Self-Improving Agent Notebook: Build AI Agents from Scratch},
  author  = {WalkingLabs},
  year    = {2026},
  url     = {https://github.com/walkinglabs/self-improving-agent-notebook},
  note    = {GitHub repository, accessed 2026}
}
```

## 许可证

许可证待定。项目遵循
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/)
的精神，与参考项目 Modern LLM Notebook 一致。

---

<p align="center">
  <sub>
    为想从内部理解 Agent 系统的工程师而建。
    <br>
    由 <a href="https://github.com/walkinglabs">walkinglabs</a> 维护。
  </sub>
</p>
