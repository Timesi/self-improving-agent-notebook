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
  <a href="#这门课能让你学会什么">能学到什么</a> ·
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

## 这门课能让你学会什么

17 本 notebook，你从零实现现代 AI Agent 背后的**每一个核心算法**——而且每写一行代码前，都先用具体
数字手算一遍。不用框架、不碰黑盒。学完后，你打开任何一篇 Agent 论文，都能一眼看出它改的是系统
的哪一块。

你会亲手实现：

- **一个 Agent 循环**——模型每轮输出"思考 + 动作"，工具执行动作，结果喂回去，直到任务完成
- **一个验证器**——训练一个小模型给答案打分：结果级看最终对错，过程级检查每一步
- **一个 ReAct Agent**——接上搜索、计算器等真实工具，看着它"边想边做"
- **一棵搜索树**——保留多条候选路径、把奖励回传（UCT 手算），而不是贪心走一步
- **一次 GRPO 更新**——推理模型背后的"组内相对强化学习"规则，逐步手算推导
- **一个进化循环**——让 Agent 设计更好的 Agent，并复现它掉进的"奖励黑客"陷阱
- **一套记忆系统**——分层上下文 + KV 缓存复用，让记忆超出上下文窗口
- **一个评测 Harness**——拟合"人类要多久、Agent 要多久"的能力曲线

每本 notebook 都遵循同一个契约：`直觉理解 → 手算验证 → 代码实现 → 实验观察`，并且无需 API key
就能用确定性 mock 离线跑通；配上 key，同一个 cell 立刻接上真实模型。

## 课程路线

17 讲是一条渐进的路：每讲加一块新能力，并接住上一讲留下的问题。

| # | 讲座 | 学完你能... |
|:--:|:--|:--|
| 01 | 课程总览 | 说清 Agent 是什么，跑通最小循环 |
| 02 | Test-time Compute | 用"多采样 + 投票"提升准确率 |
| 03 | 鲁棒验证 | 训练一个验证器，分清答案好坏 |
| 04 | 工具与代码反馈 | 做一个会调真实工具的 ReAct Agent |
| 05 | 多步推理与规划 | 在动作空间里做树搜索，而不是猜一步 |
| 06 | 训练期 RL | 手算一次 GRPO 更新，把推理训练进模型 |
| 07 | 开放进化 | 让 Agent 改进 Agent，并识破奖励黑客 |
| 08 | 搜索与深度研究 | 跑 AlphaCode 规模的"采样-过滤-聚类" |
| 09 | 后训练演进 | 看清一个模型从 Chatbot 到 Agent 的一生 |
| 13 | 软件工程智能体 | 让 Agent 迭代地修复真实代码 |
| 14 | 智能体记忆 | 给 Agent 一份超出上下文窗口的记忆 |
| 17 | Agent 评测 | 度量 Agent 到底能完成什么 |
| 15 | LLM 推理 | 解释 CoT 为什么有效、"涌现"何时为真 |
| 16 | 数学推理 | 把神经网络的提议和符号验证器结合起来 |
| 18 | 自治系统 | 检错、恢复、决定何时交给人类 |
| 19 | 多模态机器人 | 把连续的机器人动作变成模型能输出的 token |
| 20 | 未来研究方向 | 点出还没解决的问题、可以从哪入手 |

### 各讲怎么连起来

不是 17 个孤立话题——概念在讲与讲之间传递：

- **投票 → 验证 → 搜索**（L2→L3→L5）：投票用的分数，变成验证器，再变成树搜索里选节点的依据
- **验证 → RL 奖励 → 适应度**（L3→L6→L7）：同一个验证器先当训练信号，再当进化的适应度
- **采样 → 规模化搜索**（L2→L8）："多问几次"放大到百万候选，就是 AlphaCode
- **循环 → 装进树 → 跑在代码上**（L4→L5→L13）：同一个 ReAct 循环，三种用法
- **奖励黑客 → 评测要防的 → Goodhart**（L7→L17→L20）

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
