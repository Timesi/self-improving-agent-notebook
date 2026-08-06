# CS329A · AI Agents — Self-Improving Agent Notebook

**AI Agents · Autumn 2025 · Stanford CS329A 动手实践版**

本仓库把 Stanford 课程 [CS329A: AI Agents](https://cs329a.stanford.edu/)（Autumn 2025）的
每一讲做成一门可运行的动手实践 notebook。课程大纲只列出每讲的论文、不展开内容——
这里为每一讲**先精读论文、再想象这门课的老师会怎么教、最后用代码亲手实现这节课的核心算法**。

17 本中文 notebook，4 个 Part，教学契约与
[Modern LLM Notebook](https://github.com/walkinglabs/modern-llm-notebook) 一致：

```text
直觉理解 -> 手算验证 -> 代码实现 -> 实验观察
```

## Schedule

| 讲座 | 日期 | 主题 | Notebook | 论文与研读笔记 |
|:---|:---|:---|:---|:---|
| 1 | Sep 22 | Course Overview | [01-course-overview](notebooks/part1-foundation/01-course-overview.ipynb) | [notes](papers/lecture-01/NOTES.md) |
| 2 | Sep 26 | Test-time Compute Scaling | [02-test-time-compute](notebooks/part1-foundation/02-test-time-compute.ipynb) | [notes](papers/lecture-02/NOTES.md) |
| 3 | Sep 29 | Robust Verification | [03-robust-verification](notebooks/part1-foundation/03-robust-verification.ipynb) | [notes](papers/lecture-03/NOTES.md) |
| 4 | Oct 3 | Learning from Feedback with Tools/Code | [04-tool-code-feedback](notebooks/part1-foundation/04-tool-code-feedback.ipynb) | [notes](papers/lecture-04/NOTES.md) |
| 5 | Oct 6 | Multi-step Reasoning / Planning | [05-multi-step-planning](notebooks/part1-foundation/05-multi-step-planning.ipynb) | [notes](papers/lecture-05/NOTES.md) |
| 6 | Oct 10 | Train-time Scaling / Scaling RL | [06-train-time-scaling-rl](notebooks/part2-training/06-train-time-scaling-rl.ipynb) | [notes](papers/lecture-06/NOTES.md) |
| 7 | Oct 13 | Open-Ended Evolution of Self-Improving Agents | [07-open-ended-evolution](notebooks/part2-training/07-open-ended-evolution.ipynb) | [notes](papers/lecture-07/NOTES.md) |
| 8 | Oct 17 | Search & Deep Research Agents | [08-search-deep-research](notebooks/part2-training/08-search-deep-research.ipynb) | [notes](papers/lecture-08/NOTES.md) |
| 9 | Oct 20 | Post-training: from Chatbots to Agents *(guest)* | [09-post-training-evolution](notebooks/part2-training/09-post-training-evolution.ipynb) | [notes](papers/lecture-09/NOTES.md) |
| 10-12 | Oct 24-Nov 1 | Mid term presentations | — | — |
| 13 | Nov 3 | Agentic Frameworks for Software Engineering | [13-swe-agents](notebooks/part3-engineering/13-swe-agents.ipynb) | [notes](papers/lecture-13/NOTES.md) |
| 14 | Nov 7 | Augmenting Agents with Memory | [14-agent-memory](notebooks/part3-engineering/14-agent-memory.ipynb) | [notes](papers/lecture-14/NOTES.md) |
| 15 | Nov 10 | LLM Reasoning *(guest: Denny Zhou)* | [15-llm-reasoning](notebooks/part4-frontiers/15-llm-reasoning.ipynb) | [notes](papers/lecture-15/NOTES.md) |
| 16 | Nov 14 | Superhuman Reasoning: AlphaProof & AlphaGeometry *(guest)* | [16-alphaproof-math](notebooks/part4-frontiers/16-alphaproof-math.ipynb) | [notes](papers/lecture-16/NOTES.md) |
| 17 | Nov 17 | Agentic Evaluations & Long-Horizon Tasks | [17-agent-evaluation](notebooks/part3-engineering/17-agent-evaluation.ipynb) | [notes](papers/lecture-17/NOTES.md) |
| 18 | Nov 21 | Building Agentic Systems for Autonomy *(guest)* | [18-autonomy-agents](notebooks/part4-frontiers/18-autonomy-agents.ipynb) | [notes](papers/lecture-18/NOTES.md) |
| 19 | Dec 1 | Multimodal AI Agents in Robotics *(guest)* | [19-multimodal-robotics](notebooks/part4-frontiers/19-multimodal-robotics.ipynb) | [notes](papers/lecture-19/NOTES.md) |
| 20 | Dec 5 | Future Research Areas | [20-future-research](notebooks/part4-frontiers/20-future-research.ipynb) | [notes](papers/lecture-20/NOTES.md) |

带 *guest* 的讲座为嘉宾课（原课程无指定论文），本仓库按嘉宾研究方向与相关论文补全内容。

## Course Description

**你会亲手实现什么**：每一本 notebook 都不把 Agent 当作黑盒。核心算法全部从零实现——

| Part | 讲座 | 你会实现 |
|:---|:---|:---|
| Part 1 · Foundation | L1-L5 | Agent 循环、重复采样与 best-of-n、验证器（ORM/PRM）、ReAct 循环与工具执行、树搜索与任务分解 |
| Part 2 · Training & Evolution | L6-L9 | GRPO 与 RL 缩放、Agent 搜索与进化、代码采样-过滤-聚类、后训练演进 |
| Part 3 · Engineering | L13/L14/L17 | SWE 修复循环、分层记忆与 KV 复用、Agent 评测 Harness |
| Part 4 · Frontiers | L15/L16/L18-L20 | CoT 与自洽性、证明搜索与验证器、自治与监督、VLA 动作离散化、开放问题 |

## Quick Start

```bash
# 1. 创建环境
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. mock 模式跑通任意 notebook（无需 API key，离线可执行）
pip install jupyter
jupyter nbconvert --to notebook --execute \
  notebooks/part1-foundation/02-test-time-compute.ipynb --output /tmp/out.ipynb
```

**LLM 配置（可选）**：需要真实 LLM 输出时，设置环境变量指向任意 OpenAI 兼容端点：

```bash
export AGENT_LLM_BASE_URL="https://api.deepseek.com"
export AGENT_LLM_API_KEY="sk-..."      # 或复用 ANTHROPIC_API_KEY
export AGENT_LLM_MODEL="deepseek-v4-flash"
```

未配置 key 时，`llm_client.get_llm()` 自动返回确定性 MockLLM，notebook 依旧完整执行，
所有 Agent 演示代码对 mock 输出都是宽容的。本仓库已内置 `llm_client.py` 统一管理两种模式。

## Reading

每讲的论文 PDF 与研读笔记见 [papers/](papers/)。研读笔记（`NOTES.md`）包含论文核心思想、
关键公式与实验数字、教学主线、代码演示点子、作业点子，是每个 notebook 的直接素材。
论文 PDF 约 150MB 不纳入版本控制，用 `python scripts/download_papers.py` 一键重新下载。

## About

- 课程大纲复刻自 [CS329A: AI Agents](https://cs329a.stanford.edu/)（Stanford, Autumn 2025）
- 风格与教学契约参考 [Modern LLM Notebook](https://github.com/walkinglabs/modern-llm-notebook)
- 完整课程地图：[OUTLINE.md](OUTLINE.md) ｜ 写作规范：[.claude/CLAUDE.md](.claude/CLAUDE.md)
