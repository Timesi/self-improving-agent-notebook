# CLAUDE.md — Self-Improving Agent Notebook 项目规范

## 项目定位

这是一套**亲手实现 AI Agent 核心组件与循环**的 Jupyter Notebook 教程。
课程大纲完全复刻 Stanford CS329A（AI Agents，Autumn 2025）。
目标读者是有 Python 基础、想深入理解 Agent 系统的工程师。
写作时默认读者是"有耐心、愿意思考的高中生"：每个新概念都用一句清楚定义 + 一个具体例子带进去。

教学契约：**直觉理解 → 手算验证 → 代码实现 → 实验观察**。

## 环境

- 运行环境：`/home/devpod/llm_nb_venv`（python 3.13，torch 2.10.0+rocm，ROCm 7.13）
- GPU：4 × AMD Instinct MI300X（ROCm，非 NVIDIA；`torch.cuda.is_available()` 为 True）
- LLM：统一走 `llm_client.py`。真实 API 用 DeepSeek OpenAI 兼容端点
  （`AGENT_LLM_BASE_URL` / `AGENT_LLM_API_KEY` / `AGENT_LLM_MODEL`，回退到 `ANTHROPIC_API_KEY`）；
  `AGENT_LLM_MOCK=1` 或无 key 时用确定性 MockLLM。
- 注意：`conda test` 环境是空的，不要使用；notebook 用 `llm_nb_venv` 跑

## 运行与验证

```bash
# 用 llm_nb_venv 验证单个 notebook（mock 模式，离线可跑）
LLM_MOCK=1 /home/devpod/llm_nb_venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/part1-foundation/01-course-overview.ipynb --output /tmp/out.ipynb

# 真实 LLM 模式（需要 key，通常只抽查）
/home/devpod/llm_nb_venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/part1-foundation/04-tool-code-feedback.ipynb --output /tmp/out.ipynb
```

## 目录结构

```
self-improving-agent-notebook/
├── README.md              # 课程总览 + 导航
├── OUTLINE.md             # 完整课程大纲（复刻 CS329A）
├── CLAUDE.md              # 本文件
├── .claude/CLAUDE.md      # Agent 写作指南（风格规范，写 notebook 时严格遵守）
├── llm_client.py          # 统一 LLM client（API + mock 双模式）
├── requirements.txt       # 依赖
├── papers/lecture-XX/     # 每讲论文 PDF + NOTES.md（研读笔记）
└── notebooks/
    ├── part1-foundation/    # 01-05：总览、test-time compute、验证、工具、规划
    ├── part2-training/      # 06-09：RL 缩放、开放进化、搜索、后训练演进
    ├── part3-engineering/   # 13/14/17：SWE 智能体、记忆、评测
    └── part4-frontiers/     # 15/16/18/19/20：推理、数学、自治、机器人、未来
```

## Notebook 间引用规范

- 每个 notebook 自包含，不依赖其他 notebook 的运行时状态
- 前情回顾只引用之前讲的核心概念，不给代码行号
- 按课程讲座号编号（01-20），与 CS329A 课程地图对应；L10-12 为中期展示，无 notebook

## 写作规范摘要

完整规则见 `.claude/CLAUDE.md`。核心几条：
- 中文 markdown + 中文注释；专有名词（ReAct、Agent、LLM、Self-Attention）保留英文
- 平静教科书语气，用"我们"；禁反问句、感叹号、AI 感修辞
- 每本 notebook 以 `## 小结`（checklist）→ `## 作业`（3 道填空+assert）→ `## 参考资料` 收尾
- LLM 演示统一走 `llm_client.py`，mock 模式下必须能完整执行
- Agent 核心算法从零实现（numpy/torch），不依赖现成 Agent 框架
