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
- LLM：统一走 `llm_client.py`，使用 DeepSeek OpenAI 兼容端点
  （`AGENT_LLM_BASE_URL` / `AGENT_LLM_API_KEY` / `AGENT_LLM_MODEL`）。没有有效 key 时直接报错。
- 注意：`conda test` 环境是空的，不要使用；notebook 用 `llm_nb_venv` 跑

## 运行与验证

```bash
# 用真实 LLM 验证单个 notebook
/home/devpod/llm_nb_venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/part1-foundation/04-tool-code-feedback.ipynb --output /tmp/out.ipynb
```

英文写作指南见 `.claude/CLAUDE.en.md`；英文 notebook 在 `notebooks-en/`，英文研读笔记为 `papers/lecture-XX/NOTES.en.md`。

## 目录结构

```
self-improving-agent-notebook/
├── README.md / README-CN.md
├── OUTLINE.md / OUTLINE.en.md
├── CLAUDE.md              # 本文件
├── .claude/CLAUDE.md      # 中文 notebook 写作指南
├── .claude/CLAUDE.en.md   # 英文 notebook 写作指南
├── llm_client.py          # 统一真实 LLM client（OpenAI 兼容）
├── requirements.txt       # 依赖
├── papers/lecture-XX/     # NOTES.md（中文）+ NOTES.en.md（英文）+ 论文 PDF
├── notebooks/             # 中文 17 本
└── notebooks-en/          # 英文 17 本，目录结构相同
    ├── part1-foundation/    # 01-05：总览、test-time compute、验证、工具、规划
    ├── part2-training/      # 06-09：RL 缩放、开放进化、搜索、后训练演进
    ├── part3-engineering/   # 10/11/14：SWE 智能体、记忆、评测
    └── part4-frontiers/     # 12/13/15/16/17：推理、数学、自治、机器人、未来
```

## Notebook 间引用规范

- 每个 notebook 自包含，不依赖其他 notebook 的运行时状态
- 前情回顾只引用之前讲的核心概念，不给代码行号
- 按学习顺序连续编号（01-17），与 README、研读笔记和在线阅读器保持一致

## 写作规范摘要

完整规则见 `.claude/CLAUDE.md`（中文 notebook）与 `.claude/CLAUDE.en.md`（英文 notebook）。核心几条：
- 中文 markdown + 中文注释；专有名词（ReAct、Agent、LLM、Self-Attention）保留英文
- 平静教科书语气，用"我们"；禁反问句、感叹号、AI 感修辞
- 每本 notebook 以 `## 小结`（checklist）→ `## 作业`（3 道填空+assert）→ `## 参考资料` 收尾
- LLM 演示统一走 `llm_client.py`，使用真实 API 返回结果
- Agent 核心算法从零实现（numpy/torch），不依赖现成 Agent 框架
