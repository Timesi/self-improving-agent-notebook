# Self-Improving Agent Notebook

<p align="center">
  <strong>亲手实现 AI Agent：复刻 Stanford CS329A 的 17 本可运行 Jupyter Notebook。</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#curriculum">Curriculum</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#how-it-works">How It Works</a> ·
  <a href="#license">License</a>
</p>

## Overview

本仓库是一门 Agent 系统实践课，课程大纲**完全复刻** Stanford CS329A
（AI Agents, Autumn 2025）。CS329A 的大纲只列出每讲的论文，没有细节内容——
本仓库为每一讲产出一本中文教学 notebook：**先精读该讲的论文，再想象 Stanford 老师会怎么教，
最后用可运行的代码把这节课的核心算法亲手实现出来**。

教学契约与 <a href="https://github.com/walkinglabs/modern-llm-notebook">Modern LLM Notebook</a>
保持一致：

```text
直觉理解 -> 手算验证 -> 代码实现 -> 实验观察
```

每一本 notebook 都不把 Agent 当作黑盒：ReAct 循环、验证器打分、树搜索、记忆分页、评测 harness……
核心算法全部用 numpy / torch 从零实现，LLM 只是循环里的"大脑"。

## Curriculum

| Part | 讲座 | 主题 |
|:---|:---|:---|
| Part 1 · Foundation | L1-L5 | 课程总览、Test-time Compute 缩放、鲁棒验证、工具与代码反馈、多步推理与规划 |
| Part 2 · Training & Evolution | L6-L9 | 训练期缩放与 RL、自改进智能体的开放进化、搜索与深度研究、后训练演进 |
| Part 3 · Agent Engineering | L13/L14/L17 | 软件工程智能体、智能体记忆、Agent 评测与长程任务 |
| Part 4 · Frontiers | L15/L16/L18-L20 | LLM 推理、数学推理智能体、自治系统、多模态机器人、未来方向 |

完整课程大纲见 [OUTLINE.md](OUTLINE.md)。论文与每讲的研读笔记见 [papers/](papers/)。

## Quick Start

```bash
# 创建环境（任选）
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 用 mock 模式跑通任意 notebook（无需 API key，离线可执行）
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
只是 LLM 输出为占位。所有 Agent 演示代码对 mock 输出都是宽容的。

## How It Works

**写作流程**：下载并精读每讲论文 → 产出研读笔记（`papers/lecture-XX/NOTES.md`）→
以"Stanford 老师会怎么教"为主线写出 notebook。

**每本 notebook 的骨架**：

- blockquote 承接上一讲 + 预告本节
- 概念性章节：直觉 → 手算 → 从零实现 → 实验
- `## 小结`：checklist 确认掌握
- `## 作业`：3 道"填空 + assert"小作业，带小提示
- `## 参考资料`：每篇论文链接 + 一句话说明

## License

待定。
