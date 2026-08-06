# Self-Improving Agent Notebook 课程大纲

复刻 Stanford CS329A（AI Agents，Autumn 2025）。按课程讲座号组织，跳过中期展示（L10-L12）。

Part 总数：4
Notebook 总数：17

---

## Part 1 · Foundation（基础与方法）— L1-L5

### 01-course-overview.ipynb — 课程总览：什么是 AI Agent

- 1. 从 LLM 应用到 Agent
- 2. Agent 的定义与组成
- 3. 一条 Agent 循环
- 4. Agent 生态地图
- 5. 课程路线
- 小结
- 作业

### 02-test-time-compute.ipynb — Test-time Compute 缩放

- 1. 训练完成后的算力：test-time compute
- 2. 重复采样：Large Language Monkeys
- 3. 从采样到投票：self-consistency 与 best-of-n
- 4. Compute-optimal 缩放：Snell 定律
- 5. 方法组合与架构搜索：Archon
- 小结
- 作业
- 参考资料

### 03-robust-verification.ipynb — 答案验证：让模型检查模型

- 1. 生成与验证的差距
- 2. 验证器的训练：Cobbe 的数学验证器
- 3. 结果验证器（ORM）与过程验证器（PRM）
- 4. Step-by-step 验证：Lightman
- 5. 无人工标注的步级验证：Math-Shepherd
- 6. 验证器与采样结合
- 小结
- 作业
- 参考资料

### 04-tool-code-feedback.ipynb — 工具使用与代码反馈

- 1. 从推理到行动：ReAct 循环
- 2. 工具的定义与调用
- 3. 代码执行作为反馈信号
- 4. RLEF：用执行反馈做强化学习
- 5. Constitutional AI：用 AI 反馈对齐 AI
- 小结
- 作业
- 参考资料

### 05-multi-step-planning.ipynb — 多步推理与规划

- 1. 单步推理的局限
- 2. 任务分解：ADaPT
- 3. 树搜索：LATS
- 4. 并行规划与执行：SPRINT
- 5. 自适应分支：Wider or Deeper
- 小结
- 作业
- 参考资料

---

## Part 2 · Training & Evolution（训练与进化）— L6-L9

### 06-train-time-scaling-rl.ipynb — 训练期缩放与强化学习

- 1. 训练期 vs 测试期缩放
- 2. STaR：用推理自举推理
- 3. DeepSeekMath 与 GRPO
- 4. DAPO：开源大模型 RL 系统
- 5. 从强化到 Agent 训练
- 小结
- 作业
- 参考资料

### 07-open-ended-evolution.ipynb — 自改进智能体的开放进化

- 1. 让 Agent 自己设计 Agent：ADAS
- 2. 自动化的科学发现：AI Scientist
- 3. 代码作为 Agent 基因组：AlphaEvolve
- 4. 开放进化的失败模式与风险
- 小结
- 作业
- 参考资料

### 08-search-deep-research.ipynb — 搜索与深度研究智能体

- 1. 程序合成中的搜索：AlphaCode
- 2. AlphaCode 2：从采样到过滤
- 3. 智能体搜索增强推理：Search-o1
- 4. 深度研究的工作流
- 小结
- 作业
- 参考资料

### 09-post-training-evolution.ipynb — 后训练演进：从 Chatbot 到 Agent

- 1. 后训练是什么
- 2. SFT 与 RLHF：Chatbot 时代
- 3. Agent 化后训练：工具、执行、反馈
- 4. 演进路线图
- 小结
- 作业
- 参考资料

---

## Part 3 · Agent Engineering（智能体工程）— L10, L11, L14

### 10-swe-agents.ipynb — 软件工程智能体

- 1. 代码任务中的 test-time compute：CodeMonkeys
- 2. 让 LLM 写高效内核：KernelBench
- 3. Agent-System 接口设计
- 4. SWE-Agent 循环的构建
- 小结
- 作业
- 参考资料

### 11-agent-memory.ipynb — 智能体记忆

- 1. 记忆为什么重要
- 2. MemGPT：LLM 作为操作系统
- 3. Cartridges：长上下文表示的自学习
- 4. CacheBlend：RAG 的 KV 缓存复用
- 5. 记忆系统的工程实践
- 小结
- 作业
- 参考资料

### 14-agent-evaluation.ipynb — Agent 评测与长程任务

- 1. 评测 Agent 的难点
- 2. 长程任务评测：Measuring Long Tasks
- 3. 真实经济价值任务：GDPVal
- 4. 深度研究评测：DeepScholar-Bench
- 5. 构建自己的评测 Harness
- 小结
- 作业
- 参考资料

---

## Part 4 · Frontiers（前沿）— L12, L13, L15-L17

### 12-llm-reasoning.ipynb — LLM 推理

- 1. 推理能力从哪来
- 2. Chain-of-Thought 与自洽性
- 3. 推理的涌现
- 4. 推理模型的发展
- 小结
- 作业
- 参考资料

### 13-alphaproof-math.ipynb — 数学推理智能体

- 1. 数学：推理的试金石
- 2. AlphaGeometry：几何的神经-符号结合
- 3. AlphaProof：形式化证明的 RL
- 4. IMO 金牌：Gemini 的路线
- 小结
- 作业
- 参考资料

### 15-autonomy-agents.ipynb — 构建自治智能体

- 1. 从演示到自治
- 2. 可靠性：错误检测与恢复
- 3. 监督与信任边界
- 4. 开放问题
- 小结
- 作业
- 参考资料

### 16-multimodal-robotics.ipynb — 多模态机器人智能体

- 1. 具身智能与 VLA
- 2. 视觉-语言-动作模型
- 3. 机器人数据的收集
- 4. 从物理世界学习的反馈
- 小结
- 作业
- 参考资料

### 17-future-research.ipynb — 未来研究方向

- 1. 当前系统的边界
- 2. 开放问题清单
- 3. 可能的研究路线
- 4. 如何参与
- 小结
- 作业
- 参考资料

---

## 课程编号

课程按学习顺序连续编号为 L1–L17。Notebook 文件名、研读笔记目录和在线阅读器使用相同编号。
