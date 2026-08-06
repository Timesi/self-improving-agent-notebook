export const PATH_STEPS = [
  { num: "01", title: "基础与方法", titleEn: "Foundation", desc: "Agent 循环、Test-time Compute、验证器、工具与规划", descEn: "Agent loops, test-time compute, verification, tools and planning", section: "foundation" },
  { num: "02", title: "训练与进化", titleEn: "Training & Evolution", desc: "强化学习缩放、开放进化、搜索与后训练", descEn: "RL scaling, open-ended evolution, search and post-training", section: "training" },
  { num: "03", title: "智能体工程", titleEn: "Agent Engineering", desc: "软件工程智能体、记忆与评测", descEn: "SWE agents, memory and evaluation", section: "engineering" },
  { num: "04", title: "前沿", titleEn: "Frontiers", desc: "推理、数学证明、自治与机器人", descEn: "Reasoning, math proof, autonomy and robotics", section: "frontiers" },
]

export const RUNNABLE_NOTEBOOKS = [
  { id: "nb-1", lessonId: "01-course-overview", title: "课程总览", titleEn: "Course Overview", desc: "什么是 Agent，一条 Agent 循环", descEn: "What is an agent, one agent loop", section: "foundation", duration: 10 },
  { id: "nb-2", lessonId: "02-test-time-compute", title: "Test-time Compute 缩放", titleEn: "Test-time Compute", desc: "重复采样与 best-of-n", descEn: "Repeated sampling and best-of-n", section: "foundation", duration: 20 },
  { id: "nb-3", lessonId: "04-tool-code-feedback", title: "ReAct 循环与工具使用", titleEn: "ReAct & Tool Use", desc: "从零实现 ReAct 循环", descEn: "Build a ReAct loop from scratch", section: "foundation", duration: 25 },
  { id: "nb-4", lessonId: "05-multi-step-planning", title: "树搜索与规划", titleEn: "Tree Search & Planning", desc: "UCT 树搜索与任务分解", descEn: "UCT tree search and decomposition", section: "foundation", duration: 30 },
  { id: "nb-5", lessonId: "06-train-time-scaling-rl", title: "GRPO 与推理 RL", titleEn: "GRPO & Reasoning RL", desc: "STaR 自举与组内相对 RL", descEn: "STaR bootstrapping and group-relative RL", section: "training", duration: 35 },
  { id: "nb-6", lessonId: "07-open-ended-evolution", title: "自改进智能体的开放进化", titleEn: "Open-Ended Evolution", desc: "让 Agent 设计 Agent", descEn: "Let agents design agents", section: "training", duration: 30 },
  { id: "nb-7", lessonId: "14-agent-memory", title: "智能体记忆", titleEn: "Agent Memory", desc: "分层上下文与 KV 复用", descEn: "Hierarchical contexts and KV reuse", section: "engineering", duration: 25 },
  { id: "nb-8", lessonId: "17-agent-evaluation", title: "Agent 评测", titleEn: "Agent Evaluation", desc: "长时程任务与胜率评测", descEn: "Long-horizon tasks and win-rate", section: "engineering", duration: 25 },
  { id: "nb-9", lessonId: "15-llm-reasoning", title: "LLM 推理", titleEn: "LLM Reasoning", desc: "CoT 与自洽性", descEn: "CoT and self-consistency", section: "frontiers", duration: 20 },
  { id: "nb-10", lessonId: "16-alphaproof-math", title: "数学推理与证明", titleEn: "Math Proof Agents", desc: "验证器与搜索证明定理", descEn: "Verifiers and search prove theorems", section: "frontiers", duration: 28 },
]
