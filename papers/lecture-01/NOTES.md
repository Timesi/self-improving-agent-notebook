English: [NOTES.en.md](NOTES.en.md)

# Lecture 01 — 什么是 AI Agent 研读笔记

> 本文件是 CS329A 第 1 讲（Course Overview）的研读笔记，是编写 `notebooks/part1-foundation/01-course-overview.ipynb` 的素材。
> 来源：https://cs329a.stanford.edu/（Autumn 2025 课程大纲）

> **模板说明**：本讲没有指定论文，是总览性的一讲。按项目要求，模板中的"论文精读"部分改为"概念精读"，逐条拆解本讲要讲清的五个核心概念。

## 课程主题

这一讲要回答的核心问题：**AI Agent 是什么？为什么说它是 LLM 的下一个形态？这门课沿着什么路线，把 Agent 从概念亲手做出来？**

为什么放在第 1 讲：它是整门课的路线图。后续每一讲都是这张地图上的一块——test-time compute（L2）、验证器（L3）、工具与代码反馈（L4）、多步规划（L5）、RL 训练（L6）、开放进化（L7）、搜索与深度研究（L8）、后训练演进（L9）、SWE 工程（L10）、记忆（L11）、评测（L14）、自治与机器人（L15/19）。它们全部服务于同一个对象：让一个 LLM 在循环里做更聪明的事。先给全局坐标，再逐讲填细节。

本讲承担三个任务：

1. 建立 Agent 的直觉与统一视角：LLM 是大脑，循环是骨架，工具与记忆是手脚。
2. 给出一张范式分类表（ReAct / Tool use / Planning / Multi-agent / Memory / Framework），后面每讲都能在表里找到位置。
3. 给出课程地图（4 部分、17 个 notebook），并预告本课的教学契约：直觉理解 → 手算验证 → 代码实现 → 实验观察。

## 概念精读

> 五个概念与 OUTLINE 的章节对应：概念 1、2 →「从 LLM 应用到 Agent」；概念 3 →「Agent 的定义与组成」「一条 Agent 循环」；概念 4 →「Agent 生态地图」；概念 5 →「课程路线」。

### 概念 1：Agent 是什么——从"一次对话"到"目标导向的执行者"

**经典定义（Wooldridge, 1995/1999）**：agent 是居于某个环境中的实体，通过传感器感知环境，通过执行器作用于环境。两个关键词。一是 **autonomy（自主性）**：在没有外部直接干预下，agent 自行决定行动以达成目标。二是 **situatedness（具身/在场）**：它面对的是一个部分可观、不断变化的环境，而不是一个格式良好的输入。这个定义先于 LLM 存在，覆盖软件 agent 与机器人。

**LLM-based Agent 的现代版本（survey 共识，Wang et al. 2023）**：一个以 LLM 为决策核心的系统。它在循环中调用 LLM 完成感知、推理、决策，把决策变成环境中的动作（通常借助工具），把动作结果作为反馈再次输入，直到目标达成。LLM 在这里扮演"大脑"，而不是全部。

**给读者的一句话定义**（notebook 可重复使用）：

> Agent 是一个循环：LLM 根据当前状态决定下一步动作，动作在环境中执行，观察结果再喂回 LLM，直到任务完成。

**与纯 LLM 应用的区别**——本讲最重要的分水岭：

| 维度 | 纯 LLM 应用 | LLM-based Agent |
|---|---|---|
| 调用方式 | 单次（或固定轮次）问答 | 循环，直到终止条件 |
| 状态 | 无（或仅靠会话历史） | 显式维护任务状态与记忆 |
| 动作 | 只输出 token | 输出可执行的工具调用/动作 |
| 环境反馈 | 无 | 有（执行结果回到上下文） |
| 目标 | 生成一个答案 | 完成一个任务（可失败、可重试） |

关键：Agent 不等于"在代码里调 LLM"。自动把问题拼进 prompt 再调一次 API，只是把 LLM 当函数调用；没有循环、状态与动作，那仍是应用而不是 Agent。

### 概念 2：为什么 Agent 是 LLM 的下一个形态

单次 forward pass 的 LLM 有四条硬边界：

1. 只能"说"不能"做"：输出 token，不接触世界；不知道网页、数据库、文件、代码执行器里发生了什么。
2. 上下文窗口有限：长任务、大量中间结果塞不进一次调用。
3. 错了不会自己纠正：单次生成没有执行反馈，无法知道"这个答案跑起来是错的"。
4. 知识静态：训练截止日期之后的新事实无法获取。

Agent 化逐一突破这四条边界：

1. 工具调用把"说"变成"做"——搜索、代码执行、文件读写、GUI（L4 专门讲）。
2. 循环与记忆突破上下文——多步执行、外部记忆（L5 规划、L11 记忆）。
3. 执行反馈引入"运行才知道对错"的强信号——验证器（L3）、代码反馈与 RLEF（L4）。
4. 搜索/采样/推理算力补偿单次生成的质量——test-time compute（L2）。

一句话概括：LLM 提供"智能"，Agent 循环提供"结构"。能力的增量来自循环把环境、工具、记忆、多步算力接入模型，而不是模型本身变强。这解释了为什么 Agent 是"下一个形态"——它把静态的问答模型升级成能自主完成任务、能随训练与进化变强的系统（L6-L9 的训练与进化接上这条线）。

### 概念 3：Agent 循环的组成

最小的闭环骨架（notebook 里是一张图加一个循环类）：

```
感知(Observation) → 决策(Reasoning + Action) → 执行(Act) → 反馈(Feedback) → 回到感知 → 终止
```

1. **感知 Observation**：把环境状态、工具返回、记忆里的相关内容组装成上下文。LLM 没有直接通道看到环境，一切都靠文本化后的 observation 输入。
2. **决策 Reasoning + Action**：LLM 基于当前状态与目标，输出"想什么 + 下一步做什么"。格式可以是 ReAct 的 Thought/Action，也可以是结构化 tool call。这是唯一由模型生成的部分。
3. **行动 Action / 工具执行**：解析动作，在环境中执行。工具是 LLM 与世界的接口（search、python REPL、文件系统、API、GUI）。
4. **反馈 Feedback**：执行结果（含报错）回到上下文，作为下一轮感知。
5. **终止**：模型输出 Final Answer，或达到步数/预算上限。

围绕循环，有几个常被讲成独立模块、其实是循环"配件"的东西：

- **记忆 Memory**：短期是上下文中的消息历史，长期是外部存储（向量库/文件），把循环从上下文窗口解放出来（L11）。
- **规划 Planning**：把目标分解成子目标，或对动作序列做搜索（L5）。
- **验证 Verification**：在循环之外加一个检查环节，让"输出"经过验证才被接受（L3）。

容易混淆的点：Agent 循环不等于多轮对话。ChatGPT 的多轮只是"记住轮次"，没有动作与反馈；Agent 循环的每一轮一定经过环境（哪怕只是读一个文件）。判断标准是中间有没有可执行的、改变世界的动作。

### 概念 4：主流范式分类（Agent 生态地图）

| 范式 | 一句话描述 | 代表工作 / 框架 | 在课程里的位置 |
|---|---|---|---|
| ReAct | 推理（Thought）与行动（Action）交替，把"想"和"做"织进一条轨迹 | Yao et al. 2022 | L4 工具与代码反馈（本课程默认骨架） |
| Tool use / Function calling | LLM 学会输出结构化工具调用，工具结果作为输入 | Toolformer (2023)、OpenAI function calling、MCP | L4；MCP 是工具接口标准 |
| Planning / Search | 先分解任务或对动作空间做搜索，而非走一步看一步 | LATS (2023)、ADaPT、SPRINT | L5 多步规划 |
| Multi-agent | 多个 agent 协作、辩论、分工，Agent 从单体变成社会 | AutoGen (2023)、CrewAI、CAMEL | L6/L7 训练与进化中的合作形态 |
| Memory-based | 显式长期记忆让 Agent 跨会话、跨长任务工作 | MemGPT (2023)、Cartridges | L11 记忆 |
| Agentic frameworks | 把循环、工具、记忆封装成库，工程上"开箱即用" | LangChain/LangGraph、AutoGen、Claude Agent SDK | 全课参照物（本课从零实现，不依赖它们） |
| 特殊形态：代码即行动 / Computer-use / 深度研究 | Agent 通过写代码、操作 GUI、多源检索完成任务 | AlphaCode、OpenAI Computer Use、深度研究 Agent | L8、L10、L13、L16 |

强调两点：

1. 范式不是互斥的。ReAct 只是决策格式，可以叠加规划（先出计划再 ReAct）、叠加记忆、叠加验证器。真实 Agent 通常是多种范式的组合。
2. 生态里"框架"很多，但本课的教学立场是核心循环从零实现（numpy/torch 加自己的工具执行器），框架只做对照演示。这样才能看清每个组件做什么，而不是被框架抽象挡住。

### 概念 5：课程路线（把地图讲给学生）

按 OUTLINE.md，课程共 4 部分、17 个 notebook（L10-12 是中期展示，无教学内容）：

- **Part 1 Foundation（L1-L5）**：从"LLM 只能生成"出发逐层加能力——test-time compute 缩放（L2）、答案验证（L3）、工具与代码反馈（L4）、多步规划（L5）。
- **Part 2 Training & Evolution（L6-L9）**：Agent 不是写死的，要训练与进化——RL 训练期缩放（L6）、开放进化让 Agent 设计 Agent（L7）、搜索与深度研究 Agent（L8）、从 Chatbot 到 Agent 的后训练演进（L9）。
- **Part 3 Agent Engineering（L10/L11/L14）**：把 Agent 做成工程——SWE Agent（L10）、记忆系统（L11）、评测与长程任务（L14）。
- **Part 4 Frontiers（L12/L13/L15-L17）**：边界在哪里——LLM 推理（L12）、数学 Agent（L13）、自治 Agent（L15）、多模态机器人（L16）、未来方向（L17）。

教学契约（本课最重要的一句话）：从这一讲起，每个概念都按 直觉理解 → 手算验证 → 代码实现 → 实验观察 推进。第一课的任务是让学生在下课前写出第一个最小循环。

## 教学主线（想象 Stanford 老师会怎么教）

第一节课老师不会讲论文，会做三件事：建立直觉、给出定义与分类、给出课程地图。

1. **从学生最熟的东西出发：ChatGPT 单轮对话**。现场演示一个 ChatGPT 答不上的任务，比如"把这个仓库里所有过期的 TODO 改掉并跑测试"。指出一个反直觉的事实：LLM 能写出很好的代码片段，却无法自己完成一件事。原因是它只能说、不能做，没有状态、没有工具、没有循环。这一下就把"LLM 应用 vs Agent"的对立立起来。

2. **给出 Agent 的定义**。先给 Wooldridge 的经典定义（situated、autonomous，感知—行动），再落到 LLM 时代：LLM 是大脑，循环是骨架。用一句可复用的话总结：Agent 是一个循环，LLM 根据当前状态决定下一步动作，动作在环境中执行，观察结果再喂回 LLM，直到任务完成。

3. **把一个具体任务走一遍 Agent 循环**。用"查询最新天气并决定要不要带伞"这类例子，手把手过感知 → 决策 → 行动 → 反馈四步，强调每轮一定经过环境。顺势澄清一个高频误解：多轮对话不是 Agent，中间必须有可执行的、改变世界的动作。

4. **给出范式分类与生态地图**。把 ReAct / Tool use / Planning / Multi-agent / Memory / Frameworks 列成表，每个范式配一个代表工作（ReAct→Yao et al.、Toolformer、LATS、AutoGen、MemGPT）。强调范式不互斥、真实 Agent 是组合。

5. **给出课程地图**。把 4 部分 17 个 notebook 摊开，说明每讲往哪个方向补能力：Part 1 补能力（推理算力、验证、工具、规划），Part 2 补"怎么变强"（训练、进化、搜索、后训练），Part 3 补工程化（SWE、记忆、评测），Part 4 补边界（推理模型、数学、自治、机器人）。结尾预告：下课前，你们要亲手写出第一个最小 Agent 循环。

读者最容易卡住的三个地方：

- **Agent vs "在代码里调 LLM"**：自动拼 prompt 再调 API 不是 Agent，缺循环、状态、动作三样。需要一个"同一任务，应用做不了、Agent 能做"的对照。
- **循环 vs 多轮对话**：判断标准是"中间有没有改变世界的动作"。
- **"Agent 框架是必需的"的错觉**：LangChain/AutoGen 是封装不是本体，核心循环几十行就能从零写出来。老师会现场把最小循环从零写一遍来破除这个错觉。

## 代码演示点子（3-6 个）

全部走 `llm_client.get_llm()`（脚本化 模式确定性输出，无 key 可跑）。Agent 循环核心逻辑从零实现，不依赖现成框架。

1. **最小 Agent 循环骨架（从零实现）**：写一个 `AgentLoop` 类，属性是消息历史（状态）、工具注册表、LLM。循环体：调用 LLM → 解析输出 → 若是 `Action` 则执行工具并把 observation 追加进历史 → 若是 `Final Answer` 则终止。用一个确定性玩具环境（如加法工具加反转字符串工具）让 脚本化 模式也能稳定推进。**关键观察**：循环结构本身只有几十行，Agent 与应用的差距在结构而不在模型。

2. **"单次调用 vs 循环"对照**：同一个多步任务，分别用 (a) 纯 LLM 单次调用（脚本化 下给占位回复）和 (b) Agent 循环（能调用工具拿到环境结果）跑，对比两者的输出与上下文长度。**关键观察**：Agent 多出来的能力来自循环与工具反馈，而不是模型本身；复现"LLM 应用做不到、Agent 能做到"的分水岭。

3. **用 llm_client 演示 tool-calling 最简版**：实现 `parse_action(text)` 解析 脚本化/真实 LLM 输出的 ReAct 格式（`Thought: ...` / `Action: name(args)` / `Final Answer: ...`），配一个 `execute(actions, registry)` 执行器，跑 2-3 轮。动作解析要宽容：脚本化 的脚本化轨迹与真实模型的 JSON 都要能消化。**关键观察**：工具调用本质是纯文本协议，格式约定（今天常用 JSON、MCP）只是把这个协议标准化。

4. **Agent 组件可视化（课程地图图）**：用 matplotlib 画感知—决策—执行—反馈的闭环图（英文标签），把记忆、规划、验证作为"配件"挂到对应环节，并给每个组件标注对应讲次（工具→L4，规划→L5，验证→L3，记忆→L11，训练→L6-L9）。**关键观察**：一图把 17 讲安到 Agent 循环上，学生看到的是整门课的地图。

5. **Agent 范式分类卡片**：用字典定义每个范式（ReAct / Tool use / Planning / Multi-agent / Memory / Framework）的关键词、代表工作、一句适用场景；再给几个真实场景让学生选范式。脚本化 下就是一个表格加映射练习。**关键观察**：范式不互斥，一个真实 Agent 常同时是 ReAct + Planning + Memory。

6. **手写一个"多步算术 Agent"（带状态的最小闭环）**：任务等于计算 `(3+5)×(7-2)`。Agent 不能一次算完（脚本化 的算术工具一次只算一步），必须逐步调用 `calc("3+5")` 得到 8 写回上下文，再调用 `calc("8×5")` 得到 40，最后输出 Final Answer。工具执行器自己写，中间状态由我们回填进消息历史。**关键观察**：中间状态由 Agent 自身维护（上下文回填），这正是 Agent 与一次调用的分水岭，也是后续所有讲的基础。

> 演示实现建议：以上全部可用确定性工具加 脚本化LLM 离线完整执行（符合 CLAUDE.md）。演示 4 是纯 matplotlib 图，演示 5 是纯数据/表格，1/2/3/6 需要 脚本化 兼容的解析器（脚本化 的 Thought/Action/Final Answer 格式已在 `llm_client.py` 内置）。

## 作业点子（3 个）

1. **补全最小循环**：填空完成 `AgentLoop.run(max_steps)` 的终止条件与"动作执行 → observation 回填"两段。给定一个固定任务（如"用工具算 7+8，再反转结果字符串"），`assert` 循环在不超过 max_steps 步内给出 Final Answer，且上下文长度随轮次递增。小提示：终止条件有两个，模型输出 Final Answer，或步数用尽；后者要给出可读的失败信息。

2. **动作解析器**：填空实现 `parse_action(text)`，支持 `Action: name(args)` / `Final Answer: ...` / 无动作三种情况，返回 `(kind, name, args)`。用 脚本化 模式的脚本化轨迹与一段真实格式文本各测一次，`assert` 边界情况（多余空白、多行、大写 Action）都能正确解析。小提示：先找 `Final Answer` 再做 `Action`，因为 Final 所在行不该被当成 Action。

3. **单次调用 vs 循环（量化差异）**：对同一任务分别跑单次调用与循环，`assert` 循环的调用次数大于 1 且上下文中包含工具返回值，而单次调用两者皆无；再写一个 `count_tool_calls(trace)` 统计一轮里的工具调用数。小提示：工具返回值靠我们自己回填上下文，所以"循环拿到了工具结果"是可断言的，不依赖具体模型。

## 参考资料

- [Intelligent Agents: Theory and Practice](https://www.csc.liv.ac.uk/~mjw/pubs/ker95.pdf)（Wooldridge & Jennings, The Knowledge Engineering Review, 1995）— Agent 经典定义的出处：situated、autonomous、感知—行动闭环。
- [Intelligent Agents（Multiagent Systems 第 1 章）](https://www.cs.ox.ac.uk/people/michael.wooldridge/pubs/maia-chapter.pdf)（Wooldridge, 1999）— 更完整的 agent 定义，含 BDI 与理性 agent 讨论。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)（Yao et al., ICLR 2023）— 推理与行动交替的范式，本课默认的 Agent 循环骨架。
- [A Survey on LLM-based Autonomous Agents](https://arxiv.org/abs/2308.11432)（Wang et al., 2023）— Agent 组成框架的共识总结（planning / memory / tool use 三大模块）。
- [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/abs/2309.07864)（Xi et al., 2023）— 更全面的 LLM-based Agent 综述与生态分类。
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)（Lilian Weng, 2023）— 把 Agent 循环（planning / memory / tool use）讲得最清晰的一篇博客，适合作为学生的第一份扩展阅读。
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)（Schick et al., 2023）— LLM 学习调用工具的早期代表作。
- [Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models](https://arxiv.org/abs/2310.04406)（Zhou et al., 2023）— 规划/搜索范式（LATS）。
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)（Packer et al., 2023）— 记忆范式。
- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)（Wu et al., 2023）— 多 Agent 协作框架。
- [Model Context Protocol](https://modelcontextprotocol.io)（Anthropic, 2024）— 工具/资源/上下文的标准接口，Agent 生态的"USB-C"。
- [CS329A Course Homepage](https://cs329a.stanford.edu/)（Stanford, Autumn 2025）— 本课课程大纲与作业来源。
