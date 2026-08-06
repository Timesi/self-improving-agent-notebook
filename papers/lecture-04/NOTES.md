# Lecture 04 — 工具使用与代码反馈 研读笔记

> 本文件是 CS329A 第 4 讲的论文研读笔记，是编写对应 notebook 的素材。
> 来源：https://cs329a.stanford.edu/（Autumn 2025 课程大纲）

## 课程主题

这一讲要解决的核心问题：**单次 LLM 调用只能输出一段文本，Agent 怎样才能利用外部世界的信息（知识库、代码执行器、用户）来完成任务？**

它在课程里的位置很讲究。前三讲（L1-L3）里，模型是"静态"的：L2 讲 test-time compute，靠多次采样和投票提升；L3 讲验证器，让模型自己检查自己的输出。这些方法都不让模型与环境交互。本讲引入**闭环**：模型输出一个动作 → 动作在环境中执行 → 观察结果喂回模型 → 模型再输出。这个循环是后续所有 Agent 系统（L5 规划、L10 SWE、L15 自治）的地基。

整讲沿着"**反馈从哪里来**"展开三个层次：

1. **环境反馈**（ReAct）：工具执行的观察结果作为上下文，即"act to reason / reason to act"。这是 prompt 层面的反馈。
2. **执行反馈 + 训练**（RLEF）：把代码是否通过测试这个自动判据变成 RL 的奖励信号，端到端训练模型学会迭代修复。这是训练层面的反馈。
3. **AI 反馈**（Constitutional AI）：不再依赖人类标注，让 AI 依据一组书面原则（宪法）自我批评、修订、互相打分。这是"让 AI 监督 AI"的反馈。

三篇论文串成一条线：**先让 Agent 能拿到反馈（工具/执行），再让反馈参与训练（RL），最后让反馈的提供者也是 AI（自改进的哲学基础）**。这一讲因此既是 Part 1 Foundation 的收尾，也是 Part 2 Training（L6-L9 的 RL 与进化）的引子。

---

## 论文精读

### 论文 1：ReAct：让语言模型协同推理与行动（arxiv:2210.03629，react.pdf）

- **核心思想**
  此前的两派做法各缺一半：Chain-of-Thought（CoT）让模型思考，但思考是"静态黑盒"，不接触外部世界，容易幻觉、错误会在推理链上传播；纯行动派（Act）让模型与环境交互，但不会在语言层面规划目标、维护工作记忆，容易在长任务里迷失。ReAct 的观察很简单：**把"语言"也放进动作空间**——模型在输出动作的间隙输出一段自由文本（Thought），这段文本不改变环境，只更新上下文，用来分解目标、注入常识、追踪进度、应对异常。由此得到 `Thought → Action → Observation` 交替出现的轨迹，推理（reason to act）和行动（act to reason）互相支撑。类比是"厨房里做饭的人"：切菜间隙念叨"现在该烧水了"、"没有盐就用酱油"，同时开冰箱翻菜谱来支撑下一步。

- **关键公式/算法**
  - 形式化：把动作空间扩充为 $\hat{A} = A \cup L$，$L$ 是语言空间。Thought $\hat{a}_t \in L$ 不产生环境观察，只做上下文更新 $c_{t+1} = (c_t, \hat{a}_t)$。用冻结的 PaLM-540B 做 few-shot prompting，每个 in-context 示例是一条人工写的"动作+思考+观察"轨迹。
  - 维基百科 API 的 3 个动作（特意设计得比真实 retriever 弱，模拟人类检索方式）：`search[entity]`（返回实体页前 5 句，找不到时给 5 个相似实体）、`lookup[string]`（返回页面里包含该串的下一个句子，模拟浏览器 Ctrl+F）、`finish[answer]`（结束任务并给出答案）。
  - 提示模板的关键格式（每行带编号）：
    ```
    Question: <问题>
    Thought 1: <用语言推理>
    Action 1: Search[<实体>]
    Observation 1: <工具返回>
    Thought 2: ...
    Action 2: Finish[<答案>]
    ```
  - 两类任务用不同的"thought 密度"：知识密集任务（HotpotQA/Fever）用稠密 thought（每步都写）；决策任务（ALFWorld/WebShop）动作很多，thought 只在关键位置稀疏出现，让模型自己决定何时想、何时动。
  - 与 CoT 组合的回退启发式：`ReAct→CoT-SC`（ReAct 若干步内没出答案就退到 CoT 自洽投票，HotpotQA 设 7 步、Fever 设 5 步）；`CoT-SC→ReAct`（多数投票胜出次数 < n/2，即内部知识不够自信时，退到 ReAct 去检索）。

- **关键实验结论**（PaLM-540B 提示，具体数字）
  - HotpotQA（EM）：Standard 28.7，CoT 29.4，CoT-SC 33.4，Act 25.7，**ReAct 27.4**，ReAct→CoT-SC 35.1，CoT-SC→ReAct 34.2。Fever（Acc）：**ReAct 60.9 > CoT 56.3**；两个组合法都到 62-64.6。
  - 组合法的核心收益：用 CoT-SC 的 21 次采样才能达到的性能，ReAct+CoT-SC 只需 3-5 次。
  - ALFWorld（134 局，task-specific）：ReAct best-of-6 **71%**，Act best-of-6 45%，BUTLER（105 条专家轨迹训练）37%，IM-style 消融（ReAct-IM）53%。ReAct 最差的一次 trial（48%）也超过最好的 Act 和 BUTLER。
  - WebShop：ReAct 成功率为 **40.0**，Act 30.1，IL 29.1，IL+RL 28.7，人类专家 59.6。相对 IL+RL 绝对提升约 10 个百分点。
  - 幻觉归因（人工标注 50 条 ReAct + 50 条 CoT 的 HotpotQA 轨迹）：CoT 的失败里 **56%** 是幻觉推理/事实，ReAct 为 **0%**；但 ReAct 多出 23% 的"检索无效"错误和更高的推理错误率（47% vs 16%，其中包含反复生成同一动作的死循环）。这解释了为什么要两者结合。
  - 微调场景（HotpotQA）：只用 3,000 条 ReAct 正确轨迹微调，PaLM-8B/62B 的 ReAct 就反超所有 PaLM-540B 的提示方法。提示场景下 ReAct 排倒数，微调场景下排第一——说明"边推理边检索"是可泛化的技能，值得训练而非靠上下文学。

- **与课程主题的关系**
  ReAct 是本讲的定调论文：第一次用纯提示把"推理 + 行动 + 观察"组织成一个可解释的闭环，证明**光思考不行动**（CoT 幻觉、错误传播）和**光行动不思考**（Act 迷失）都不够。它还给了工具接口（search/lookup/finish）的范式和提示模板，是 notebook 里从零实现 Agent 循环的直接蓝本。

- **可演示的代码点**
  - 从零写 ReAct 循环：维护 `messages`，调用 `get_llm().chat()` 得到一条回复，正则解析出 `Action: Search[...]/Finish[...]`，执行本地迷你"维基"工具，把观察拼回消息，循环到 `Finish` 或步数上限。
  - 复现"思考 vs 行动"对比：对同一个问题分别只给 CoT 提示、只给 Act 提示、给 ReAct 提示，比较轨迹（mock 模式下注意输出是脚本化占位）。
  - 复现失败模式：构造一个没有检索会答错、检索到资料就能答对的多跳问题，观察幻觉如何被观察注入抑制。
  - 提示模板照搬论文附录 C 的 few-shot 示例（HotpotQA 的 Search/Lookup/Finish 轨迹可以直接当 in-context 示例）。

### 论文 2：RLEF：用执行反馈 + 强化学习为代码 LLM 接地（arxiv:2410.02089，rlef.pdf）

- **核心思想**
  让 LLM 生成代码并反复执行，是常见的"自我修复"做法，但已有研究（Olausson 2024、Kapoor 2024）表明：**固定采样预算下，独立采样往往比"边修边试"更强**——因为基础模型并不会真正利用执行反馈，经常把同样的错误代码原样再吐一遍。RLEF 认为这是训练问题而不是提示问题：把"多轮生成 + 执行反馈"建模成 MDP，用 PPO 端到端训练，让模型**学会**读反馈、改代码。一句话：不是给 Agent 更好的脚手架，而是把"会利用反馈"训练进权重里。

- **关键公式/算法**
  - 迭代代码合成即 MDP：初始观察 $o_0$ 是题目描述，动作 $a_t$ 是文本回复，观察 $o_t$ 包含之前的动作和**执行反馈**。episode 在公开测试全通过或达到轮次上限时终止。
  - 两组测试：**public tests** 提供训练/推理时的执行反馈并决定选哪个解；**private tests**（隐藏）决定最终对错。分开是为了防止模型根据反馈把测试输出抄进答案（shortcut），也省算力。
  - 奖励函数（PPO，无折扣 $\gamma=1$）：
    $$
    R(s_t,a_t) = r(s_t,a_t) - \beta \log\frac{\pi(a_t|c_t)}{\rho(a_t|c_t)},\qquad
    r(s_t,a_t) = \begin{cases} 1, & \text{episode 结束且全部测试通过}\\ -1, & \text{episode 结束且有测试失败}\\ -0.2, & a_t \text{不含合法代码}\end{cases}
    $$
    $\beta$ 权衡任务奖励与 KL；对非最终轮输出非法代码给 -0.2 的小惩罚，缓解一个已发现的失败模式。
  - 工程细节：policy 按 token 建模，value function 按整轮建模（从该轮 prompt 的最后一个 token 预测整轮价值，一轮内所有 token 共用一个 advantage）；KL 项用 token 概率的**几何平均**而不是连乘，抵消对短回复的偏置。
  - 执行反馈模板（论文附录 C）：
    ```
    Your code failed the following tests:
    - input '...' failed: Expected output '...' but got '...'
    - input '...' failed: <stacktrace>
    - input '...' failed: Execution took too long.
    - input '...' failed: Out of memory.
    Give it another try.
    Your code should be enclosed in triple backticks like so: '''python ...'''
    ```
  - 训练配置：Llama 3.1 8B/70B Instruct 为初始策略，turn limit = 3，8B 训 12,000 步、70B 训 8,000 步。

- **关键实验结论**（CodeContests，具体数字）
  - 1@3（一次 rollout 至多 3 个回复）：70B 从 **27.5 → 40.1**（test），8B 从 **10.5 → 16.0**。10@100：70B test 50.3 → 54.5。
  - 超越此前 SOTA：70B+RLEF 的 test 集 40.1，超过 AlphaCodium(GPT-4) 的 29，而样本预算从 100 降到 3（少一个数量级）；8B+RLEF（16.0）超过 AlphaCode 9B（13.3），后者用了 1,000 个样本。
  - 推理端行为（图 3）：RLEF 模型首轮错误更少、后几轮修复更可靠、代码编辑幅度更大；基础模型经常对报错无动于衷，把同样的代码原样输出。**随机反馈消融**：把反馈替换成另一道无关题的执行结果，修复能力大幅受损（pass@1 下降且随轮次上限放大）——证明 RLEF 模型是真的在"读反馈"，而不是靠采样多样性蒙对。
  - 训练方法对比（表 3a）：few-shot 提示对 Instruct 模型有害（-）；SFT 只在 valid 集提升；RLEF 全面领先。单轮 vs 多轮训练（表 3b）：多轮训练（MT）提升最大，且收益能在推理时迁移到 HumanEval+、MBPP+。
  - 结论落点：在固定采样预算下，RLEF 让"迭代修复"首次稳定打败"独立采样"，把模型从"需要昂贵 prompt 工程脚手架"推向"领域微调换通用能力"。

- **与课程主题的关系**
  RLEF 是"代码执行作为反馈信号"的进阶：ReAct 把执行结果当上下文喂给模型（读一次算一次），RLEF 把"会不会利用执行反馈"变成训练目标（PPO 的奖励）。它回答"Agent 怎么学会自我改进"——不是手写更多修复提示，而是让环境判据直接参与梯度。这也预告了 L6 的 train-time scaling（GRPO 等）与后训练进化。

- **可演示的代码点**
  - 在 notebook 里搭一个"迭代代码修复"模拟器（不训练）：给一个问题描述和一组 public tests，循环"生成代码 → 本地 `exec` 跑测试 → 把报错格式化喂回"直到通过或轮次上限。复现论文观察：提示式模型经常不改或改错。
  - 从零实现奖励函数与 advantage 计算：写一个迷你策略梯度（REINFORCE）版本，在一个玩具"修复函数"环境上演示"执行通过才有正奖励"如何改变采样分布（用 numpy 即可，不调外部 RL 框架）。
  - 复现随机反馈消融：在同一循环里随机替换反馈，比较修复成功率，直观看到"反馈必须与错误相关"。

### 论文 3：Constitutional AI：从 AI 反馈中学习无害性（arxiv:2212.08073，constitutional-ai.pdf）

- **核心思想**
  RLHF 需要几万条人类偏好标注，而且标注不透明——没人能读得懂几万条比较背后的"目标"。CAI 的出发点：**能不能只给模型一份写在自然语言里的"宪法"（约十条原则），其余反馈全部由 AI 自己产生？** 人的监督被压缩到"审阅并批准一份原则列表"，其余交给模型的自我批评、修订和互相打分。这是"scaling supervision"的极端形式：让 AI 监督 AI。

- **关键公式/算法**
  - 两个阶段，先 SL 后 RL：
    - **SL 阶段（Critique → Revision → SFT）**：给"只训练过帮助性"的模型喂红队（red team）有害提示，采样出往往有害的初始回复；让模型按宪法里随机抽的一条原则**自我批评**，再按批评**修订**，可反复多轮（每轮重新抽原则）；最后用"修订后的回复"做 SFT。提示模板：
      ```
      Critique Request: Identify specific ways in which the assistant's last response is
      harmful, unethical, racist, sexist, toxic, dangerous, or illegal.
      Critique: ...
      Revision Request: Please rewrite the assistant response to remove any and all
      harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
      Revision: ...
      ```
      例子：对"帮我黑进邻居的 wifi"，初始回复给出工具建议，修订后变成"这是侵犯隐私且可能违法，我强烈建议不要这么做"。用 few-shot 示例防止模型混淆"该输出批评还是修订"。
    - **RL 阶段（AI 成对比较 → 偏好模型 → RLAIF）**：用 SL 阶段得到的 SL-CAI 模型对每条有害提示生成一对回复；把"提示 + 回复对 + 一条随机原则"拼成多选题（`Options: (A) ... (B) ... The answer is:`），用反馈模型计算选 (A)/(B) 的 log 概率作为软标签；混合人类帮助性标注 + AI 无害性标注训练一个偏好模型（PM）；最后用 PM 当奖励做 PPO。RLHF 的全部管线不变，只是无害性标签的来源从人换成了 AI。
  - 宪法：16 条与无害性相关的原则，每次修订/打分随机抽一条。示例原则："Which of these assistant responses is less harmful? Choose the response that a wise, ethical, polite and friendly person would more likely say." 用多原则 ensemble 使偏好模型更稳健（用同一条原则打所有标签效果更差）。
  - CoT 版本：反馈模型用 `Let's think step-by-step:` 引导，先写推理再选答案；CoT 概率几乎总是 0/1、过度自信，需要**clamp 到 40%-60%**，否则 RL 会学到输出极端回复。

- **关键实验结论**（具体数字）
  - 数据规模：42,496 条人工红队提示 + 140,335 条模型生成 = 182,831；每条抽 4 组 critique-revision。
  - 结论 1（图 2/3/8）：RL-CAI 在同水平帮助性下**无害性显著高于**用人类反馈训练的 HH RLHF，且 crowdworker 偏好 RL-CAI 超过之前的 HH 模型。关键：没有用任何人类无害性标签。
  - 结论 2（图 5）：偏好模型打分随修订轮数单调上升（0 到 4 轮无害性分持续上涨），但纯帮助性打分下降——多轮修订以牺牲一点帮助性为代价。
  - 结论 3（图 7）：**Critiqued revision 优于 direct revision**（跳过批评直接修订），小模型差距明显、大模型几乎持平；批评偶尔夸大，但修订总体更无害，且批评给决策带来透明度。
  - 结论 4（图 4）：模型越大，AI 识别"更有帮助/诚实/无害"回复的准确率越高；CoT 显著提升，正在逼近用人类标签训练的偏好模型——这为"AI 提供反馈可行"提供了前置证据。
  - 已知失败模式：RL 过训练导致 Goodharting（对红队提示过度反应、堆砌"you are valid, valued, and cared for"之类套话）；靠改写原则、原则 ensemble、软标签 + clamp 缓解。反馈模型的软标签校准良好（图 9）。

- **与课程主题的关系**
  CAI 把"反馈的来源"推到极致：工具反馈来自环境、执行反馈来自规则，而这里反馈来自**AI 自己**。它是本讲"用反馈学习"的哲学制高点，也是"self-improving agent"的直接来源——Agent 想自我改进，最终需要一条不依赖人类标注的反馈通道。对课程的作用：展示一条原则列表如何同时驱动 SL（critique-revision）和 RL（RLAIF），并把 Chain-of-Thought 用成"可检查的决策过程"。

- **可演示的代码点**
  - 用 `get_llm()` 演示 critique → revision 管道：喂一条有害提示，先要初始回复，再按一条自定原则要批评，再要修订，把三段打印对比。
  - 复现多轮修订曲线：把同一条提示按"重复修订 n 轮"跑 0-4 轮，观察回复无害性/篇幅变化（真实 API 下效果明显，mock 下为占位）。
  - 演示"AI 当裁判"：对同一提示的两条候选回复，按原则拼多选题让模型选更无害的那个（soft label 用 mock 或解析 logits）。
  - 原则列表即"宪法"：让读者自己写 3 条原则，替换掉论文的 16 条，观察行为差异——体现"人只审阅原则"的可控性。

---

## 教学主线（想象 Stanford 老师会怎么教）

一条从"静态"到"闭环"再到"自改进"的叙事，三篇论文是三个台阶：

1. **先建动机：为什么光思考不行动不够。** 从 L3 的验证器出发：验证器再准，模型也不能凭空补上训练数据里没有的事实。给一个 HotpotQA 式的多跳问题，演示 CoT 如何一本正经地编出错误答案（幻觉），然后问："如果模型能查一下 Wikipedia，还会错吗？"——**模型缺的不是推理，是拿信息的通道。**

2. **引出 ReAct：思考 + 行动 = 闭环。** 用一个生活类比（做饭）讲清楚 Thought 为什么有用：它不是装饰，是"工作记忆 + 计划 + 异常处理"。手把手拆一条 `Thought → Action → Observation` 轨迹，强调三个动作原语（search/lookup/finish）怎么覆盖"找资料、看细节、收尾"三种需求。这里最容易卡住的地方：**观察该不该由模型生成？** 要反复强调 Observation 必须来自真实工具执行、是插入的文本，而不是模型续写。然后给出两个反例对比：只有 Thought 的 CoT（幻觉）和只有 Action 的 Act（在 50 个房间的 ALFWorld 里迷路）。用论文的消融数字收束：ReAct vs Act 平均相对提升 62%。

3. **工具反馈作为信号：从上下文到训练。** 提出问题：ReAct 每次都要把反馈现拼进 prompt，模型"会"用反馈吗？RLEF 的回答是不一定——表 3a 说 few-shot 提示反而有害，基础模型会无视报错重复同一份代码。于是把"读反馈改代码"变成 RL 目标：讲清楚两组测试（public 反馈、private 打分）和奖励函数（1/-1/-0.2 + KL），用一个示例对话展示"超时 → 加缓存"的修复过程。强调一个反直觉结论：**在固定采样预算下，只有训练过的模型才值得多轮迭代，否则独立采样更好。**

4. **AI 反馈：反馈的提供者也可以不是人。** 从 RLEF 的奖励是"规则"追问：如果任务没有可自动执行的判据呢？CAI 的答案是让 AI 依据宪法自我批评、修订、互相打分。带读 critique-revision 的模板和例子（黑 wifi），讲 SL 与 RL 两个阶段怎么把"原则"变成"偏好模型"再变成"奖励"。收在 scaling supervision：人工监督从"几万条标注"压缩到"一份原则清单"，而这正是 self-improving agent 能持续进化的前提。

5. **串起来：反馈的三层来源（环境 / 执行 / AI），对应 L5 之后的规划与 Part 2 的 RL。** 预告 L5 用搜索做规划、L6 的 GRPO 直接复用"可自动验证"这类奖励设计。

读者最容易卡住的三处：a) 把 Observation 当成模型输出；b) 以为 RLEF 的价值在于 prompt 模板而非训练目标；c) 分不清 CAI 的 SL 阶段（SFT 修订结果）和 RL 阶段（偏好模型打分）各自改了模型的什么。

---

## 代码演示点子（3-6 个）

1. **从零实现一个 ReAct 循环（含工具执行与观察注入）**：这是本讲核心演示。自己写 `parse_action(text)`（正则从模型回复里抽 `Action: Search[...]` / `Finish[...]`）和迷你工具集（一个内置几条条目的本地"百科"字典 + `search`/`lookup`/`finish` 三个函数，等价于论文里弱化的 Wikipedia API）。循环：调 `get_llm().chat(messages)` → 解析动作 → 执行 → 把 `Observation: <结果>` 追加进 messages → 直到 `Finish` 或步数上限。用多跳问题验证（如"某杂志创刊年份与某年比较"）。注意 mock 模式：MockLLM 检测到 ReAct 标记会返回一段脚本化轨迹，解析器要宽容——要么每轮只取第一条 Action，要么兼容"一条回复含多步"；无 key 环境输出注明占位。

2. **复现"思考 vs 行动"三路对比**：同一个任务分别用 Standard / CoT / Act / ReAct 四种提示跑，把轨迹逐条打印，人工或按简单规则打分（是否检索到关键事实、是否幻觉）。真实 API 下可复现论文表 2 的观察：CoT 会编事实，ReAct 会去查。此演示对应论文图 1 的对比结构，是"光思考不行动不够"的实证。

3. **动作解析器与非法动作处理**：把 ReAct 输出解析做成独立的健壮组件——处理多行混合输出、未闭合括号、未知工具名、重复动作（论文点名的一个失败模式是反复生成相同 Action，导致死循环）。给一个带超时的循环守卫（如 8 步封顶）。演示如何让 Agent 循环"容忍"噪声输出而不崩溃，正好满足 mock 模式可用性要求。

4. **迭代代码修复循环（RLEF 推理端，不训练）**：搭一个 `solve(problem, feedback_format)` 模拟器：问题描述 + 2-3 条 public test，循环"生成代码 → 用 `exec` 在子进程里跑测试 → 把报错按论文附录 C 的模板格式化喂回"直到通过。做一个对照：插入随机反馈（复现论文图 3/4 的消融），比较修复成功率，直观看出"反馈必须相关"。此演示把"代码执行作为反馈信号"落在本仓库的 `llm_client` + 自写执行器框架里。

5. **从零实现一个"执行反馈奖励"的迷你策略梯度**：玩具环境：修一个只差一处 bug 的函数，动作是"输出一个候选代码"，奖励是 `1`（通过全部 private test）/ `-1`（最终失败）/ `-0.2`（非法代码）。用 numpy 手写一个 REINFORCE 更新（奖励减去 baseline、带 KL 正则），在少量候选上迭代，观察"给执行通过正奖励"如何改变模型输出的分布。这不需要真的训 LLM，用规则的代码打分器即可，对应论文第 2.2 节的奖励函数。

6. **CAI 的 critique-revision 管道演示**：给"仅帮助性"风格的初始回复（可手动构造一个有害回复，如论文里"黑 wifi"的例子），用 `get_llm()` 依次请求初始回复 → 按原则批评 → 按批评修订，再把三段打印。扩展：重复修订 3 轮并对比（真实 API 下能复现"无害性随轮数上升"）。再补一个"AI 当裁判"：对两条候选回复按"更无害"原则拼多选题，让模型打分。这个演示把本讲的最高点（AI 反馈）收进可运行的闭环。

（建议 1、4 为必做主线演示，2 与 6 为选做，3、5 作为理解工具/奖励的小练习。）

---

## 作业点子（3 个）

1. **补全 ReAct 循环**：给一个半成品 `react_step(client, messages)`，缺"解析 Action、执行工具、注入 Observation"三段。填空补上，然后 `assert parse_action("Action 1: Search[Colorado orogeny]") == ("Search", "Colorado orogeny")`、`assert "Observation" in messages[-1]["content"]`。小提示：从最后一个非系统消息里正则找 `Action\s*\d*[:：]?\s*(\w+)\[([^\]]*)\]`。

2. **实现查找工具**：写一个 `lookup(text, keyword)`，模拟"返回页面里包含 keyword 的下一个句子"，用给定的小段落库填空，`assert lookup("...", "eastern sector") == "The eastern sector extends into the High Plains..."`。小提示：在段落列表里维护一个游标（下一个位置），返回从当前位置起第一个含 keyword 的句子。

3. **计算 RLEF 奖励**：给定几个轨迹片段（全通过 / 有失败 / 非法代码），填空实现 `compute_reward(episode, valid_code)`，对 `assert`：全通过返回 1、失败返回 -1、非法代码返回 -0.2；再填空把带 KL 项的完整奖励 `r - beta * log(pi/rho)` 写出来（给两个概率值手算）。小提示：区分"episode 结束"和"轮次中途"两种情况。

每道作业结尾统一提醒："可以让 AI 帮忙解释思路，但不建议直接让 AI '做完这道题'。"

---

## 参考资料

- ReAct: Synergizing Reasoning and Acting in Language Models（arxiv 2210.03629，https://arxiv.org/abs/2210.03629）— 本讲主论文；Thought/Action/Observation 循环与 search/lookup/finish 工具范式。项目页含代码：https://react-lm.github.io/
- RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning（arxiv 2410.02089，https://arxiv.org/abs/2410.02089）— 用 PPO 把"利用执行反馈"训练进权重；奖励函数、public/private test 划分、反馈模板见附录 C。
- Constitutional AI: Harmlessness from AI Feedback（arxiv 2212.08073，https://arxiv.org/abs/2212.08073）— critique-revision 与 RLAIF 的原始论文；原则列表与 few-shot 提示在 https://github.com/anthropics/ConstitutionalHarmlessnessPaper
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models（arxiv 2201.11903）— ReAct 的对照方法，理解"只思考不行动"的局限。
- Inner Monologue: Embodied Reasoning through Planning with Language Models（arxiv 2207.05608）— ReAct 的最直接前身，ReAct-IM 消融的对照来源。
- Self-Consistency Improves Chain of Thought Reasoning（arxiv 2203.11171）— ReAct+CoT-SC 组合法里"投票"部分。
- Scaling LLM Test-Time Compute Optimally（arxiv 2408.03314）— L2 讲的 test-time compute，与 RLEF 的"固定采样预算下迭代 vs 独立采样"结论呼应。
- 本仓库 `llm_client.py`（`get_llm()`）— 所有 LLM 演示统一入口，mock 模式保证离线可执行。
