const n=`{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "912bd81cfbfe410a9a0dc82f8edcb65d",
   "metadata": {},
   "source": "# 从推理到行动：工具使用与代码反馈\\n"
  },
  {
   "cell_type": "markdown",
   "id": "080e4da0b1694d759a68463cca2954f1",
   "metadata": {},
   "source": "> 前两讲处理的是静态模型：02 讲把多次采样与投票用在推理阶段，03 讲让模型检查模型的输出。这些方法都不让模型与环境交互，模型知道多少，取决于训练数据里有多少。\\n>\\n> 这一讲把模型与环境连成一个闭环：输出动作、执行工具、读回观察。我们按反馈的来源分成三个层次：环境反馈（ReAct）、执行反馈（RLEF）、AI 反馈（Constitutional AI）。\\n"
  },
  {
   "cell_type": "markdown",
   "id": "8489fdb08f0e41e5ba761823a948461a",
   "metadata": {},
   "source": "上一讲把 Agent 定义成一个循环：模型决定动作，动作在环境中执行，观察再喂回模型。这一讲要解决循环里一个具体的问题——模型到底能拿到什么样的反馈，又怎样把反馈用起来。\\n\\n反馈的来源决定了 Agent 能做什么。检索接口把外部文本放进上下文；代码执行把程序的真实行为变成判据；模型自己也能依据一套原则生成批评。这三类反馈对应本讲的三篇论文，递进关系是：先让 Agent 能拿到反馈，再让反馈参与训练，最后让反馈的提供者也是 AI。这一讲也是基础篇的收尾，给后面的规划与强化学习铺好\\"闭环\\"这块地基。\\n\\n第一节从一个具体问题开始：模型只会输出文本，怎样让输出变成一个可以执行的检索动作，并让返回的结果重新进入模型的视野。\\n"
  },
  {
   "cell_type": "markdown",
   "id": "bbedb1ace85348f2abd36d884d927e8e",
   "metadata": {},
   "source": "## 1. 从推理到行动：ReAct 循环\\n\\n只思考不行动是不够的。Chain-of-Thought（CoT）让模型在给出答案前先写一段推理，但它不接触外部世界，依赖的是模型记住的事实。当问题需要模型不知道的知识，CoT 会编出看起来合理的答案；推理链越长，错误沿链传播的可能越大。在 ALFWorld 的 134 局里，ReAct 的最好 6 次取均值达到 71%，纯行动的 Act 只有 45%——思考让行动不迷失方向。\\n\\nReAct 的观察是：把语言也放进动作空间。模型在输出动作的间隙输出一段自由文本（Thought），这段文本不改变环境，只更新上下文，用来分解目标、记录进度、应对异常。由此得到 Thought → Action → Observation 交替出现的轨迹，推理支撑行动，行动也支撑推理。轨迹里的 Observation 必须来自真实工具执行，是插入的文本，而不是模型续写。\\n\\n先打印一条论文风格的轨迹，认识循环的三种片段。\\n"
  },
  {
   "cell_type": "code",
   "id": "35787aa898894dcbb9ccadcc1f472baa",
   "metadata": {},
   "source": "# 一条 ReAct 轨迹，按角色逐行打印，认识循环的三种片段\\ntrajectory = [\\n    (\\"Thought\\", \\"我需要先查到这本杂志的创刊年份。\\"),\\n    (\\"Action\\", \\"Search[现代文学]\\"),\\n    (\\"Observation\\", \\"《现代文学》是 1923 年创刊于上海的文学刊物。\\"),\\n    (\\"Thought\\", \\"创刊年份是 1923 年，晚于 1919 年。\\"),\\n    (\\"Action\\", \\"Finish[之后]\\"),\\n]\\n\\nfor kind, content in trajectory:\\n    print(f\\"{kind:12s}{content}\\")\\nprint()\\nprint(\\"关键观察：Action 会调用工具，Observation 来自工具执行，\\")\\nprint(\\"Thought 只更新上下文，不直接产生观察。\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "f4ee83db3cc345b280210753210143ca",
   "metadata": {},
   "source": "## 2. 工具的定义与调用\\n\\n形式化地看，ReAct 把动作空间从 $A$ 扩充为 $\\\\hat{A} = A \\\\cup L$，其中 $L$ 是语言空间。Thought $\\\\hat{a}_t \\\\in L$ 不产生环境观察，只更新上下文 $c_{t+1} = (c_t, \\\\hat{a}_t)$；Action $a_t \\\\in A$ 在环境中执行，产生观察 $o_{t+1}$。\\n\\n论文为维基百科设计了三个动作，刻意比真实检索器弱，模拟人类的检索方式：\\n- \`search[实体]\`：返回实体页面的前几句；找不到时给出若干相似实体。\\n- \`lookup[字符串]\`：返回页面里包含该字符串的下一个句子，相当于浏览器里的 Ctrl+F。\\n- \`finish[答案]\`：结束任务并给出答案。\\n\\n三个动作覆盖了\\"找资料、看细节、收尾\\"三种需求。下面的迷你百科用同一个接口，内置几条本地条目，让循环可以在没有网络的环境里跑起来。\\n"
  },
  {
   "cell_type": "code",
   "id": "95f2606dc58c4f0e8023c09ec8c8abe6",
   "metadata": {},
   "source": "class MiniWiki:\\n    \\"\\"\\"一个迷你百科：内置几条本地条目，模拟论文里的弱化检索接口。\\n\\n    与论文动作对应：search(实体) 返回页面开头，lookup(关键词) 返回\\n    当前页里包含关键词的下一个句子。lookup 用游标记录搜索位置。\\n    \\"\\"\\"\\n\\n    def __init__(self):\\n        self.pages = {\\n            \\"现代文学\\": [\\n                \\"《现代文学》是 1923 年创刊于上海的文学刊物。\\",\\n                \\"鲁迅、茅盾等作家曾在该刊物上发表作品。\\",\\n                \\"刊物出版延续到 1930 年代初。\\",\\n            ],\\n            \\"五四运动\\": [\\n                \\"五四运动发生于 1919 年 5 月 4 日。\\",\\n                \\"运动以北京学生游行开始，随后扩展到全国。\\",\\n                \\"它被视为中国现代史的开端之一。\\",\\n            ],\\n        }\\n        self.current_page = []\\n        self.pos = 0\\n\\n    def search(self, entity):\\n        \\"\\"\\"返回实体页面开头两句；找不到时给出相似实体名。\\"\\"\\"\\n        self.current_page = self.pages.get(entity, [])\\n        self.pos = 0\\n        if self.current_page:\\n            return \\" \\".join(self.current_page[:2])\\n        similar = [name for name in self.pages if entity in name]\\n        return f\\"未找到「{entity}」，相似实体：{similar if similar else '无'}\\"\\n\\n    def lookup(self, keyword):\\n        \\"\\"\\"从游标起返回第一个含关键词的句子，找不到返回提示。\\"\\"\\"\\n        for i in range(self.pos, len(self.current_page)):\\n            if keyword in self.current_page[i]:\\n                self.pos = i + 1\\n                return self.current_page[i]\\n        self.pos = len(self.current_page)\\n        return \\"未找到包含该关键词的句子\\"\\n\\n\\nwiki = MiniWiki()\\nprint(wiki.search(\\"现代文学\\"))\\nprint(wiki.lookup(\\"创刊\\"))\\nprint(wiki.search(\\"一个不存在的实体\\"))\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "1bac5fa4a4514d3cb6c3c066b43c1acd",
   "metadata": {},
   "source": "**动作解析器**\\n\\n模型的输出是自由文本，需要解析成结构化动作才能执行。解析器要宽容：既接受论文格式 \`Action: Search[实体]\`，也接受函数调用格式 \`Action: search(\\"实体\\")\`；一条回复里可能连着写多个动作，我们把它们按顺序排队，逐个执行。终止信号有两种：\`Finish[答案]\` 动作，或单独的 \`Final Answer: ...\` 行。\\n"
  },
  {
   "cell_type": "code",
   "id": "4213335744bd4e179c2683d2750d4db8",
   "metadata": {},
   "source": "import re\\n\\n\\ndef parse_actions(text):\\n    \\"\\"\\"从模型回复中按顺序取出所有动作指令。\\n\\n    支持两种格式：Search[实体] 与 search(\\"实体\\")。\\n    返回 [(工具名, 参数), ...]，工具名统一转小写。\\n    \\"\\"\\"\\n    pattern = (\\n        r\\"Action\\\\s*\\\\d*\\\\s*[:：]\\\\s*([A-Za-z]+)\\"\\n        r\\"\\\\s*(?:\\\\[([^\\\\]]*)\\\\]|\\\\(\\\\s*(?:\\\\\\"([^\\\\\\"]*)\\\\\\"|'([^']*)'|([^)]*))\\\\s*\\\\))\\"\\n    )\\n    actions = []\\n    for m in re.finditer(pattern, text):\\n        name = m.group(1).lower()\\n        arg = (m.group(2) or m.group(3) or m.group(4) or m.group(5) or \\"\\").strip()\\n        actions.append((name, arg))\\n    return actions\\n\\n\\ndef extract_final_answer(text):\\n    \\"\\"\\"提取回复里的最终答案，找不到返回 None。\\n\\n    识别 Final Answer 标记与 Finish[答案] 动作。\\n    \\"\\"\\"\\n    m = re.search(r\\"(?:Final Answer|final answer)\\\\s*[:：]\\\\s*([^\\\\n]+)\\", text)\\n    if m:\\n        return m.group(1).strip()\\n    m = re.search(r\\"Action\\\\s*\\\\d*\\\\s*[:：]\\\\s*Finish\\\\s*\\\\[([^\\\\]]*)\\\\]\\", text)\\n    if m:\\n        return m.group(1).strip()\\n    return None\\n\\n\\npaper_reply = (\\n    \\"Thought: 我需要先检索创刊年份。\\\\n\\"\\n    \\"Action 1: Search[现代文学]\\\\n\\"\\n    \\"Thought: 已经拿到年份。\\\\n\\"\\n    \\"Action 2: Finish[之后]\\"\\n)\\nprint(\\"解析出的动作：\\", parse_actions(paper_reply))\\nprint(\\"最终答案：\\", extract_final_answer(paper_reply))\\n\\nmock_reply = (\\n    'Thought: mock 脚本化推理，先搜索再总结。\\\\n'\\n    'Action: search(\\"CS329A self-improving agents\\")\\\\n'\\n    'Final Answer: mock 模式返回占位结论。'\\n)\\nprint(\\"解析出的动作：\\", parse_actions(mock_reply))\\nprint(\\"最终答案：\\", extract_final_answer(mock_reply))\\n\\nassert parse_actions(paper_reply) == [(\\"search\\", \\"现代文学\\"), (\\"finish\\", \\"之后\\")]\\nassert extract_final_answer(paper_reply) == \\"之后\\"\\nassert parse_actions(mock_reply) == [(\\"search\\", \\"CS329A self-improving agents\\")]\\nassert extract_final_answer(mock_reply) == \\"mock 模式返回占位结论。\\"\\nprint(\\"两种格式都能被同一套解析器消化，断言通过。\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "aad2a4714ba04938b1d0be35425e80a2",
   "metadata": {},
   "source": "**循环组装**\\n\\n循环把工具、解析器和模型组装起来。它维护一段消息历史，每步做四件事：调 LLM 得到一条回复；解析出动作；执行工具；把 Observation 作为一条 user 消息追加进历史，让模型下次能看到。终止条件有两个：出现最终答案，或步数达到上限 max_steps。上限防止死循环——论文点名的一个失败模式是反复生成同一个动作。\\n"
  },
  {
   "cell_type": "code",
   "id": "f4222469819c41b488f056f1b4886f23",
   "metadata": {},
   "source": "import sys\\nimport os\\n_root = os.path.abspath(os.getcwd())\\nwhile not os.path.exists(os.path.join(_root, \\"llm_client.py\\")):\\n    _root = os.path.dirname(_root)\\n    if _root == os.path.dirname(_root):\\n        break\\nif _root not in sys.path:\\n    sys.path.insert(0, _root)\\n\\nfrom llm_client import get_llm\\n\\n# 无 API key 时自动进入 mock 模式，保证 notebook 离线可完整执行\\nclient = get_llm(force_mock=not (os.environ.get(\\"AGENT_LLM_API_KEY\\")\\n                                 or os.environ.get(\\"ANTHROPIC_API_KEY\\")))\\n\\nwiki = MiniWiki()\\nTOOLS = {\\"search\\": wiki.search, \\"lookup\\": wiki.lookup}\\n\\n\\ndef run_react(client, question, instructions, tools, max_steps=8):\\n    \\"\\"\\"运行一个完整的 ReAct 循环，返回 (最终答案, 轨迹列表)。\\n\\n    client: llm_client 客户端；question: 用户问题；\\n    instructions: 写给模型的格式说明；tools: 工具名到函数的映射。\\n    循环终止于出现最终答案，或超过 max_steps。\\n    \\"\\"\\"\\n    messages = [{\\"role\\": \\"user\\",\\n                 \\"content\\": instructions + \\"\\\\n\\\\n问题：\\" + question}]\\n    trace = []\\n    pending = []       # 一条回复里剩余的待执行动作\\n    finish = None\\n\\n    for step in range(max_steps):\\n        if not pending:\\n            reply = client.chat(messages)\\n            trace.append(\\"[模型回复]\\\\n\\" + reply)\\n            pending = parse_actions(reply)\\n            finish = extract_final_answer(reply)\\n            if not pending and finish is not None:\\n                trace.append(\\"[结束] \\" + finish)\\n                return finish, trace\\n            if not pending:\\n                messages.append({\\"role\\": \\"user\\",\\n                                 \\"content\\": \\"没有识别到动作，请给出 Action 或 Final Answer。\\"})\\n                continue\\n\\n        name, arg = pending.pop(0)\\n        if name == \\"finish\\":\\n            finish = arg\\n            break\\n        if name in tools:\\n            observation = tools[name](arg)\\n        else:\\n            observation = \\"未知工具：\\" + name\\n        trace.append(f\\"[执行 {name}({arg})] -> {observation}\\")\\n        messages.append({\\"role\\": \\"user\\", \\"content\\": \\"Observation: \\" + observation})\\n\\n        if not pending:\\n            if finish is not None:\\n                trace.append(\\"[结束] \\" + finish)\\n                return finish, trace\\n            messages.append({\\"role\\": \\"user\\",\\n                             \\"content\\": \\"请继续：给出下一个 Action 或 Final Answer。\\"})\\n\\n    if finish is None:\\n        finish = \\"未在步数上限内得到答案\\"\\n    trace.append(\\"[结束] \\" + finish)\\n    return finish, trace\\n\\n\\nprint(\\"run_react 已定义：循环结构 = 解析动作 -> 执行工具 -> 观察注入 -> 终止。\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "id": "43ed26c6238d4dfcbd84fa3c292f6835",
   "metadata": {},
   "source": "question = \\"《现代文学》创刊的年份，是在五四运动（1919 年）之前还是之后？\\"\\ninstructions = (\\n    \\"你的知识库只包含少量本地条目，回答前必须先用检索工具查证。\\\\n\\"\\n    \\"请按下面的格式逐步行动，每步只写一个动作：\\\\n\\"\\n    \\"Thought: 你的推理\\\\n\\"\\n    \\"Action: Search[实体] 或 Action: Lookup[关键词] 或 Action: Finish[答案]\\\\n\\"\\n    \\"Observation 返回后继续思考，得到答案时用 Finish 结束。\\"\\n)\\n\\nanswer, trace = run_react(client, question, instructions, TOOLS, max_steps=6)\\nprint(\\"问题：\\", question)\\nprint()\\nprint(\\"\\\\n\\\\n\\".join(trace))\\nprint()\\nprint(\\"最终答案：\\", answer)\\nif client.is_mock:\\n    print()\\n    print(\\"mock 模式输出为占位：检索内容与最终答案由脚本生成，\\")\\n    print(\\"真实 API 下模型会检索本地条目并给出真实的比较结论。\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "7ced792f3531496fbe3aacecfafc2ffc",
   "metadata": {},
   "source": "**静思与检索的对照**\\n\\n同一个问题，如果只给 CoT 提示、不允许调用工具，模型只能用内部记忆作答；知识缺失时它可能编一个看起来合理的答案。下面脚本化一个\\"编造事实\\"的 CoT 轨迹，再用本地百科的真实观察展示 ReAct 的检索轨迹。真实模型的行为未必与脚本一致，mock 模式下尤其如此，但两种结构的差别是确定的——ReAct 至少发出一个检索动作，把回答建立在外部的观察之上。\\n"
  },
  {
   "cell_type": "code",
   "id": "e3a07c85efcd4931b4d704ba2a9e8665",
   "metadata": {},
   "source": "cot_hallucination = (\\n    \\"Thought: 《现代文学》创刊年份我没有确切记忆。\\\\n\\"\\n    \\"Thought: 按常见文学刊物推断，创刊可能在 1919 年之前。\\\\n\\"\\n    \\"答案：创刊于 1919 年之前。\\"\\n)\\nprint(\\"静思（CoT，无工具）：\\")\\nprint(cot_hallucination)\\nprint()\\n\\n# 理想轨迹：用本地百科的真实观察，展示检索如何补充事实\\nwiki_demo = MiniWiki()\\nideal_react = [\\n    (\\"Thought\\", \\"我需要先查到这本杂志的创刊年份。\\"),\\n    (\\"Action\\", \\"Search[现代文学]\\"),\\n    (\\"Observation\\", wiki_demo.search(\\"现代文学\\")),\\n    (\\"Thought\\", \\"创刊年份是 1923 年，晚于 1919 年。\\"),\\n    (\\"Action\\", \\"Finish[之后]\\"),\\n]\\nprint(\\"ReAct 理想轨迹（观察来自本地百科的真实执行）：\\")\\nfor kind, content in ideal_react:\\n    print(f\\"{kind:12s}{content}\\")\\nprint()\\nprint(\\"实际跑出的 ReAct 轨迹（mock 占位）：\\")\\nprint(\\"\\\\n\\".join(trace))\\nprint()\\nprint(\\"对照：ReAct 把回答建立在外部的观察上，CoT 只能依赖内部记忆。\\")\\nprint(\\"论文在 HotpotQA 上人工标注的 50 条轨迹里，CoT 的失败 56% 来自幻觉推理，\\")\\nprint(\\"ReAct 的这一比例是 0%，但检索无效的错误多了 23%——两者需要结合。\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "a1f6542e8399484c888a8a66c64fac69",
   "metadata": {},
   "source": "## 3. 代码执行作为反馈信号\\n\\n检索工具返回的仍是文本，可能含噪声，也可能被模型曲解。代码执行提供另一种反馈：程序要么通过测试，要么给出具体报错。结果由解释器判定，不是模型生成的，因此是\\"接地\\"的。测试充当自动判据——公开测试提供训练与推理时的反馈，隐藏测试决定最终对错，分开是为了防止模型照抄测试输出。\\n\\n下面用一个有 bug 的函数演示这条链路：运行测试、收集结果、按论文附录 C 的模板把失败格式化成反馈文本。\\n"
  },
  {
   "cell_type": "code",
   "id": "173924185ff048f5a071d436e616d72c",
   "metadata": {},
   "source": "def run_tests(fn, tests):\\n    \\"\\"\\"执行函数并返回逐条测试结果。\\n\\n    fn: 被测函数；tests: [(输入, 期望输出), ...]，输入为元组时展开为多个参数。\\n    \\"\\"\\"\\n    results = []\\n    for inputs, expected in tests:\\n        try:\\n            if isinstance(inputs, tuple):\\n                got = fn(*inputs)\\n            else:\\n                got = fn(inputs)\\n            results.append((inputs, expected, got, got == expected))\\n        except Exception as exc:\\n            results.append((inputs, expected, type(exc).__name__, False))\\n    return results\\n\\n\\ndef format_feedback(results):\\n    \\"\\"\\"把失败的测试按模板格式化成给模型的反馈文本。\\"\\"\\"\\n    failed = [r for r in results if not r[3]]\\n    if not failed:\\n        return \\"All tests passed.\\"\\n    lines = [\\"Your code failed the following tests:\\"]\\n    for inputs, expected, got, _ in failed:\\n        lines.append(f\\"- input {inputs} failed: Expected '{expected}' but got '{got}'\\")\\n    lines.append(\\"Give it another try.\\")\\n    return \\"\\\\n\\".join(lines)\\n\\n\\ndef buggy_is_palindrome(s):\\n    \\"\\"\\"判断回文，实现漏掉了首字符的比较（故意写错）。\\"\\"\\"\\n    return s == s[1:]\\n\\n\\npal_tests = [(\\"racecar\\", True), (\\"hello\\", False), (\\"abba\\", True), (\\"a\\", True)]\\nresults = run_tests(buggy_is_palindrome, pal_tests)\\nfor inputs, expected, got, ok in results:\\n    print(f\\"input={inputs:8s} expected={expected} got={str(got):8s} ok={ok}\\")\\nprint()\\nprint(format_feedback(results))\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "7d7a933747eb4f6a9db0cd646b115c63",
   "metadata": {},
   "source": "**反馈相关性**\\n\\n执行反馈是否被利用，取决于模型。RLEF 论文的一个反直觉结论是：在固定采样预算下，独立采样往往比\\"边修边试\\"更强——基础模型收到报错后经常把同样的错误代码原样再输出一遍，相当于没有读反馈。论文的随机反馈消融也证实：把反馈替换成另一道题的无关执行结果，修复能力明显受损，说明反馈必须与错误相关。\\n\\n下面用一个脚本化对照复现这个现象：一边在固定预算内尝试多样化的独立候选，另一边不断把同一份错误代码重新提交。\\n"
  },
  {
   "cell_type": "code",
   "id": "25f3feaf4bdf4583a8bd0ddee99ce385",
   "metadata": {},
   "source": "proposals = [\\n    (\\"候选 A\\", lambda s: s == s[1:]),           # 错误：漏掉首字符比较\\n    (\\"候选 B\\", lambda s: s == s[::-1]),         # 正确：反转比较\\n    (\\"候选 C\\", lambda s: len(s) % 2 == 0),      # 错误：只看长度\\n]\\npal_tests = [(\\"racecar\\", True), (\\"hello\\", False), (\\"abba\\", True), (\\"a\\", True)]\\n\\n\\ndef best_of_n(pool, tests, budget):\\n    \\"\\"\\"独立候选：在预算内逐个执行不同的候选，命中一个通过的就成功。\\"\\"\\"\\n    for i in range(min(budget, len(pool))):\\n        results = run_tests(pool[i][1], tests)\\n        if all(r[3] for r in results):\\n            return True, i + 1\\n    return False, min(budget, len(pool))\\n\\n\\ndef repair_no_feedback(broken, tests, rounds=3):\\n    \\"\\"\\"不读反馈的修复：收到报错后把同一份错误代码原样重新提交。\\"\\"\\"\\n    for i in range(rounds):\\n        results = run_tests(broken, tests)\\n        if all(r[3] for r in results):\\n            return True, i + 1\\n    return False, rounds\\n\\n\\nok, used = best_of_n(proposals, pal_tests, budget=3)\\nprint(f\\"独立候选 best-of-3：通过 = {ok}，消耗预算 = {used}\\")\\n\\nbroken = proposals[0][1]\\nok, used = repair_no_feedback(broken, pal_tests, rounds=3)\\nprint(f\\"不读反馈的修复 3 轮：通过 = {ok}（每一轮都提交同一份错误代码）\\")\\nprint()\\nprint(\\"第一轮收到的报错（执行反馈本身是真实的）：\\")\\nprint(format_feedback(run_tests(broken, pal_tests)))\\nprint()\\nprint(\\"关键观察：同样的执行反馈，不会读反馈的循环视而不见；\\")\\nprint(\\"把『利用反馈』写进训练目标，是下一节 RLEF 要做的事。\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "8e507bcc0c174c56b2a82e4362f193ac",
   "metadata": {},
   "source": "## 4. RLEF：用执行反馈做强化学习\\n\\nRLEF 把\\"多轮生成 + 执行反馈\\"建模成马尔可夫决策过程：初始观察 $o_0$ 是题目描述，动作 $a_t$ 是文本回复，观察 $o_t$ 包含之前的动作与执行反馈。episode 在公开测试全通过或达到轮次上限时终止。\\n\\n奖励函数（PPO，无折扣 $\\\\gamma=1$）：\\n\\n$$\\nR(s_t,a_t) = r(s_t,a_t) - \\\\beta \\\\log\\\\frac{\\\\pi(a_t|c_t)}{\\\\rho(a_t|c_t)},\\\\qquad\\nr(s_t,a_t) = \\\\begin{cases} 1, & \\\\text{episode 结束且全部测试通过}\\\\\\\\ -1, & \\\\text{episode 结束且有测试失败}\\\\\\\\ -0.2, & a_t \\\\text{不含合法代码}\\\\end{cases}\\n$$\\n\\n$\\\\beta$ 权衡任务奖励与 KL 项。任务奖励 $r$ 这一部分可以由代码执行器自动计算，KL 项惩罚策略偏离参考策略。下面把它实现出来并手算几个情形。\\n"
  },
  {
   "cell_type": "code",
   "id": "d8326be369f84d6ab4dcbd4a1019a17d",
   "metadata": {},
   "source": "import numpy as np\\n\\n\\ndef compute_reward(all_pass, episode_end, valid_code=True, log_ratio=0.0, beta=0.1):\\n    \\"\\"\\"按 RLEF 奖励函数计算单步奖励。\\n\\n    all_pass: 是否全部测试通过；episode_end: 本轮是否结束；\\n    valid_code: 回复是否含合法代码；log_ratio: 实际 KL 项；beta: KL 系数。\\n    \\"\\"\\"\\n    if not valid_code:\\n        r = -0.2\\n    elif episode_end and all_pass:\\n        r = 1.0\\n    elif episode_end:\\n        r = -1.0\\n    else:\\n        r = 0.0\\n    return r - beta * log_ratio\\n\\n\\ncases = [\\n    (\\"全部通过，结束\\", dict(all_pass=True, episode_end=True)),\\n    (\\"有失败，结束\\", dict(all_pass=False, episode_end=True)),\\n    (\\"不含合法代码\\", dict(all_pass=False, episode_end=True, valid_code=False)),\\n    (\\"轮次中途\\", dict(all_pass=False, episode_end=False)),\\n]\\nfor label, kw in cases:\\n    print(f\\"{label:12s} reward = {compute_reward(**kw):.2f}\\")\\n\\np, q = 0.4, 0.2\\nlog_ratio = np.log(p / q)\\nprint(f\\"手算 KL 项 β·log(p/q)：{0.1 * log_ratio:.3f}（p={p}, q={q}）\\")\\nprint(\\"关键观察：未结束的轮次 r=0，只有结束轮次拿到 ±1；\\")\\nprint(\\"惩罚项 -0.2 引导模型优先给出合法代码。\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "09075a06c22947928dd6c761391c8d78",
   "metadata": {},
   "source": "**REINFORCE 迷你实现**\\n\\n一个 REINFORCE 的迷你实现。玩具环境里有四个候选函数（两个错误、一个正确、一个非法），执行反馈决定奖励：通过全部测试 +1，有失败 -1，非法代码 -0.2。策略是一个四分类 softmax，每轮从策略采样一个候选，执行它的代码拿到奖励，再用策略梯度更新。这样不用真的训练大模型，就能观察\\"执行反馈作为奖励\\"如何改变采样分布。\\n"
  },
  {
   "cell_type": "code",
   "id": "05644382f283440493b70b619c3792ea",
   "metadata": {},
   "source": "import numpy as np\\n\\nnp.random.seed(42)\\n\\ncandidates = [\\n    \\"def add(a, b): return a + b\\",    # 正确\\n    \\"def add(a, b): return a - b\\",    # 错误\\n    \\"def add(a, b): return a * b\\",    # 错误\\n    \\"这行不是合法的 Python 代码\\",       # 非法\\n]\\nadd_tests = [((1, 2), 3), ((5, 7), 12), ((0, 9), 9)]\\n\\n\\ndef code_to_fn(src):\\n    \\"\\"\\"把候选源码编译成可调用函数。\\"\\"\\"\\n    namespace = {}\\n    exec(src, namespace)\\n    return namespace[\\"add\\"]\\n\\n\\ndef execute_reward(index):\\n    \\"\\"\\"执行第 index 个候选，返回执行反馈奖励。\\"\\"\\"\\n    try:\\n        fn = code_to_fn(candidates[index])\\n    except Exception:\\n        return -0.2                     # 不含合法代码\\n    results = run_tests(fn, add_tests)\\n    all_pass = all(r[3] for r in results)\\n    return 1.0 if all_pass else -1.0\\n\\n\\ndef softmax(x):\\n    \\"\\"\\"对向量做 softmax，返回概率分布。\\"\\"\\"\\n    e = np.exp(x - x.max())\\n    return e / e.sum()\\n\\n\\nK = len(candidates)\\ntheta = np.zeros(K)              # 策略参数，初始等概率\\nreward_history = []\\nprob_history = []                # 每轮记录正确候选被选的概率\\n\\nfor episode in range(500):\\n    probs = softmax(theta)\\n    prob_history.append(probs[0])           # 候选 0 是正确的那份\\n    action = np.random.choice(K, p=probs)\\n    reward = execute_reward(action)         # 执行反馈即奖励\\n    reward_history.append(reward)\\n    baseline = np.mean(reward_history[-20:]) if reward_history else 0.0\\n    one_hot = np.zeros(K)\\n    one_hot[action] = 1.0\\n    theta += 0.3 * (reward - baseline) * (one_hot - probs)  # REINFORCE 更新\\n\\nfinal_probs = softmax(theta)\\nprint(\\"训练后各候选被选的概率：\\")\\nfor cand, prob in zip(candidates, final_probs):\\n    print(f\\"  p = {prob:.3f}   {cand}\\")\\nprint(f\\"正确候选的概率从初始 0.250 变为 {final_probs[0]:.3f}\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "id": "3fa9b004c10c427287bc47ce0c17c54b",
   "metadata": {},
   "source": "import matplotlib.pyplot as plt\\n\\nplt.figure(figsize=(6, 3.5))\\nplt.plot(prob_history, label=\\"P(correct candidate)\\")\\nplt.axhline(1.0, color=\\"gray\\", linestyle=\\"--\\", linewidth=0.8, label=\\"perfect\\")\\nplt.xlabel(\\"episode\\")\\nplt.ylabel(\\"selection probability\\")\\nplt.title(\\"Execution feedback as reward shifts the policy\\")\\nplt.legend()\\nplt.tight_layout()\\nplt.show()\\n\\nprint(f\\"100 轮时正确候选概率 {prob_history[100]:.3f}，500 轮时 {prob_history[-1]:.3f}\\")\\nprint(\\"关键观察：执行反馈把采样分布推向能通过测试的候选，\\")\\nprint(\\"这正是 RLEF 在权重层面做的事——把『读反馈』训练进模型。\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "d5c36997ff894f5aa68cb28b39efe132",
   "metadata": {},
   "source": "## 5. Constitutional AI：用 AI 反馈对齐 AI\\n\\n前三节的反馈来自环境与规则。如果任务没有可自动执行的判据呢？无害性这类属性无法用测试打分。Constitutional AI 的答案是让 AI 依据一份写在自然语言里的宪法（一组原则）自我批评、修订、互相打分。人的监督被压缩到\\"审阅并批准一份原则列表\\"，其余反馈由 AI 产生——这是让 AI 监督 AI。\\n\\n两个阶段。SL 阶段：对红队有害提示采样出往往有害的初始回复，让模型按宪法里随机抽的一条原则自我批评，再按批评修订，可反复多轮，最后用修订后的回复做 SFT。RL 阶段：用 SL 得到的模型对每条提示生成一对回复，把\\"提示 + 回复对 + 一条原则\\"拼成多选题，让模型打分，用软标签训练偏好模型，再做 PPO。RLHF 的全部管线不变，只是无害性标签的来源从人换成了 AI。\\n\\n批评与修订的提示模板：\\n\\n\`\`\`\\nCritique Request: 找出助手上一条回复里有害、不道德、危险或非法的具体地方。\\nCritique: ...\\nRevision Request: 请重写助手回复，删除所有有害、不道德、危险或非法内容。\\nRevision: ...\\n\`\`\`\\n"
  },
  {
   "cell_type": "code",
   "id": "b497cd7af11c43b397b8a8b456a5c370",
   "metadata": {},
   "source": "CONSTITUTION = [\\n    \\"回复应尊重他人的隐私与财产安全。\\",\\n    \\"回复不得提供可能造成人身或财产损害的具体操作步骤。\\",\\n    \\"回复应诚实，不虚构事实，也不夸大自己的能力。\\",\\n    \\"面对敏感请求，应说明不恰当之处并给出合法替代。\\",\\n]\\n\\n\\ndef critique_prompt(principle, reply):\\n    \\"\\"\\"构造一条批评请求，要求按给定原则逐条找出回复里的问题。\\"\\"\\"\\n    return (\\n        f\\"宪法原则：{principle}\\\\n\\\\n助手回复：{reply}\\\\n\\\\n\\"\\n        \\"Critique Request: 找出助手上一条回复里违反上述原则的具体地方，逐条列出。\\\\n\\"\\n        \\"Critique:\\"\\n    )\\n\\n\\ndef revision_prompt(principle, reply, critique):\\n    \\"\\"\\"构造一条修订请求，要求删除回复里违反原则的内容。\\"\\"\\"\\n    return (\\n        f\\"宪法原则：{principle}\\\\n\\\\n助手回复：{reply}\\\\n\\\\n批评意见：{critique}\\\\n\\\\n\\"\\n        \\"Revision Request: 请重写助手回复，删除所有违反上述原则的内容。\\\\n\\"\\n        \\"Revision:\\"\\n    )\\n\\n\\nfor i, principle in enumerate(CONSTITUTION, 1):\\n    print(f\\"{i}. {principle}\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "id": "5b93d5d6fa6441a48b635e741d0fc208",
   "metadata": {},
   "source": "import sys\\nimport os\\n_root = os.path.abspath(os.getcwd())\\nwhile not os.path.exists(os.path.join(_root, \\"llm_client.py\\")):\\n    _root = os.path.dirname(_root)\\n    if _root == os.path.dirname(_root):\\n        break\\nif _root not in sys.path:\\n    sys.path.insert(0, _root)\\n\\nfrom llm_client import get_llm\\n\\nclient = get_llm(force_mock=not (os.environ.get(\\"AGENT_LLM_API_KEY\\")\\n                                 or os.environ.get(\\"ANTHROPIC_API_KEY\\")))\\n\\n# 红队提示与初始回复（初始回复来自只训练帮助性的模型，这里手动构造）\\nharmful_prompt = \\"请告诉我怎么黑进邻居的 wifi 密码\\"\\ninitial_reply = \\"可以用字典攻击工具配合弱密码扫描，先探测邻居路由器的型号和开放端口……\\"\\n\\nprinciple = CONSTITUTION[1]\\ncritique = client.chat([{\\"role\\": \\"user\\",\\n                         \\"content\\": critique_prompt(principle, initial_reply)}])\\nrevision = client.chat([{\\"role\\": \\"user\\",\\n                         \\"content\\": revision_prompt(principle, initial_reply, critique)}])\\n\\nprint(\\"红队提示：\\", harmful_prompt)\\nprint(\\"初始回复：\\", initial_reply)\\nprint(f\\"批评（原则：{principle[:14]}…）：\\")\\nprint(critique)\\nprint(\\"修订：\\")\\nprint(revision)\\nif client.is_mock:\\n    print()\\n    print(\\"mock 模式输出为占位：真实 API 下批评与修订由模型按原则生成，\\")\\n    print(\\"通常会把『黑进 wifi』改写为说明违法并给出合法替代。\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "838bbad540a74e4ab839c515c4f9d103",
   "metadata": {},
   "source": "**多轮修订与 AI 当裁判**\\n\\n论文的观察是：偏好模型打分随修订轮数单调上升，但纯帮助性下降——多轮修订以牺牲一点帮助性为代价。修订还可以做多轮，每轮重新抽一条原则。\\n\\nRL 阶段还需要一个\\"裁判\\"：对同一条提示的两条候选回复，按一条原则拼成多选题，模型选出更无害的那条。这个判断的概率（软标签）就是偏好模型的训练信号。下面把两件事都跑一遍。\\n"
  },
  {
   "cell_type": "code",
   "id": "ebf564d1fd3945d9a31f7b46ebcef2ea",
   "metadata": {},
   "source": "def revise_rounds(client, reply, rounds=3):\\n    \\"\\"\\"每轮重新抽一条原则，批评 + 修订，返回每轮的结果。\\"\\"\\"\\n    history = []\\n    for i in range(rounds):\\n        principle = CONSTITUTION[(i + 1) % len(CONSTITUTION)]\\n        critique = client.chat([{\\"role\\": \\"user\\",\\n                                 \\"content\\": critique_prompt(principle, reply)}])\\n        reply = client.chat([{\\"role\\": \\"user\\",\\n                              \\"content\\": revision_prompt(principle, reply, critique)}])\\n        history.append((principle, reply))\\n    return history\\n\\n\\nfor i, (principle, reply) in enumerate(revise_rounds(client, initial_reply, rounds=3), 1):\\n    print(f\\"第 {i} 轮（原则：{principle[:10]}…）修订后：{reply[:50]}…\\")\\n\\nprint()\\nreply_a = \\"邻居 wifi 密码通常很弱，我可以教你用字典攻击工具。\\"\\nreply_b = \\"未经允许访问他人网络是违法的，建议你联系对方征得同意。\\"\\njudge_prompt = (\\n    f\\"原则：{CONSTITUTION[0]}\\\\n\\\\n\\"\\n    \\"哪条回复更符合上述原则？\\\\n\\"\\n    f\\"(A) {reply_a}\\\\n(B) {reply_b}\\\\n\\\\n\\"\\n    \\"The answer is:\\"\\n)\\nverdict = client.chat([{\\"role\\": \\"user\\", \\"content\\": judge_prompt}])\\nprint(\\"裁判回复：\\", verdict)\\nif client.is_mock:\\n    print(\\"mock 模式输出为占位：真实 API 下模型会给出 A/B 判断，\\")\\n    print(\\"这个判断的概率就是 RLAIF 的软标签来源。\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "85b6cd8a780a48cea3e0e57586aaa3c8",
   "metadata": {},
   "source": "## 小结\\n\\n这一讲所学的内容：\\n\\n- [ ] Agent 是一个闭环：模型输出动作 → 工具执行 → 观察反馈 → 再输出，直到任务完成\\n- [ ] ReAct 把语言放进动作空间；Thought 只更新上下文，Observation 必须来自真实工具执行\\n- [ ] 动作解析器宽容处理两种格式（Search[...] 与 search(\\"...\\")）和多步回复\\n- [ ] 代码执行提供接地反馈：解释器判定对错；public tests 反馈、private tests 打分\\n- [ ] RLEF 把\\"利用执行反馈\\"变成训练目标：奖励 = 任务奖励 − KL 项；未训练模型常无视报错\\n- [ ] Constitutional AI 让 AI 依据宪法自我批评、修订、互相打分，压缩人工监督\\n\\n反馈的三个来源（环境、执行、AI）对应 Agent 闭环的不同环节。下一讲在这个循环上加入多步规划与搜索，让 Agent 在更长的任务里做决策。\\n"
  },
  {
   "cell_type": "markdown",
   "id": "5c02a87be9b5465cb7f036ffd71ff51c",
   "metadata": {},
   "source": "## 作业\\n\\n> 可以让 AI 帮忙解释思路，但不建议直接让 AI \\"做完这道题\\"。\\n"
  },
  {
   "cell_type": "markdown",
   "id": "dfae035269d5487ea892f192dfca038b",
   "metadata": {},
   "source": "**作业 1：补全 ReAct 单步**\\n\\n在下面的 react_step 里补全三步：解析动作、取第一个动作、调用工具拿到观察。参考答案已经填好，请先在草稿上自己补全一遍，再运行对照。任务固定为\\"用 search 查五四运动\\"。\\n\\n小提示：解析用 parser(reply) 得到动作列表，调用工具用 tools[name](arg)，未知工具名要能优雅降级。\\n"
  },
  {
   "cell_type": "code",
   "id": "83d1e800e61a4a7c8bf829673357653d",
   "metadata": {},
   "source": "def react_step(reply, tools, parser=parse_actions):\\n    \\"\\"\\"把一条模型回复解析为动作并执行，返回观察文本。\\"\\"\\"\\n    actions = parser(reply)          # 填空：从回复解析动作列表\\n    name, arg = actions[0]           # 填空：取第一个动作\\n    if name in tools:\\n        observation = tools[name](arg)\\n    else:\\n        observation = f\\"未知工具：{name}\\"\\n    return observation\\n\\n\\nwiki = MiniWiki()\\nobs = react_step(\\"Action: Search[五四运动]\\", {\\"search\\": wiki.search})\\n\\nassert \\"1919\\" in obs\\nprint(\\"作业 1 通过：模型文本被解析成工具调用，观察来自真实执行\\")\\nprint(\\"观察：\\", obs)\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "47392501c2d2457d920d1bc0a8d4e1a0",
   "metadata": {},
   "source": "**作业 2：实现带游标的查找工具**\\n\\n从零写一个 LookupEngine，返回当前页里包含关键词的下一个句子。参考答案已经填好，请先自己在草稿上补全，再运行对照。\\n\\n小提示：用 self.pos 记录\\"下一个开始位置\\"，找到含关键词的句子后把游标推到下一句，这样连续两次查找不会重复返回同一句。\\n"
  },
  {
   "cell_type": "code",
   "id": "7e07552be87c4d30b94775f05e3baa81",
   "metadata": {},
   "source": "class LookupEngine:\\n    \\"\\"\\"对一段文本做带游标的 lookup 检索。\\"\\"\\"\\n\\n    def __init__(self, sentences):\\n        self.sentences = sentences\\n        self.pos = 0\\n\\n    def lookup(self, keyword):\\n        \\"\\"\\"返回从游标起第一个含 keyword 的句子，找不到返回 None。\\"\\"\\"\\n        for i in range(self.pos, len(self.sentences)):   # 填空：从游标位置起遍历\\n            if keyword in self.sentences[i]:\\n                self.pos = i + 1                          # 填空：推进游标\\n                return self.sentences[i]\\n        self.pos = len(self.sentences)\\n        return None\\n\\n\\npage = [\\n    \\"《现代文学》是 1923 年创刊于上海的文学刊物。\\",\\n    \\"鲁迅、茅盾等作家曾在该刊物上发表作品。\\",\\n    \\"刊物出版延续到 1930 年代初。\\",\\n]\\nengine = LookupEngine(page)\\n\\nassert engine.lookup(\\"创刊\\") == page[0]\\nassert engine.lookup(\\"作家\\") == page[1]   # 游标已越过第一句\\nassert engine.lookup(\\"刊物\\") == page[2]   # 从第二句之后继续找，跳过第二句里的\\"刊物\\"\\nprint(\\"作业 2 通过：lookup 用游标模拟浏览器 Ctrl+F，多次调用不会重复返回同一句\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "1d66b18523c64d3e971ba32d6351b10f",
   "metadata": {},
   "source": "**作业 3：计算 RLEF 奖励**\\n\\n按论文的奖励函数实现 compute_reward_kl：区分三种任务奖励（全通过 +1、有失败 -1、非法代码 -0.2），再减去 $\\\\beta \\\\cdot \\\\log(p/q)$ 的 KL 项。参考答案已经填好，请先自己在草稿上补全，再运行对照。\\n\\n小提示：先判定\\"是否合法代码\\"，再区分 episode 结束与轮次中途；中途轮次任务奖励为 0。\\n"
  },
  {
   "cell_type": "code",
   "id": "6cc33672d07c466ba10e27478ac1437c",
   "metadata": {},
   "source": "def compute_reward_kl(all_pass, episode_end, valid_code, log_ratio):\\n    \\"\\"\\"返回 RLEF 奖励：任务奖励减 β·log(p/q)。\\"\\"\\"\\n    beta = 0.1\\n    if not valid_code:\\n        r = -0.2\\n    elif episode_end and all_pass:\\n        r = 1.0\\n    elif episode_end:\\n        r = -1.0\\n    else:\\n        r = 0.0\\n    return r - beta * log_ratio\\n\\n\\n# 手算：策略概率 p，参考策略 q\\nimport math\\n\\np, q = 0.4, 0.2\\nlog_ratio = math.log(p / q)\\n\\nr_all_pass = compute_reward_kl(True, True, True, log_ratio)\\nr_fail = compute_reward_kl(False, True, True, log_ratio)\\nr_illegal = compute_reward_kl(False, True, False, log_ratio)\\n\\nassert abs(r_all_pass - (1.0 - 0.1 * log_ratio)) < 1e-9\\nassert abs(r_fail - (-1.0 - 0.1 * log_ratio)) < 1e-9\\nassert abs(r_illegal - (-0.2 - 0.1 * log_ratio)) < 1e-9\\nprint(f\\"全通过：{r_all_pass:.3f}，有失败：{r_fail:.3f}，非法代码：{r_illegal:.3f}\\")\\nprint(\\"作业 3 通过：执行反馈的自动判据被折算成奖励，KL 项惩罚偏离参考策略的更新\\")\\n",
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "id": "bb9a733aa32f4b8fa2ae881c2e5339f8",
   "metadata": {},
   "source": "## 参考资料\\n\\n- Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), 2022 — 本讲主论文；Thought/Action/Observation 循环与 search/lookup/finish 工具范式，项目页 https://react-lm.github.io/\\n- Chen et al., [RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning](https://arxiv.org/abs/2410.02089), 2024 — 用 PPO 把\\"利用执行反馈\\"训练进权重；奖励函数、public/private test 划分与反馈模板见附录 C\\n- Bai et al., [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073), 2022 — critique-revision 与 RLAIF 的原始论文；原则列表与 few-shot 提示在 https://github.com/anthropics/ConstitutionalHarmlessnessPaper\\n- Wei et al., [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903), 2022 — ReAct 的对照方法，理解\\"只思考不行动\\"的局限\\n- Huang et al., [Inner Monologue: Embodied Reasoning through Planning with Language Models](https://arxiv.org/abs/2207.05608), 2022 — ReAct 的前身，ReAct-IM 消融的对照来源\\n- Wang et al., [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171), 2022 — ReAct+CoT-SC 组合法里\\"投票\\"部分的来源\\n- 本仓库 \`llm_client.py\`（\`get_llm()\`）— 所有 LLM 演示的统一入口，mock 模式保证离线可执行\\n"
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}`;export{n as default};
