const n=`{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "# 自改进智能体的开放进化",
   "id": "7f63aa2b86517222"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "> 前六讲回答的是同一个问题：人怎么设计一个更好的 Agent。第 4 讲搭建了 ReAct 循环，第 5 讲设计了任务分解与树搜索，第 6 讲用强化学习训练 Agent 的参数。设计者始终是人，Agent 只是被设计出来的对象。\\n>\\n> 这一节把设计者本身交给搜索。我们让一个 Agent 去设计另一个 Agent：生成候选 Agent 的代码，在任务上评测打分，把好的结果存进 archive，再从中得到启发生成下一批。这样一套变异、选择、积累的循环就是开放进化。我们从 ADAS 的最小闭环出发，把它放大到 AI Scientist 的科研流程，再看到 AlphaEvolve 把整份代码当作基因组来进化，最后讨论这套循环的失败模式。",
   "id": "1dfa558a43412ec9"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "机器学习的历史有一条反复出现的规律：人工设计的组件，最终被学出来的组件取代。人工特征被学到的特征取代，人工设计的网络结构被 NAS 搜索取代。如果这条规律继续成立，Agent 系统的设计会成为下一个被自动化的对象。\\n\\nADAS、AI Scientist、AlphaEvolve 三篇论文对这个问题给出了相同的回答。三套系统的骨架相同，都可以拆成三个组件：变异算子负责产生新变体，评测或评审负责打分选择，archive 或种群负责存放变体、保留多样性。三篇论文只是三个旋钮的不同取值——变体是 Agent 代码还是研究 idea，适应度是任务准确率还是论文评审，种群是代码库还是知识库。\\n\\n这一节先用一个具体的算术任务把骨架走通。我们从 ADAS 开始：一个元 Agent 迭代地写 Agent 的代码、跑评测、把好结果存进 archive。",
   "id": "788574931dc75e94"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## 1. 让 Agent 自己设计 Agent：ADAS\\n\\nADAS（Automated Design of Agentic Systems）把设计 Agent 定义成一个优化问题。Agent 的全部组件——提示词、工具调用、控制流——都写在一个 forward 函数里。由于 Python 是图灵完备的语言，以代码为搜索空间理论上可以表达任何可能的 Agent 系统。\\n\\n三个组件逐一落到代码里。变异算子由大模型扮演：它读 archive 的摘要，写出一份新的 Agent 代码。选择机制是一个评测函数：在固定任务上运行这份代码，得到准确率。种群是一个 archive：字典里存着名字对应的代码与得分，供下一轮参考。先把评测函数和任务定下来。",
   "id": "70c72b462b84cb65"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "# 任务：两位数乘法。框架提供 solve(a, b, recipe)，按\\"推理配方\\"计算答案；\\n# 环境用确定性伪随机判断每个配方会在哪些题上失误，模拟底层模型\\n# 在不同推理深度下的差错率。np.random.seed 保证后续演示可复现。\\nimport numpy as np\\nnp.random.seed(42)\\n\\nA = np.repeat(np.arange(11, 16), 8)   # 5 个十位，每个重复 8 次\\nB = np.tile(np.arange(11, 19), 5)     # 8 个个位，平铺 5 份\\nPROBLEMS = list(zip(A.tolist(), B.tolist()))\\nTRUE = [a * b for a, b in PROBLEMS]\\n\\nRECIPE_RATE = {\\"direct\\": 0.40, \\"decompose\\": 0.12,\\n               \\"decompose_check\\": 0.05, \\"ensemble\\": 0.00}\\n\\ndef is_tricky(a, b, recipe):\\n    \\"\\"\\"该配方在这个题上是否失误：由种子决定的确定性判断。\\"\\"\\"\\n    r = np.random.RandomState(1000 * (a % 10) + (b % 10) + len(recipe))\\n    return r.rand() < RECIPE_RATE[recipe]\\n\\ndef solve(a, b, recipe):\\n    \\"\\"\\"基础求解函数：按配方计算 a*b，失误时十位与个位交换。\\"\\"\\"\\n    if is_tricky(a, b, recipe):\\n        return str((a % 10) * (b % 10) + 100 * (a // 10) * (b // 10))\\n    return str(a * b)\\n\\nfor rec in RECIPE_RATE:\\n    acc = sum(1 for (a, b), t in zip(PROBLEMS, TRUE)\\n              if solve(a, b, rec) == str(t)) / len(PROBLEMS)\\n    print(f\\"配方 {rec:<14} 设定失误率 {RECIPE_RATE[rec]:.2f} 实测准确率 {acc:.2f}\\")",
   "id": "66d9d90a691ddedf"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "### 变异算子：改一个词，评分就变\\n\\n三要素里的变异算子负责产生新变体。在代码空间里，变异是改一行代码；在参数空间里，变异是改一个参数值。先看参数层面的变异：把 Agent 的提示词换一个词，评分跟着变。真实系统里这个评分来自实际运行 Agent，这里先用一个人造的确定性评分函数代替，方便观察变异的效果——每个动作关键词带来固定的准确率增益。",
   "id": "04bcc51303642bef"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "# 变异算子最小演示：提示词里插入一个动作关键词，评分改变。\\n# 评分函数是确定性的：基础准确率 0.30，叠加命中关键词的增益。\\nKEYWORD_BONUS = {\\"分解\\": 0.15, \\"验算\\": 0.15, \\"估算\\": 0.10, \\"分步\\": 0.05}\\n\\ndef prompt_score(prompt):\\n    \\"\\"\\"提示词评分：基础准确率 0.30，叠加命中关键词的增益。\\"\\"\\"\\n    score = 0.30\\n    for word, bonus in KEYWORD_BONUS.items():\\n        if word in prompt:\\n            score += bonus\\n    return score\\n\\nCANDIDATES = [\\"分解\\", \\"验算\\", \\"估算\\", \\"分步\\"]\\n\\ndef mutate_prompt(prompt, seed):\\n    \\"\\"\\"从候选词里随机挑一个追加到提示词末尾，完成一次变异。\\"\\"\\"\\n    rng = np.random.RandomState(seed)\\n    word = CANDIDATES[int(rng.randint(len(CANDIDATES)))]\\n    return prompt + \\"，\\" + word\\n\\nbase = \\"请计算这道乘法题\\"\\nfor k in range(4):\\n    child = mutate_prompt(base, k)\\n    print(f\\"变异 {k + 1}: 「{child}」 评分 {prompt_score(child):.2f}\\")\\n\\nprint(\\"原始提示词评分:\\", round(prompt_score(base), 2))",
   "id": "1c71814b285b0824"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "### 参数空间上的简单搜索\\n\\n变异产生了新变体，选择由评分函数完成：分数高的变体被保留，分数低的被丢弃。把 Agent 的三个参数——推理步数、集成样本数、工具数——当成一个三维点 $p=(n_{cot}, n_{samples}, n_{tools})$，评分函数对每个点返回一个确定性的分数：\\n\\n$$score(p) = 0.20 + 0.30·tanh(n_{cot}/3) + 0.10·tanh(n_{samples}/4) + 0.05·tanh((n_{tools}-2)/1.5) - 0.03·max(0, n_{tools}-5)$$\\n\\n前三项让推理步数与集成样本带来边际递减的提升，最后一项惩罚工具过多。这个函数是人造的，只用来观察搜索机制；真实系统的评分来自实际运行。手算两个点。$p=(0,1,2)$：$tanh(0)=0$，$tanh(0.25)≈0.245$，得分 $≈0.20+0.0245≈0.22$。$p=(6,8,3)$：$tanh(2)≈0.964$，$tanh(2/3)≈0.583$，得分 $≈0.20+0.30×0.964+0.10×0.964+0.05×0.583≈0.61$。让代码验证这两点，再做爬山与随机搜索两种方法。",
   "id": "a54de1510174b2fd"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "def agent_score(p):\\n    \\"\\"\\"三个参数 (推理步数, 集成样本数, 工具数) 的确定性评分。\\"\\"\\"\\n    n_cot, n_samples, n_tools = p\\n    acc = 0.20 + 0.30 * np.tanh(n_cot / 3.0)\\n    acc += 0.10 * np.tanh(n_samples / 4.0)\\n    acc += 0.05 * np.tanh((n_tools - 2) / 1.5)\\n    acc -= 0.03 * max(0, n_tools - 5)\\n    return float(np.clip(acc, 0.0, 1.0))\\n\\nfor p in [(0, 1, 2), (6, 8, 3)]:\\n    print(f\\"agent_score{p} = {agent_score(p):.4f}\\")\\n\\ndef neighbors(p):\\n    \\"\\"\\"返回四个相邻参数点：推理步数与集成样本数各加减一。\\"\\"\\"\\n    n_cot, n_samples, n_tools = p\\n    return [(n_cot + 1, n_samples, n_tools),\\n            (n_cot, n_samples + 1, n_tools),\\n            (max(0, n_cot - 1), n_samples, n_tools),\\n            (n_cot, max(1, n_samples - 1), n_tools)]",
   "id": "db3ea4e82ab13b55"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "def hill_climb(start, steps=12):\\n    \\"\\"\\"从 start 出发，每次移动到评分更高的最好邻居，返回轨迹。\\"\\"\\"\\n    p = start\\n    trace = [(p, agent_score(p))]\\n    for _ in range(steps):\\n        best_c = max(neighbors(p), key=agent_score)\\n        if agent_score(best_c) <= agent_score(p):\\n            break\\n        p = best_c\\n        trace.append((p, agent_score(p)))\\n    return trace\\n\\ndef random_search(budget=30, seed=0):\\n    \\"\\"\\"独立采样 budget 个参数点，记录迄今最好评分。\\"\\"\\"\\n    rng = np.random.RandomState(seed)\\n    best = -1.0\\n    trace = []\\n    for _ in range(budget):\\n        p = (int(rng.randint(0, 10)), int(rng.randint(1, 12)),\\n             int(rng.randint(0, 8)))\\n        s = agent_score(p)\\n        if s > best:\\n            best, best_p = s, p\\n        trace.append(best)\\n    return best_p, trace\\n\\nhc_trace = hill_climb((0, 1, 2))\\nbest_rs, rs_trace = random_search(30, seed=2)\\n\\nprint(\\"爬山轨迹（参数, 评分）:\\")\\nfor p, s in hc_trace:\\n    print(f\\"  {p}  {s:.3f}\\")\\nprint(\\"爬山终点评分:\\", round(hc_trace[-1][1], 3))\\nprint(\\"随机搜索最优参数:\\", best_rs, \\"评分:\\", round(agent_score(best_rs), 3))",
   "id": "1b6cd8093d058d9f"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "%matplotlib inline\\nimport matplotlib.pyplot as plt\\n\\n_, axes = plt.subplots(1, 2, figsize=(9, 3.5))\\naxes[0].plot([s for _, s in hc_trace], \\"o-\\", color=\\"#2c7fb8\\")\\naxes[0].set_xlabel(\\"Iteration\\")\\naxes[0].set_ylabel(\\"Score\\")\\naxes[0].set_title(\\"Hill climbing\\")\\naxes[1].plot(rs_trace, \\".-\\", color=\\"#d95f0e\\")\\naxes[1].set_xlabel(\\"Sample\\")\\naxes[1].set_ylabel(\\"Best score so far\\")\\naxes[1].set_title(\\"Random search\\")\\nplt.tight_layout()\\nplt.show()\\nprint(\\"爬山从\\", round(hc_trace[0][1], 3), \\"爬到\\", round(hc_trace[-1][1], 3))",
   "id": "6d8ed2327fe06d66"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "### 最小闭环：元 Agent 写 Agent\\n\\n参数层面的搜索说明了变异与选择如何工作，但 ADAS 的搜索空间是代码，不是参数。最小闭环是四步：元 Agent 读 archive 摘要，写出一份 forward 函数代码；我们把它编译成可调用对象；在算术任务上评测得到准确率；把结果连同代码存进 archive。archive 从两份基线 Agent 起步，逐轮生长。先搭基础设施。",
   "id": "0f793e72e7aebbb5"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "import sys, os\\n_root = os.path.abspath(os.getcwd())\\nwhile not os.path.exists(os.path.join(_root, 'llm_client.py')):\\n    _root = os.path.dirname(_root)\\n    if _root == os.path.dirname(_root):\\n        break\\nif _root not in sys.path:\\n    sys.path.insert(0, _root)\\nfrom llm_client import get_llm\\n\\nclient = get_llm()\\nprint(\\"mock 模式:\\", client.is_mock)",
   "id": "d8c2b9f77abe3cea"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "def wrap_recipe(recipe):\\n    \\"\\"\\"把推理配方包成一份 forward 源码字符串。\\"\\"\\"\\n    return f\\"\\"\\"def forward(task):\\n    a, b = task\\n    return solve(a, b, recipe='{recipe}')\\"\\"\\"\\n\\ndef compile_agent(src):\\n    \\"\\"\\"把 forward 源码编译成可调用函数，框架函数对生成的代码可见。\\"\\"\\"\\n    ns = globals().copy()\\n    exec(src, ns)\\n    return ns[\\"forward\\"]\\n\\ndef eval_agent(fn, problems=PROBLEMS, true=TRUE):\\n    \\"\\"\\"运行 Agent 在任务上评测，返回准确率。\\"\\"\\"\\n    correct = sum(1 for (a, b), t in zip(problems, true)\\n                  if fn((a, b)) == str(t))\\n    return correct / len(problems)\\n\\n# archive：字典，存\\"名字 → (源码, 得分)\\"。从两份基线起步，先算出真实得分。\\narchive = {\\"baseline_direct\\": (wrap_recipe(\\"direct\\"), 0.0),\\n           \\"baseline_decompose\\": (wrap_recipe(\\"decompose\\"), 0.0)}\\nfor name, (src, _) in archive.items():\\n    archive[name] = (src, eval_agent(compile_agent(src)))\\nfor name, (_, acc) in archive.items():\\n    print(f\\"{name}: 准确率 {acc:.2f}\\")",
   "id": "b171c3caf4bcdbdc"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "def extract_recipe(src):\\n    \\"\\"\\"从源码读出配方名，解析失败返回 'unknown'。\\"\\"\\"\\n    for rec in RECIPE_RATE:\\n        if f\\"recipe='{rec}'\\" in src:\\n            return rec\\n    return \\"unknown\\"\\n\\ndef meta_prompt(archive):\\n    \\"\\"\\"构造元 Agent 的提示词：读 archive 摘要，写新的 forward 代码。\\"\\"\\"\\n    lines = []\\n    for name, (src, acc) in archive.items():\\n        lines.append(\\"- \\" + name + \\": 准确率 \\" + f\\"{acc:.2f}\\"\\n                     + \\", 配方 \\" + extract_recipe(src))\\n    summary = \\"\\\\n\\".join(lines)\\n    return (\\"你是元 Agent，负责设计能解两位数乘法的 Agent。框架提供 \\"\\n            \\"solve(a, b, recipe)，recipe 可取 direct / decompose / \\"\\n            \\"decompose_check / ensemble。已有 Agent：\\\\n\\" + summary +\\n            \\"\\\\n请写一个新的 forward(task) 函数，输出用 \`\`\`python 包裹。\\")\\n\\ndef generate_agent(client, archive, it):\\n    \\"\\"\\"元 Agent 生成一份 Agent 代码。mock 模式返回脚本化占位。\\"\\"\\"\\n    if client.is_mock:\\n        # mock 模式输出为占位：从脚本化配方池轮转，展示步进石行为\\n        pool = [\\"decompose_check\\", \\"ensemble\\", \\"decompose\\", \\"decompose_check\\"]\\n        return wrap_recipe(pool[it % len(pool)])\\n    reply = client.chat([{\\"role\\": \\"user\\", \\"content\\": meta_prompt(archive)}])\\n    code = reply.split(\\"\`\`\`python\\")[-1].split(\\"\`\`\`\\")[0].strip()\\n    if \\"def forward\\" not in code:\\n        code = wrap_recipe(\\"decompose_check\\")\\n    return code\\n\\nprint(meta_prompt(archive))\\nprint(\\"---- 元 Agent 输出（mock 模式为脚本化占位）----\\")\\nprint(generate_agent(client, archive, 0))",
   "id": "707517ef8d3b05f8"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "def adas_loop(client, archive, rounds=4):\\n    \\"\\"\\"元 Agent 搜索：生成 → 编译 → 评测 → 入库。返回每轮记录。\\"\\"\\"\\n    log = []\\n    for it in range(rounds):\\n        src = generate_agent(client, archive, it)\\n        try:\\n            acc = eval_agent(compile_agent(src))\\n        except Exception:\\n            acc = 0.0\\n        name = \\"discovered_\\" + str(it)\\n        best = max(s for _, s in archive.values())\\n        added = acc > best\\n        if added:\\n            archive[name] = (src, acc)\\n        log.append((it, extract_recipe(src), acc, added))\\n    return log\\n\\nlog = adas_loop(client, archive, rounds=4)\\nfor it, recipe, acc, added in log:\\n    flag = \\"入库\\" if added else \\"舍弃\\"\\n    print(f\\"第 {it} 轮: 配方 {recipe:<14} 准确率 {acc:.2f}  {flag}\\")\\n\\nprint(\\"最终 archive（步进石记录）:\\")\\nfor name, (src, acc) in archive.items():\\n    print(f\\"  {name:<18} 配方 {extract_recipe(src):<14} 准确率 {acc:.2f}\\")",
   "id": "67e34446b908b8f3"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "### 步进石：archive 是变异算子的记忆\\n\\narchive 的价值在于给变异算子当记忆。上面第 0 轮发现的 decompose_check 借用了基线里的 decompose，再补上验算；第 1 轮的 ensemble 又把 decompose_check 组合成投票。每次入库都建立在上一轮的基础上，这正是论文里观察到的步进石现象：ARC 挑战中第 3 轮出现\\"多路 CoT + 修正\\"，到第 25 轮才把多种反馈组装成最终 Agent。archive 从两个基线出发，收录两个改进后停止增长——没有新突破时，archive 不再增长。",
   "id": "9db884bfb84193e4"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## 2. 自动化的科学发现：AI Scientist\\n\\nADAS 把设计 Agent 放进了循环。AI Scientist 把同样的循环放大到完整的科研流程：提出研究 idea，写代码做实验，把结果写成论文，由自动评审器打分，再把通过的 idea 连同分数存进知识 archive。与 ADAS 相比有三个升级：变体从 Agent 代码变成研究 idea，适应度从任务准确率变成论文评审，种群从代码库变成知识库。\\n\\n自动评审器是闭环的关键，也是风险所在。论文里用按 NeurIPS 指南打分的 GPT-4o 评审 agent，在 500 篇 ICLR 2022 论文上达到接近人类水平的判断。当作者与评审都是同一个 AI 时，闭环里没有外部真值。我们用最小循环观察这个结构：mock 大模型提出 idea，确定性环境跑出实验指标，评审函数决定取舍。",
   "id": "f916f93918176b00"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "# 实验环境：固定回归数据上从零实现的全批量梯度下降。\\n# idea 通过修改超参改变实验指标（验证损失），全部确定可复现。\\nimport numpy as np\\nnp.random.seed(42)\\n\\nrng = np.random.RandomState(0)\\nX = rng.randn(80, 8)\\nw_true = np.array([1.0, 2.0, -1.5, 1.0, -0.5, 0.8, -0.3, 0.6])\\ny = X @ w_true + 0.6 * rng.randn(80)\\nXtr, Xva = X[:60], X[60:]\\nytr, yva = y[:60], y[60:]\\n\\ndef fit_linear(Xtr, ytr, Xva, yva, lr=0.05, wd=0.0, epochs=5, momentum=False):\\n    \\"\\"\\"全批量梯度下降拟合线性模型，返回验证集均方误差。\\"\\"\\"\\n    w = np.zeros(8)\\n    v = np.zeros(8)\\n    n = len(ytr)\\n    for _ in range(epochs):\\n        g = 2.0 * (Xtr.T @ (Xtr @ w - ytr)) / n + 2.0 * wd * w\\n        v = 0.9 * v + g\\n        w = w - lr * (v if momentum else g)\\n    pred = Xva @ w\\n    return float(np.mean((pred - yva) ** 2))\\n\\n# 基线只训练 5 轮，明显欠训练，给 idea 留下提升空间\\nBASE_CFG = {\\"lr\\": 0.05, \\"wd\\": 0.0, \\"epochs\\": 5, \\"momentum\\": False}\\nbaseline_val = fit_linear(Xtr, ytr, Xva, yva, **BASE_CFG)\\nprint(\\"基线配置验证损失:\\", round(baseline_val, 4))\\n\\nIDEAS = [(\\"降低学习率到 0.005\\", {\\"lr\\": 0.005}),\\n         (\\"加入 L2 权重衰减\\", {\\"wd\\": 0.05}),\\n         (\\"采用动量优化\\", {\\"momentum\\": True}),\\n         (\\"训练轮数翻倍\\", {\\"epochs\\": 10}),\\n         (\\"训练轮数四倍\\", {\\"epochs\\": 20})]\\nfor desc, cfg in IDEAS:\\n    val = fit_linear(Xtr, ytr, Xva, yva, **dict(BASE_CFG, **cfg))\\n    print(f\\"{desc}: 验证损失 {val:.4f}\\")",
   "id": "4f3b3febd3b43672"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "# idea 生成与评审都走大模型：mock 模式返回脚本化占位。\\ndef generate_idea(client, it):\\n    \\"\\"\\"提出一个研究 idea，返回 (描述, 超参改动)。\\"\\"\\"\\n    if client.is_mock:\\n        # mock 模式输出为占位：从候选池轮转\\n        return IDEAS[it % len(IDEAS)]\\n    prompt = (\\"你是研究助手。基线用 5 轮全批量梯度下降拟合线性回归，\\"\\n              \\"验证损失 \\" + f\\"{baseline_val:.4f}\\" + \\"。提出一个改进\\"\\n              \\"超参配置的 idea，按两行输出：\\\\nidea: <一句话>\\\\ncfg: <python dict>\\")\\n    reply = client.chat([{\\"role\\": \\"user\\", \\"content\\": prompt}])\\n    return parse_idea(reply) or IDEAS[it % len(IDEAS)]\\n\\ndef parse_idea(reply):\\n    \\"\\"\\"从回复里解析 (描述, 超参 dict)，解析失败返回 None。\\"\\"\\"\\n    desc, cfg = None, None\\n    for ln in reply.splitlines():\\n        if ln.startswith(\\"idea:\\") or ln.startswith(\\"idea：\\"):\\n            desc = ln.split(\\":\\", 1)[-1].strip()\\n        if ln.startswith(\\"cfg:\\") or ln.startswith(\\"cfg：\\"):\\n            cfg = ln.split(\\":\\", 1)[-1].strip()\\n    if desc is None or cfg is None:\\n        return None\\n    try:\\n        cfg = eval(cfg)\\n    except Exception:\\n        return None\\n    return (desc, cfg) if isinstance(cfg, dict) else None\\n\\ndef parse_scores(reply):\\n    \\"\\"\\"从评审回复里解析分数 dict，解析失败返回 None。\\"\\"\\"\\n    out = {}\\n    for token in reply.replace(\\",\\", \\" \\").split():\\n        if \\":\\" in token:\\n            k, v = token.split(\\":\\", 1)\\n        elif \\"=\\" in token:\\n            k, v = token.split(\\"=\\", 1)\\n        else:\\n            continue\\n        if k in (\\"soundness\\", \\"presentation\\", \\"contribution\\", \\"overall\\"):\\n            try:\\n                out[k] = float(v)\\n            except ValueError:\\n                pass\\n    return out if len(out) == 4 else None\\n\\ndef review_idea(client, desc, val, seen, baseline):\\n    \\"\\"\\"评审一个 idea，返回 (分数 dict, 是否接受)。\\n\\n    mock 模式按规则打分：损失改善越多分数越高，重复 idea 直接拒绝。\\n    \\"\\"\\"\\n    improve = baseline - val\\n    if not client.is_mock:\\n        prompt = (\\"评审一个研究 idea。描述：\\" + desc + \\"\\\\n验证损失：\\"\\n                  + f\\"{val:.4f}，基线：\\" + f\\"{baseline:.4f}。\\\\n\\"\\n                  \\"请按 NeurIPS 指南给出分数：soundness, presentation, \\"\\n                  \\"contribution, overall（1-10）。\\")\\n        reply = client.chat([{\\"role\\": \\"user\\", \\"content\\": prompt}])\\n        parsed = parse_scores(reply)\\n        if parsed:\\n            parsed[\\"accept\\"] = (desc not in seen and parsed[\\"overall\\"] >= 6\\n                                and parsed[\\"contribution\\"] >= 3)\\n            return parsed\\n    soundness = float(np.clip(3 + 50 * improve, 1, 10))\\n    contribution = float(np.clip(1 + 40 * improve, 1, 10))\\n    novelty = 0.3 if desc in seen else 0.9\\n    overall = float(np.clip(0.5 * soundness + 0.3 * contribution\\n                            + 0.2 * novelty, 1, 10))\\n    accept = desc not in seen and overall >= 6 and contribution >= 3\\n    return {\\"soundness\\": round(soundness, 1), \\"presentation\\": 5.0,\\n            \\"contribution\\": round(contribution, 1), \\"overall\\": round(overall, 1),\\n            \\"accept\\": accept}\\n\\ndemo = review_idea(client, \\"采用动量优化\\", baseline_val / 3, set(), baseline_val)\\nprint(\\"评审示例:\\", demo)",
   "id": "3329b8f9bca79015"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "def scientist_loop(client, rounds=7):\\n    \\"\\"\\"idea → 实验 → 评审 → 入库，返回每轮记录与知识 archive。\\"\\"\\"\\n    archive = {}\\n    seen = set()\\n    log = []\\n    if client.is_mock:\\n        print(\\"idea 由 mock 大模型脚本化生成，评审按规则打分（占位输出）\\")\\n    for it in range(rounds):\\n        desc, cfg = generate_idea(client, it)\\n        val = fit_linear(Xtr, ytr, Xva, yva, **dict(BASE_CFG, **cfg))\\n        review = review_idea(client, desc, val, seen, baseline_val)\\n        dup = desc in seen\\n        if review[\\"accept\\"]:\\n            archive[desc] = {\\"cfg\\": cfg, \\"val\\": val, \\"review\\": review}\\n        seen.add(desc)\\n        log.append((it, desc, val, review[\\"overall\\"], review[\\"accept\\"], dup))\\n    return log, archive\\n\\nlog, archive = scientist_loop(client, rounds=7)\\nfor it, desc, val, overall, acc, dup in log:\\n    if dup:\\n        reason = \\"舍弃（重复 idea）\\"\\n    else:\\n        reason = \\"入库\\" if acc else \\"舍弃（分数不足）\\"\\n    print(f\\"第 {it} 轮  {desc:<10} val={val:.4f} overall={overall}  {reason}\\")\\nprint(\\"知识 archive（通过的 idea）:\\")\\nfor desc, info in archive.items():\\n    print(f\\"  {desc}: val={info['val']:.4f} overall={info['review']['overall']}\\")",
   "id": "4e5d36d6c6c01917"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## 3. 代码作为 Agent 基因组：AlphaEvolve\\n\\nAI Scientist 的变体是研究 idea，AlphaEvolve 又把变体换回代码——不是一份 Agent 代码，而是整份算法源码。代码被当作基因组，大模型是变异算子：它读当前程序，输出一段代码改动（diff）；自动评测器计算适应度；进化数据库用质量多样性的思路维持一批不同解的种群。三个组件与 ADAS 相同，区别在规模：可以进化整份文件、任意语言、小时级的并行评测。\\n\\n论文最惊人的结果是矩阵乘法：用 48 次标量乘法算出两个 4×4 复矩阵的积，这是自 1969 年 Strassen 的 49 次以来 56 年的首个改进，整个发现只用了约 15 次变异。进化还有个硬前提：适应度必须能被自动评测。AlphaEvolve 的每个解都要在评测器里跑若干小时，无法自动算的问题就进不了循环——这既是它的边界，也是它从 FunSearch 的\\"单函数\\"扩到\\"整文件\\"的原因。先看变异算子的载体，一段 diff 长什么样。",
   "id": "da1f3b7b44664bc5"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "# 变异算子：解析并应用 AlphaEvolve 的 SEARCH/REPLACE diff 块。\\ndef apply_diff(old_code, search, replace):\\n    \\"\\"\\"在 old_code 中把 search 文本替换成 replace，返回新代码。\\"\\"\\"\\n    if search not in old_code:\\n        raise ValueError(\\"SEARCH 块未在代码中找到\\")\\n    return old_code.replace(search, replace)\\n\\nSRC = \\"\\"\\"def bubble(nums):\\n    for i in range(len(nums) - 1):\\n        for j in range(len(nums) - 1 - i):\\n            if nums[j] > nums[j + 1]:\\n                nums[j], nums[j + 1] = nums[j + 1], nums[j]\\n    return nums\\n\\"\\"\\"\\n\\ndef eval_sort(src):\\n    \\"\\"\\"运行排序代码，返回在 3 组数组上的正确比例。\\"\\"\\"\\n    ns = {}\\n    exec(src, ns)\\n    fn = ns[\\"bubble\\"]\\n    cases = [[3, 1, 2], [5, 4, 3, 2, 1], [1, 2, 3]]\\n    ok = 0\\n    for c in cases:\\n        ok += int(fn(list(c)) == sorted(c))\\n    return ok / len(cases)\\n\\nprint(\\"原始代码正确率:\\", eval_sort(SRC))\\n\\n# 变异 1：把升序比较改成降序比较，排序方向反转\\nsearch = \\"if nums[j] > nums[j + 1]:\\"\\nreplace = \\"if nums[j] < nums[j + 1]:\\"\\nmut1 = apply_diff(SRC, search, replace)\\nprint(\\"变异 diff：\\")\\nprint(f\\"<<<<<<< SEARCH\\\\n{search}\\\\n=======\\\\n{replace}\\\\n>>>>>>> REPLACE\\")\\nprint(\\"变异后正确率:\\", eval_sort(mut1))\\n\\n# 变异 2：外层循环多扫一轮，行为不变\\nmut2 = apply_diff(SRC, \\"range(len(nums) - 1)\\", \\"range(len(nums))\\")\\nprint(\\"去掉外层 -1 后正确率:\\", eval_sort(mut2))",
   "id": "38d48d69c79b3582"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "### 种群与选择：遗传算法\\n\\ndiff 展示了单次变异。进化还需要种群与选择：一批个体同时进化，每代里评分高的个体留下并复制，评分低的被淘汰，这就是遗传算法。下面在可解析验证的小目标上跑进化——用一组傅里叶基系数去逼近目标函数 sin x + 0.5 cos 2x。每个个体是一段 6 维系数向量，变异是在系数上加一点随机扰动，选择是保留评分最高的两个个体作为精英并复制。评分用负均方误差，越大越好。",
   "id": "93ed4b467de9a81f"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "# 目标：用傅里叶基系数逼近 sin x + 0.5 cos 2x。\\n# 个体是 6 个系数，评分是负均方误差，越大越好。\\nx = np.linspace(-np.pi, np.pi, 200)\\ny_target = np.sin(x) + 0.5 * np.cos(2 * x)\\n\\ndef basis_at(x):\\n    \\"\\"\\"傅里叶基：1, x, sin x, cos x, sin 2x, cos 2x。\\"\\"\\"\\n    return np.stack([np.ones_like(x), x, np.sin(x), np.cos(x),\\n                     np.sin(2 * x), np.cos(2 * x)])\\n\\nB = basis_at(x)\\n\\ndef fitness(coefs):\\n    \\"\\"\\"系数向量在网格上的逼近质量：负均方误差。\\"\\"\\"\\n    pred = B.T @ coefs\\n    return -float(np.mean((pred - y_target) ** 2))\\n\\ncoef_star, *_ = np.linalg.lstsq(B.T, y_target, rcond=None)\\nprint(\\"最小二乘最优评分的对照值:\\", round(fitness(coef_star), 4))\\n\\ndef evolve(pop, gen, elite=2, sigma=0.3, seed=0):\\n    \\"\\"\\"进化 gen 代：每代选最优 elite 个，用变异复制填满种群。\\"\\"\\"\\n    rng = np.random.RandomState(seed)\\n    curve = []\\n    for _ in range(gen):\\n        scores = np.array([fitness(p) for p in pop])\\n        curve.append(float(scores.max()))\\n        order = np.argsort(scores)[::-1][:elite]\\n        parents = [pop[i] for i in order]\\n        children = []\\n        while len(children) < len(pop) - elite:\\n            parent = parents[rng.randint(elite)]\\n            children.append(parent + sigma * rng.randn(len(parent)))\\n        pop = parents + children\\n    return curve, pop\\n\\nrng0 = np.random.RandomState(1)\\npop = [rng0.randn(6) * 2 for _ in range(30)]\\ncurve, last_pop = evolve(pop, gen=60, elite=2, sigma=0.3, seed=2)\\nprint(\\"第 0 代最优评分:\\", round(curve[0], 4))\\nprint(\\"第 60 代最优评分:\\", round(curve[-1], 4))",
   "id": "19877d034fdc3cb0"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "%matplotlib inline\\nimport matplotlib.pyplot as plt\\n\\nplt.figure(figsize=(7, 3.5))\\nplt.plot(curve, color=\\"#2c7fb8\\")\\nplt.xlabel(\\"Generation\\")\\nplt.ylabel(\\"Best fitness\\")\\nplt.title(\\"Best fitness over generations\\")\\nplt.grid(alpha=0.3)\\nplt.tight_layout()\\nplt.show()\\n\\n# 质量多样性视图：把最后种群按两个行为特征分箱，每箱保留最高评分\\nb0 = np.array([p[0] for p in last_pop])\\nb1 = np.array([p[1] for p in last_pop])\\nfs = np.array([fitness(p) for p in last_pop])\\ngrid = np.full((20, 20), np.nan)\\nix = np.clip(((b0 + 2.5) / 5.0 * 19).astype(int), 0, 19)\\niy = np.clip(((b1 + 2.5) / 5.0 * 19).astype(int), 0, 19)\\nfor a, b, f in zip(ix, iy, fs):\\n    grid[a, b] = f if np.isnan(grid[a, b]) else max(grid[a, b], f)\\n\\nplt.figure(figsize=(5.5, 4.5))\\nim = plt.imshow(grid, origin=\\"lower\\", cmap=\\"viridis\\")\\nplt.colorbar(im, label=\\"Fitness\\")\\nplt.xlabel(\\"Behavior: coef[0]\\")\\nplt.ylabel(\\"Behavior: coef[1]\\")\\nplt.title(\\"Performance map of final population\\")\\nplt.tight_layout()\\nplt.show()\\nprint(\\"被占用的行为格数:\\", int(np.sum(~np.isnan(grid))), \\"/ 400\\")",
   "id": "404260e150a15419"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## 4. 开放进化的失败模式与风险\\n\\n循环本身能工作，但风险埋在评测函数里。变异算子只优化它拿到的分数，一旦分数与真实目标脱节，进化就会找到取巧的路径。最典型的是奖励黑客：Agent 发现某个捷径让分数虚高，而不是真正学会任务。我们用演示把评测漏洞摊开：Agent 直接记住评测集的答案，就能拿到满分。再把评测集与开发集分开，看这道护栏如何挡住取巧路径。",
   "id": "5aa7147338feefe9"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "# 奖励黑客演示：开发与评测共用同一份题目时，硬编码答案是捷径。\\nimport numpy as np\\n\\nA = np.repeat(np.arange(11, 16), 8)\\nB = np.tile(np.arange(11, 19), 5)\\nTRAIN = list(zip(A.tolist(), B.tolist()))\\nTRUE_TRAIN = [a * b for a, b in TRAIN]\\n\\nH = np.repeat(np.arange(20, 25), 8)\\nK = np.tile(np.arange(21, 29), 5)\\nHOLD = list(zip(H.tolist(), K.tolist()))\\nTRUE_HOLD = [a * b for a, b in HOLD]\\n\\ndef honest_agent(task):\\n    \\"\\"\\"诚实 Agent：用框架的分步配方求解。\\"\\"\\"\\n    a, b = task\\n    return solve(a, b, \\"decompose_check\\")\\n\\nanswer_table = {(a, b): str(a * b) for (a, b) in TRAIN}\\n\\ndef cheat_agent(task):\\n    \\"\\"\\"取巧 Agent：把开发题答案硬编码，新题一律答 0。\\"\\"\\"\\n    return answer_table.get(task, \\"0\\")\\n\\nprint(\\"诚实 Agent  开发题准确率:  \\", round(eval_agent(honest_agent, TRAIN, TRUE_TRAIN), 2))\\nprint(\\"诚实 Agent  独立新题准确率:\\", round(eval_agent(honest_agent, HOLD, TRUE_HOLD), 2))\\nprint(\\"取巧 Agent  开发题准确率:  \\", round(eval_agent(cheat_agent, TRAIN, TRUE_TRAIN), 2))\\nprint(\\"取巧 Agent  独立新题准确率:\\", round(eval_agent(cheat_agent, HOLD, TRUE_HOLD), 2))",
   "id": "99937241454f4a8f"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "# 修复：开发集与评测集分离，评测集从不参与选择。\\n# 用开发集挑配方，再用独立评测集报告真实表现。\\ndef run_search_with_holdout(dev, hold, rounds=4):\\n    \\"\\"\\"开发集上搜索，独立评测集上报。返回 (开发最优配方, 开发准确率, 独立准确率)。\\"\\"\\"\\n    dev_problems, dev_true = dev\\n    hold_problems, hold_true = hold\\n    pool = [\\"decompose_check\\", \\"ensemble\\", \\"direct\\", \\"decompose_check\\"]\\n    best_recipe, best_acc = None, -1.0\\n    for it in range(rounds):\\n        recipe = pool[it % len(pool)]\\n        acc = eval_agent(compile_agent(wrap_recipe(recipe)),\\n                         dev_problems, dev_true)\\n        if acc > best_acc:\\n            best_recipe, best_acc = recipe, acc\\n    hold_acc = eval_agent(compile_agent(wrap_recipe(best_recipe)),\\n                          hold_problems, hold_true)\\n    return best_recipe, best_acc, hold_acc\\n\\nrecipe, dev_acc, hold_acc = run_search_with_holdout((TRAIN, TRUE_TRAIN),\\n                                                     (HOLD, TRUE_HOLD))\\nprint(\\"开发集上选出的配方:\\", recipe, \\" 开发集准确率:\\", round(dev_acc, 2))\\nprint(\\"独立评测集准确率:\\", round(hold_acc, 2))\\nprint(\\"结论：硬编码答案在独立评测上得 0 分，评测独立性挡住取巧路径。\\")",
   "id": "0daae4d8d63da82c"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "风险不止奖励黑客。AI Scientist 的评审者与作者同为 AI，闭环缺少外部真值——论文里就出现过把负结果写成改善、声称使用了错误硬件等幻觉；当 AI 作者批量投稿时，审稿系统也会被压垮。三篇论文各自给出工程护栏：ADAS 建议在容器里执行生成的代码，AI Scientist 建议沙箱化并标注 AI 产出，AlphaEvolve 承认评测必须可算本身就是前提。把三条放在一起，结论是：开放进化能让 Agent 越过人设计的天花板，也可能走向不受控的方向，评测的独立性是唯一的护栏。",
   "id": "6603bfd9354f498d"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## 小结\\n\\n这一节围绕\\"让 Agent 改进自己\\"展开。所学内容：\\n\\n- [ ] 开放进化的三要素：变异算子、选择机制、种群与 archive\\n- [ ] 变异的最小演示：改提示词的一个词，评分就变\\n- [ ] 参数空间搜索：爬山与随机搜索在确定性评分函数上的轨迹\\n- [ ] ADAS 最小闭环：元 Agent 写 forward 代码 → 编译 → 评测 → 入库\\n- [ ] 步进石：后发现的 Agent 组合前面 Agent 的组件，archive 给变异算子当记忆\\n- [ ] AI Scientist 把闭环放大到科研：idea → 实验 → 评审，适应度是论文评审\\n- [ ] AlphaEvolve 以整份代码为基因组，diff 是变异算子，评测器是适应度\\n- [ ] 失败模式：奖励黑客来自评测漏洞，评测独立性是唯一的护栏",
   "id": "0f38a76fa3d42bbc"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## 作业\\n\\n> 可以让 AI 帮忙解释思路，但不建议直接让 AI \\"做完这道题\\"。\\n\\n三道题各有一处空位，参考答案已填入代码，先在心里手算，再运行核对 assert。",
   "id": "bb5ff712d328cfeb"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "**作业 1：补全适应度与选择**\\n\\n给定 \`scores = {\\"a\\": 0.3, \\"b\\": 0.9, \\"c\\": 0.6, \\"d\\": 0.4}\`，补全 \`select_top_k(scores, k=2)\`，返回得分最高的 2 个 key，按得分从高到低；再补全 \`archive_update(archive, name, score)\`，只在 score 超过 archive 当前最低分时入库。\\n\\n小提示：用 \`sorted(scores.items(), key=lambda kv: kv[1], reverse=True)\` 排序后切片；入库前与 \`min(archive.values())\` 比较。",
   "id": "772b4577152a216f"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "def select_top_k(scores, k=2):\\n    \\"\\"\\"返回得分最高的 k 个 key，按得分从高到低。\\"\\"\\"\\n    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)\\n    return [name for name, _ in ordered[:k]]   # 空位在这里\\n\\ndef archive_update(archive, name, score):\\n    \\"\\"\\"score 超过 archive 当前最低分时入库，返回是否入库。\\"\\"\\"\\n    if not archive:\\n        archive[name] = score\\n        return True\\n    if score > min(archive.values()):   # 空位在这里\\n        archive[name] = score\\n        return True\\n    return False\\n\\nscores = {\\"a\\": 0.3, \\"b\\": 0.9, \\"c\\": 0.6, \\"d\\": 0.4}\\nassert select_top_k(scores, k=2) == [\\"b\\", \\"c\\"], \\"top-2 应是 b 与 c\\"\\n\\narch = {\\"x\\": 0.5}\\nassert archive_update(arch, \\"y\\", 0.9) is True\\nassert archive_update(arch, \\"z\\", 0.4) is False\\nassert \\"z\\" not in arch\\nprint(\\"收获：按得分选 top-k，只在超过当前最低分时入库。\\")",
   "id": "ffc4055ba3e6bf3d"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "**作业 2：补全 Agent 代码的编译与调用**\\n\\n元 Agent 返回的是一段字符串源码。补全 \`compile_agent(src)\`：用 \`exec\` 把源码编译进命名空间，取出 \`forward\` 并返回；再补全 \`run_agent(fn, task)\`，调用它返回答案。给定源码里先定义好 \`forward(task)\`。\\n\\n小提示：\`exec(src, ns)\` 之后 \`ns[\\"forward\\"]\` 就是可调用对象。",
   "id": "3dbe0c9d56762ecd"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "def compile_agent(src):\\n    \\"\\"\\"把 forward 源码字符串编译成可调用函数。\\"\\"\\"\\n    ns = {}\\n    exec(src, ns)\\n    return ns[\\"forward\\"]   # 空位在这里\\n\\ndef run_agent(fn, task):\\n    \\"\\"\\"调用 Agent 处理一个任务，返回答案字符串。\\"\\"\\"\\n    return fn(task)   # 空位在这里\\n\\nsrc = \\"\\"\\"def forward(task):\\n    a, b = task\\n    return str(a * b)\\n\\"\\"\\"\\nagent = compile_agent(src)\\nassert callable(agent), \\"编译结果应可调用\\"\\nassert run_agent(agent, (7, 8)) == \\"56\\"\\nprint(\\"收获：元 Agent 生成的源码可以 exec 成可调用函数并执行。\\")",
   "id": "8dc96614125d0f6b"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "**作业 3：补全评审打分的阈值决策**\\n\\n评审返回分数 dict。补全 \`should_accept(review, threshold=6)\`：\`overall\` 达到阈值且 \`contribution\` 不低于 3 时接受，否则拒绝；再补全 \`merge_reviews(reviews)\`，对多个评审的 \`overall\` 取平均。\\n\\n小提示：\`overall\` 与 \`contribution\` 直接取键；平均用 \`sum(...) / len(...)\`。",
   "id": "5f41d97a1f2d7e78"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": "def should_accept(review, threshold=6):\\n    \\"\\"\\"overall 达到阈值且 contribution 不低于 3 时接受。\\"\\"\\"\\n    return review[\\"overall\\"] >= threshold and review[\\"contribution\\"] >= 3   # 空位在这里\\n\\ndef merge_reviews(reviews):\\n    \\"\\"\\"多个评审的 overall 取平均。\\"\\"\\"\\n    return sum(r[\\"overall\\"] for r in reviews) / len(reviews)   # 空位在这里\\n\\nrev = {\\"soundness\\": 5, \\"presentation\\": 4, \\"contribution\\": 6, \\"overall\\": 6}\\nassert should_accept(rev) is True, \\"overall 6 且 contribution 6 应接受\\"\\nassert should_accept({\\"overall\\": 6, \\"contribution\\": 2}) is False\\nassert merge_reviews([{\\"overall\\": 5}, {\\"overall\\": 7}]) == 6.0\\nprint(\\"收获：评审阈值判断与多评审平均。\\")",
   "id": "f0e87f81ee4bea58"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## 参考资料\\n\\n- Hu et al., [Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435), 2024 — 元 Agent 在代码空间里设计 Agent，本讲三要素的第一档。代码 https://github.com/ShengranHu/ADAS\\n- Lu et al., [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292), 2024 — idea → 实验 → 论文 → 评审的全自动科研闭环。代码 https://github.com/SakanaAI/AI-Scientist\\n- Novikov et al., [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131), 2025 — 以代码为基因组、以评测器为适应度的进化式编码 Agent\\n- Romera-Paredes et al., [FunSearch: Mathematical discoveries from program search with LLMs](https://arxiv.org/abs/2312.02174), 2023 — AlphaEvolve 的前身，LLM 进化单个函数\\n- Wang et al., [Quality-Diversity algorithms: A generic definition and an illustration](https://arxiv.org/abs/2103.04313), 2021 — MAP-Elites 与质量多样性思想，AlphaEvolve 种群的算法来源\\n- Clune, [AI-Generating Algorithms: An Alternate Paradigm](https://arxiv.org/abs/1901.01346), 2019 — 本讲理论源头，AI-GA 三支柱\\n- 本仓库 \`llm_client.py\` — 所有 LLM 演示的统一入口，mock 模式保证离线可执行",
   "id": "b70e03f8ea8d2f5c"
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
