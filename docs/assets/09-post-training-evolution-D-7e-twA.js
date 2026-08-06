const n=`{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-00",
   "source": "# 后训练演进：从 Chatbot 到 Agent\\n"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-01",
   "source": "> 前几讲各处理了一类方法：06 讲给出强化学习的算法引擎（STaR、GRPO、DAPO），07 讲展示 Agent 自我改进的目标形态，08 讲把搜索与深度研究做成一个具体的 Agent。这些方法分布在不同的章节里，各自回答一个局部问题。\\n>\\n> 本讲把它们收拢成一条时间线，回答一个总的问题：预训练只给了模型续写文本的能力，它怎样变成听话的助手，又怎样变成会用工具完成任务的 Agent。我们跟随一个模型的一生，观察每一阶段喂进去什么新信号，以及训练信号如何从人类手里一步步交到环境手里。\\n"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-02",
   "source": "后训练指预训练结束之后，为了让模型服务于特定目标而做的全部额外训练。预训练优化的是下一个 token 的似然，模型因此学会语法、知识和一部分世界模型，但不会听指令——指令对它是待续写的文本，不是待执行的任务。要得到能用的模型，必须在预训练之后继续训练，这就是后训练。\\n\\n整条后训练的推进主线是训练信号来源的迁移。SFT 用人类书写的示范，RLHF 用人类给出的偏好排序，RLVR 用可验证的答案正确性，Agent 后训练用环境执行结果。每一次迁移都源于前一种信号太贵、太稀疏，或太容易被模型钻空子。我们把这种\\"信号从人流向环境\\"的过程作为贯穿全讲的线索。\\n\\n第一节从起点出发，观察一个只有预训练能力的模型面对指令时输出什么。\\n"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-03",
   "source": "## 1. 后训练是什么：从续写器到听话的模型\\n\\n一个只有预训练能力的模型是文本续写器。给定任何前缀，它输出最像\\"互联网文本的继续\\"的下一个 token。把一段指令拼在开头，它也会顺着把指令续写下去，而不是把指令当作待执行的任务。预训练模型不缺少能力，缺少的是方向——它优化的目标不是用户意图。\\n\\n从续写器到能完成任务的 Agent，训练信号在四个阶段里换了四种来源。下表是整讲的地图，每一行回答两个问题：模型从哪种信号里学到什么，以及这种信号为什么在下个阶段被替换。\\n\\n| 阶段 | 信号来源 | 数据形态 | 目标 | 代表工作 |\\n|---|---|---|---|---|\\n| SFT | 人类示范 | (指令, 期望回复) | 学怎么做 | FLAN、InstructGPT-SFT |\\n| RLHF | 人类偏好 | 候选回复排序 | 学什么是好 | InstructGPT、ChatGPT |\\n| RLVR | 可验证正确性 | 规则判据 | 学什么是正确 | DeepSeek-R1、DAPO |\\n| Agent 后训练 | 环境执行结果 | 整条轨迹 + 成败 | 学什么动作序列能完成任务 | RLEF、WebRL、MiRA |\\n\\n先用一个最小的续写器建立\\"模型不会听话\\"的直觉。\\n"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-04",
   "source": "import numpy as np\\nfrom collections import defaultdict\\n\\n# 迷你语料：只有正常对话，没有\\"指令-执行\\"这种格式\\ncorpus = [\\"天气不错我们出门散步\\", \\"晚餐吃什么我想吃面条\\", \\"今天的工作完成了早点休息\\"]\\n\\n# 字符级 bigram 统计：记录每个字符后各字符出现的次数\\ncnt = defaultdict(lambda: defaultdict(int))\\nalphabet = set()\\nfor sent in corpus:\\n    for ch in sent:\\n        alphabet.add(ch)\\n    for ch, nxt in zip(sent, sent[1:]):\\n        cnt[ch][nxt] += 1\\nalphabet = sorted(alphabet)\\n\\n\\ndef next_char(ch):\\n    \\"\\"\\"按条件概率从 ch 的下一个字符分布里采样。\\"\\"\\"\\n    if ch not in cnt or len(cnt[ch]) == 0:\\n        return np.random.choice(alphabet)\\n    options = list(cnt[ch].keys())\\n    probs = np.array([cnt[ch][c] for c in options], dtype=float)\\n    probs /= probs.sum()\\n    return np.random.choice(options, p=probs)\\n\\n\\ndef continue_text(prefix, length=24):\\n    \\"\\"\\"从 prefix 出发续写 length 个字符。\\"\\"\\"\\n    out = list(prefix)\\n    ch = out[-1]\\n    for _ in range(length):\\n        nxt = next_char(ch)\\n        out.append(nxt)\\n        ch = nxt\\n    return \\"\\".join(out)\\n\\n\\nnp.random.seed(42)\\nprompt = \\"用户：请计算 12 加 8 等于几？\\"\\nprint(\\"输入指令：\\", prompt)\\nprint(\\"模型续写：\\", continue_text(prompt))\\nprint()\\nprint(\\"关键观察：续写器没有回答数字，而是顺着指令写了一段像对话的文本。\\")\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-05",
   "source": "## 2. SFT 与 RLHF：Chatbot 时代的对齐\\n\\n\`SFT\`（监督微调）把人类书写的示范当监督信号。做法是收集一批 (指令, 期望回复)，用交叉熵微调模型，让示范里的输出概率变高。SFT 只做这一件事——它不引入任何\\"好坏\\"的判断，示范里没有出现的回复，模型不会学到，也无法超越示教者的水平。\\n\\n\`RLHF\`（基于人类反馈的强化学习）补上\\"好坏\\"这一维度。先让模型对同一指令输出多个候选，标注员排序，训练一个\`RM\`（奖励模型）给回复打分，再用强化学习最大化 RM 分数，同时用 KL 项约束策略不要偏离 SFT 模型太远。RM 分数是标注偏好的标量代理，只能衡量\\"这段文本好不好\\"。\\n\\n下面用一套 toy 数据，把 SFT 的交叉熵和 RLHF 的优化目标逐项手算，看同一组概率被推向哪里。RLVR 与 Agent 后训练的损失形态留在第三节，届时一并对比。\\n"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-06",
   "source": "import numpy as np\\n\\n\\ndef softmax(x):\\n    \\"\\"\\"把 logits 沿最后一维归一化成概率分布。\\"\\"\\"\\n    e = np.exp(x - x.max(axis=-1, keepdims=True))\\n    return e / e.sum(axis=-1, keepdims=True)\\n\\n\\n# toy 设定：一条指令 x，模型的输出词表只有 4 个 token\\ntokens = [\\"A\\", \\"B\\", \\"C\\", \\"D\\"]\\nlogits = np.zeros(4)                          # 初始 logits 全 0 → 均匀分布\\nprobs = softmax(logits)\\n\\n# 三类标签，对应三种时代的监督信号\\ndemo_token = \\"B\\"                              # SFT：人类示范期望回复是 B\\nrm_scores = np.array([0.5, 1.0, -0.2, 0.0])   # RLHF：RM 给四个候选打分\\ncorrectness = np.array([0.0, 1.0, 0.0, 1.0])  # RLVR：B 与 D 正确\\n\\nprint(\\"token        :\\", tokens)\\nprint(\\"当前概率 π   :\\", np.round(probs, 3))\\nprint(\\"SFT 示范 token:\\", demo_token)\\nprint(\\"RLHF RM 分数 :\\", rm_scores)\\nprint(\\"RLVR 正确性  :\\", correctness)\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-07",
   "source": "SFT 的损失就是输出序列的交叉熵。在单 token 的设定里，它退化为\\n\\n$$L_{\\\\mathrm{SFT}} = -\\\\log \\\\pi_\\\\theta(y_{\\\\mathrm{demo}}).$$\\n\\n梯度只把示范 token 的概率推高。下面扫描示范 token 的概率从 0.1 到 0.9，观察损失怎样随概率变化，再用数值差分看梯度方向。\\n"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-08",
   "source": "import numpy as np\\n\\ndemo_idx = 1  # token B\\n\\n# 扫描示范 token 的概率，看 SFT 损失随概率的变化\\nprint(\\"示范概率 p(B) | SFT 损失 -log p\\")\\nfor p in np.arange(0.1, 0.95, 0.1):\\n    print(f\\"      {p:.1f}      |   {-np.log(p):.4f}\\")\\nprint()\\n\\n\\ndef sft_loss(theta):\\n    \\"\\"\\"单 token 设定下的 SFT 交叉熵损失。\\"\\"\\"\\n    return -np.log(softmax(theta)[demo_idx])\\n\\n\\ntheta = np.zeros(4)\\neps = 1e-4\\ngrad = np.array([(sft_loss(theta + eps * np.eye(4)[j]) -\\n                  sft_loss(theta - eps * np.eye(4)[j])) / (2 * eps)\\n                 for j in range(4)])\\nprint(\\"SFT 梯度 dL/dθ :\\", np.round(grad, 3))\\nprint(\\"关键观察：示范 token B 的梯度为负（概率被推高），其余 token 梯度为正（概率被压低）。\\")\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-09",
   "source": "RLHF 的训练分两步。第一步用标注排序训练 RM：对同一指令的一组候选，用成对排序损失让被偏好的回复得分更高。第二步把 RM 分数当作奖励优化模型，同时用 KL 项约束策略不要偏离参考策略太远。整体要最小化的目标写作\\n\\n$$L_{\\\\mathrm{RLHF}} = -\\\\mathbb{E}_{y\\\\sim\\\\pi}\\\\big[r_\\\\theta(x,y)\\\\big] + \\\\beta\\\\, D_{\\\\mathrm{KL}}\\\\big(\\\\pi \\\\,\\\\|\\\\, \\\\pi_{\\\\mathrm{ref}}\\\\big).$$\\n\\nRM 分数高的候选被推高，KL 项把它拉回参考策略，防止模型钻 RM 的空子走得太远。\\n"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-10",
   "source": "import numpy as np\\n\\nrm_scores = np.array([0.5, 1.0, -0.2, 0.0])\\nbeta = 0.5\\nref = np.full(4, 0.25)  # 参考策略：均匀分布\\n\\n\\ndef rlhf_loss(theta):\\n    \\"\\"\\"RLHF 目标：负期望 RM 分数 + KL 约束。\\"\\"\\"\\n    p = softmax(theta)\\n    reward = rm_scores @ p\\n    kl = (p * (np.log(p) - np.log(ref))).sum()\\n    return -reward + beta * kl\\n\\n\\ntheta = np.zeros(4)\\nprint(\\"logit θ(B) | p(B)   | E[r]   | β·KL   | RLHF 损失\\")\\nfor tb in [-1.5, -0.5, 0.0, 0.5, 1.5]:\\n    th = theta.copy()\\n    th[1] = tb\\n    p = softmax(th)\\n    kl = (p * (np.log(p) - np.log(ref))).sum()\\n    print(f\\"  {tb:+.1f}   | {p[1]:.3f} | {rm_scores@p:.3f} | {beta * kl:.3f} | {rlhf_loss(th):.4f}\\")\\n\\neps = 1e-4\\ngrad = np.array([(rlhf_loss(theta + eps * np.eye(4)[j]) -\\n                  rlhf_loss(theta - eps * np.eye(4)[j])) / (2 * eps)\\n                 for j in range(4)])\\nprint()\\nprint(\\"RLHF 梯度 dL/dθ :\\", np.round(grad, 3))\\nprint(\\"关键观察：RM 分数高于均值(0.325)的 token 被推高，低于均值的被压低；KL 项把整体拉回均匀。\\")\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-11",
   "source": "RLHF 的 RM 是代理目标，模型可能骗过它。如果 RM 只统计回复里是否出现某个关键词，模型就会堆砌关键词拿高分，而这些文本对用户没有价值——这是 \`reward hacking\`。RM 还只能衡量文本质量，衡量不了\\"这个动作序列能不能完成任务\\"。Constitutional AI 曾尝试压缩人工标注成本，让 AI 依据原则列表自我批评与修订，04 讲已精读，这里不再展开。\\n\\n这两条限制是下一阶段更换信号来源的直接动机：与其学一个奖励函数，不如检查答案对不对。\\n"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-12",
   "source": "## 3. Agent 化后训练：工具、执行与反馈\\n\\n去掉学习式 RM，改用规则化的可验证奖励，就是 \`RLVR\`（用可验证奖励做强化学习）。奖励不再是神经网络，而是一个判据：答案与标准答案是否一致、代码是否通过测试。DeepSeek-R1 的 R1-Zero 从 base model 出发，不做任何 SFT，纯用 RLVR 训练，长链推理自发涌现——模型自己学会先想再答，因为想得越多越容易拿分。奖励只要对、且可验证，聪明的策略不需要人示范。\\n\\nRLVR 的优化用组内 advantage：同一指令采样一组候选，把每个候选的奖励减去组均值、除以组标准差。这就是 \`GRPO\` 的核心，它不需要价值网络。下面先手算这组归一化 advantage，再把三种时代的损失形态并排对比，看它们各自把概率推向哪里。\\n"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-13",
   "source": "import numpy as np\\n\\n\\ndef group_advantage(rewards, eps=1e-9):\\n    \\"\\"\\"组内归一化 advantage；(r - mean) / (std + eps) 防全对/全错组除零。\\"\\"\\"\\n    return (rewards - rewards.mean()) / (rewards.std() + eps)\\n\\n\\nrewards = np.array([1.0, -1.0, 0.0, 1.0])\\nmean = rewards.mean()\\nstd = rewards.std()\\nadvantage = group_advantage(rewards)\\n\\nprint(\\"奖励 r          :\\", rewards)\\nprint(\\"组均值          :\\", round(mean, 3))\\nprint(\\"组标准差        :\\", round(std, 3))\\nprint(\\"组内 advantage  :\\", np.round(advantage, 3))\\nprint(\\"关键观察：正确候选(1.0)获得正 advantage 被推高，错误候选获得负 advantage 被压低。\\")\\n\\n# GRPO 目标随正确 token 概率的变化\\nprint()\\nprint(\\"p(D) | GRPO 加权对数似然目标\\")\\nfor p in [0.1, 0.3, 0.5, 0.7, 0.9]:\\n    pi = np.array([(1 - p) / 3] * 3 + [p])\\n    adv = group_advantage(np.array([0.0, 1.0, 0.0, 1.0]))\\n    obj = -(adv * np.log(pi)).sum() / 4\\n    print(f\\" {p:.1f} |        {obj:.4f}\\")\\n\\n# 全对组与全错组：advantage 恒为 0\\nprint()\\nfor name, r in [(\\"全对组\\", np.array([1.0, 1.0, 1.0, 1.0])),\\n                (\\"全错组\\", np.array([-1.0, -1.0, -1.0, -1.0]))]:\\n    adv = group_advantage(r)\\n    print(f\\"{name} r={r.tolist()} -> advantage={np.round(adv, 3)}\\")\\nprint(\\"关键观察：组内无差异时 advantage 全为 0，这一组样本不产生任何梯度。\\")\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-14",
   "source": "import numpy as np\\n\\ndemo_idx = 1\\nrm_scores = np.array([0.5, 1.0, -0.2, 0.0])\\ncorrectness = np.array([0.0, 1.0, 0.0, 1.0])\\nbeta = 0.5\\nref = np.full(4, 0.25)\\n\\n\\ndef sft_loss(theta):\\n    \\"\\"\\"SFT 交叉熵。\\"\\"\\"\\n    return -np.log(softmax(theta)[demo_idx])\\n\\n\\ndef rlhf_loss(theta):\\n    \\"\\"\\"RLHF 目标：负期望 RM 分数 + KL 约束。\\"\\"\\"\\n    p = softmax(theta)\\n    return -(rm_scores @ p) + beta * (p * (np.log(p) - np.log(ref))).sum()\\n\\n\\ndef grpo_loss(theta):\\n    \\"\\"\\"组内 advantage 加权的对数似然（策略梯度的等价目标）。\\"\\"\\"\\n    p = softmax(theta)\\n    adv = group_advantage(correctness)\\n    return -(adv @ np.log(p)).sum() / len(correctness)\\n\\n\\ndef grad_of(loss, theta):\\n    \\"\\"\\"数值差分求梯度。\\"\\"\\"\\n    eps = 1e-4\\n    return np.array([(loss(theta + eps * np.eye(4)[j]) -\\n                      loss(theta - eps * np.eye(4)[j])) / (2 * eps)\\n                     for j in range(4)])\\n\\n\\ntheta = np.zeros(4)\\nfor name, loss in [(\\"SFT\\", sft_loss), (\\"RLHF\\", rlhf_loss), (\\"RLVR/GRPO\\", grpo_loss)]:\\n    g = grad_of(loss, theta)\\n    direction = [\\"↑推高\\" if x < 0 else \\"↓压低\\" for x in g]\\n    print(f\\"{name:9s} 梯度 {np.round(g, 3)}  方向 {direction}\\")\\nprint()\\nprint(\\"关键观察：SFT 只推高示范 token B；RLHF 推高 RM 分数高的 A、B；RLVR 推高正确的 B、D。\\")\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-15",
   "source": "上面的对比是静态的：给定一套概率看梯度方向。真实的训练是动态的——每更新一步，模型重新采样、环境重新打分。下面在 toy bandit 上分别实现 \`PPO\`（近端策略优化）与 GRPO，观察两种 advantage 估计方式如何影响收敛。\\n\\n设定是一个上下文 bandit：5 个问题，每个问题有唯一正确的动作（共 10 个候选）。环境按规则判定动作对错，奖励 +1/-1，没有可 hack 的中间层。PPO 用一个价值网络估计基线，GRPO 用组内均值做基线。两者都要回答同一件事：让每个问题的正确动作概率升上去。\\n"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-16",
   "source": "import numpy as np\\n\\nrng = np.random.default_rng(42)\\n\\nNUM_Q = 5\\nNUM_A = 10\\ncorrect = rng.integers(0, NUM_A, size=NUM_Q)   # 每个问题的正确动作\\n\\nlogits = np.zeros((NUM_Q, NUM_A))   # 策略参数\\nvalue = np.zeros(NUM_Q)             # 价值网络：每个问题一个标量基线\\nlr_policy = 0.05\\nlr_value = 0.1\\n\\n\\ndef sample_one():\\n    \\"\\"\\"采样一条 (问题, 动作)，返回问题、动作、奖励与采样概率。\\"\\"\\"\\n    q = int(rng.integers(0, NUM_Q))\\n    p = softmax(logits[q])\\n    a = int(rng.choice(NUM_A, p=p))\\n    r = 1.0 if a == correct[q] else -1.0\\n    return q, a, r, p\\n\\n\\ndef ppo_step():\\n    \\"\\"\\"PPO 简化版：advantage = 奖励 - 价值基线，策略按 advantage 加权更新。\\"\\"\\"\\n    for _ in range(32):\\n        q, a, r, p = sample_one()\\n        adv = r - value[q]\\n        logits[q] += lr_policy * adv * (np.eye(NUM_A)[a] - p)\\n        value[q] += lr_value * (r - value[q])   # 价值向奖励回归\\n\\n\\nppl_acc = []\\nfor it in range(80):\\n    ppo_step()\\n    p = softmax(logits)\\n    acc = (np.argmax(p, axis=1) == correct).mean()\\n    ppl_acc.append(acc)\\n\\nprint(\\"PPO 正确率（每 20 轮）:\\", [round(float(x), 2) for x in ppl_acc[::20]])\\nprint(\\"PPO 末期每个问题正确动作的概率:\\",\\n      np.round(softmax(logits)[np.arange(NUM_Q), correct], 2))\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-17",
   "source": "import numpy as np\\n\\nrng = np.random.default_rng(7)\\n\\n# 与 PPO 同一组问题，重新初始化策略\\nlogits_g = np.zeros((NUM_Q, NUM_A))\\nlr_g = 0.2\\nK = 4  # 每组采样 4 个动作\\n\\n\\ndef grpo_update(q):\\n    \\"\\"\\"GRPO：对一个问题采样一组动作，组内归一化 advantage，无价值网络。\\"\\"\\"\\n    p = softmax(logits_g[q])\\n    acts = rng.choice(NUM_A, size=K, p=p)\\n    rewards = np.array([1.0 if a == correct[q] else -1.0 for a in acts])\\n    adv = group_advantage(rewards)\\n    for i, a in enumerate(acts):\\n        onehot = np.zeros(NUM_A)\\n        onehot[a] = 1.0\\n        logits_g[q] += lr_g / K * adv[i] * (onehot - p)\\n    return rewards\\n\\n\\ng_acc = []\\nwasted_history = []\\nfor it in range(80):\\n    wasted = 0\\n    for _ in range(8):   # 每轮 8 个组\\n        q = int(rng.integers(0, NUM_Q))\\n        rewards = grpo_update(q)\\n        if rewards.max() == rewards.min():\\n            wasted += 1\\n    wasted_history.append(wasted)\\n    p = softmax(logits_g)\\n    g_acc.append((np.argmax(p, axis=1) == correct).mean())\\n\\nprint(\\"GRPO 正确率（每 20 轮）:\\", [round(float(x), 2) for x in g_acc[::20]])\\nprint(\\"每轮全对/全错组数量（前 10 轮）:\\", wasted_history[:10])\\nprint(\\"关键观察：GRPO 不需要价值网络，但全对组与全错组不产生任何梯度，这批样本白算。\\")\\n\\nimport matplotlib.pyplot as plt\\n\\nplt.figure(figsize=(6.2, 3.8))\\nplt.plot(ppl_acc, label=\\"PPO (critic baseline)\\")\\nplt.plot(g_acc, label=\\"GRPO (group baseline)\\")\\nplt.xlabel(\\"round\\")\\nplt.ylabel(\\"accuracy\\")\\nplt.title(\\"Correct-arm accuracy on a toy bandit\\")\\nplt.legend()\\nplt.tight_layout()\\nplt.show()\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-18",
   "source": "RLVR 解决了\\"文本好不好\\"的问题，但它只适用于能写判据的任务——数学、代码、谜题。开放问题（写一封得体的邮件）没有标准答案，也就没有验证器。Agent 化后训练的关键转移是：任务虽然开放，但环境本身可以当验证器。任务成功可以用测试是否通过、目标是否达成、终端状态是否收敛来判定。\\n\\n训练单位也从一段文本变成一整条轨迹。动作空间从下一个 token 扩展成工具调用序列：查数据库、执行代码、写文件、返回结果。奖励不再是任务开始前写死的判据，而是环境跑出来的结果——可能稀疏，也可能延迟到轨迹末端才出现。下面先把四个阶段的信号来源放在同一张图上，看成本与可靠性如何一路变化。\\n"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-19",
   "source": "import matplotlib.pyplot as plt\\n\\n# 四个信号来源，投影到 (标注成本, 抗 hack 可靠性) 平面\\nsignals = [\\n    (\\"SFT human demo\\", 1.0, 0.30),\\n    (\\"RLHF human pref\\", 0.8, 0.50),\\n    (\\"RLVR rule\\", 0.4, 0.90),\\n    (\\"Agent RL env\\", 0.2, 1.00),\\n]\\nnames = [s[0] for s in signals]\\ncosts = [s[1] for s in signals]\\nrel = [s[2] for s in signals]\\n\\nplt.figure(figsize=(6.2, 4.4))\\nplt.scatter(costs, rel, s=260, c=range(4), cmap=\\"viridis\\")\\nfor i, name in enumerate(names):\\n    plt.annotate(name, (costs[i], rel[i]),\\n                 textcoords=\\"offset points\\", xytext=(6, 4), fontsize=9)\\nplt.xlabel(\\"annotation cost (lower is cheaper)\\")\\nplt.ylabel(\\"resistance to reward hacking\\")\\nplt.title(\\"Reward signal sources across post-training stages\\")\\nplt.xlim(0, 1.2)\\nplt.ylim(0, 1.2)\\nplt.tight_layout()\\nplt.show()\\n\\nprint(\\"关键观察：信号来源从右上角迁向左下角——标注成本一路下降，可靠性一路上升。\\")\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-20",
   "source": "轨迹级奖励把整条轨迹压缩成一个 +1 或 -1。如果一条轨迹有几十步，只有最后一步给信号，模型无法判断中间哪一步错了——这是 \`credit assignment\` 问题，也是稀疏延迟奖励的核心困难。步级（里程碑）奖励把总奖励拆到每一步：每达成一个子目标就给一份，末尾成功再给一份，模型因此收到过程信号。\\n\\n下面在一个 toy 三步环境里数值对比两种奖励的梯度方差与收敛速度。轨迹奖励在每一步之间分摊同一个终局值，方差被放大；里程碑奖励每步只携带自己的子目标信号，方差更小。\\n"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-21",
   "source": "import numpy as np\\n\\nrng = np.random.default_rng(1)\\nN_ACT = 3\\ncorrect_steps = np.array([0, 1, 2])   # 第 3 步的正确选择决定终局成败\\n\\n\\ndef sample_trajectory(logits):\\n    \\"\\"\\"按策略逐步采样三步动作。\\"\\"\\"\\n    return np.array([rng.choice(N_ACT, p=softmax(logits[t])) for t in range(3)])\\n\\n\\ndef trajectory_reward(acts):\\n    \\"\\"\\"只有三步全对给 +1，否则 -1。\\"\\"\\"\\n    return 1.0 if (acts == correct_steps).all() else -1.0\\n\\n\\ndef milestone_rewards(acts):\\n    \\"\\"\\"每步子目标达成 +0.3，末尾成功再 +0.1。\\"\\"\\"\\n    per = 0.3 * (acts == correct_steps).astype(float)\\n    if (acts == correct_steps).all():\\n        per[2] += 0.1\\n    return per\\n\\n\\ndef reinforce_grad(logits, mode):\\n    \\"\\"\\"一次 episode 的策略梯度估计。\\"\\"\\"\\n    acts = sample_trajectory(logits)\\n    g = np.zeros_like(logits)\\n    if mode == \\"trajectory\\":\\n        R = trajectory_reward(acts)\\n        for t in range(3):\\n            g[t] = R * (np.eye(N_ACT)[acts[t]] - softmax(logits[t]))\\n    else:\\n        r = milestone_rewards(acts)\\n        for t in range(3):\\n            g[t] = r[t] * (np.eye(N_ACT)[acts[t]] - softmax(logits[t]))\\n    return g, acts\\n\\n\\n# 从同一均匀策略出发，各采样 500 次梯度，比较方差\\nlogits0 = np.zeros((3, N_ACT))\\ntraj_grads, mile_grads = [], []\\nfor _ in range(500):\\n    g1, _ = reinforce_grad(logits0, \\"trajectory\\")\\n    g2, _ = reinforce_grad(logits0, \\"milestone\\")\\n    traj_grads.append(g1)\\n    mile_grads.append(g2)\\nvar_t = np.array(traj_grads).var()\\nvar_m = np.array(mile_grads).var()\\nprint(\\"均匀策略下梯度方差：轨迹奖励\\", round(var_t, 4), \\" vs 里程碑奖励\\", round(var_m, 4))\\nprint()\\n\\n# 训练对比：400 轮，画 50 轮滑动平均的回报\\nM = 400\\nlr = 0.1\\n\\n\\ndef train(mode):\\n    \\"\\"\\"跑 M 轮 REINFORCE，返回逐轮终局回报。\\"\\"\\"\\n    logits = np.zeros((3, N_ACT))\\n    returns = []\\n    for it in range(M):\\n        g, acts = reinforce_grad(logits, mode)\\n        logits += lr * g\\n        returns.append(1.0 if (acts == correct_steps).all() else -1.0)\\n    return returns\\n\\n\\ndef running_mean(x, w=50):\\n    \\"\\"\\"w 窗口的滑动平均。\\"\\"\\"\\n    out = []\\n    for i in range(len(x)):\\n        lo = max(0, i - w + 1)\\n        out.append(np.mean(x[lo:i + 1]))\\n    return out\\n\\n\\nret_t = running_mean(train(\\"trajectory\\"))\\nret_m = running_mean(train(\\"milestone\\"))\\nprint(\\"400 轮末的滑动平均回报：轨迹奖励\\", round(ret_t[-1], 3),\\n      \\" vs 里程碑奖励\\", round(ret_m[-1], 3))\\n\\nimport matplotlib.pyplot as plt\\n\\nplt.figure(figsize=(6.2, 3.8))\\nplt.plot(ret_t, label=\\"trajectory reward\\")\\nplt.plot(ret_m, label=\\"milestone reward\\")\\nplt.xlabel(\\"episode\\")\\nplt.ylabel(\\"running mean return\\")\\nplt.title(\\"Credit assignment: sparse vs dense reward\\")\\nplt.legend()\\nplt.tight_layout()\\nplt.show()\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-22",
   "source": "把整条链路合起来，就是一个环境反馈飞轮。模型输出动作序列 → 环境执行并判定成败 → 成败信号当作奖励更新策略 → 过滤成功的轨迹、重新采样。这就是 STaR 与 WebRL 的骨架。失败的任务有两种命运：直接丢弃，或改造成可验证的形式重新注入训练集，后者让数据池随训练逐轮膨胀。\\n\\n下面用一个 toy 算术任务把这个飞轮跑起来。模型（llm_client 的 mock 实例）对每个问题提出答案，环境核对答案并返回 +1/-1，我们据此更新一个轻量的答案策略。每轮统计成功率，观察它逐轮上升，同时演示失败任务的两种命运。\\n"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-23",
   "source": "import sys, os, re\\nimport numpy as np\\n\\n# 统一走仓库根目录的 llm_client.py，mock 模式下同样可跑\\n_root = os.path.abspath(os.getcwd())\\nwhile not os.path.exists(os.path.join(_root, \\"llm_client.py\\")):\\n    _root = os.path.dirname(_root)\\n    if _root == os.path.dirname(_root):\\n        break\\nif _root not in sys.path:\\n    sys.path.insert(0, _root)\\nfrom llm_client import get_llm\\n\\nclient = get_llm()\\nprint(\\"当前 LLM 模式:\\", \\"mock\\" if client.is_mock else \\"real API\\")\\n\\n# 问题池：前三题 mock 能直接算出加法，最后一题乘法 mock 不会处理\\nquestions = [\\"计算 13 加 24 等于几\\", \\"计算 35 加 19 等于几\\",\\n             \\"计算 7 加 46 等于几\\", \\"计算 12 乘 7 等于几\\"]\\nanswers = {\\"计算 13 加 24 等于几\\": 37, \\"计算 35 加 19 等于几\\": 54,\\n           \\"计算 7 加 46 等于几\\": 53, \\"计算 12 乘 7 等于几\\": 84}\\n\\n\\ndef extract_number(text):\\n    \\"\\"\\"从回复里提取第一个整数；提取不到返回 None。\\"\\"\\"\\n    m = re.search(r\\"-?\\\\d+\\", text)\\n    return int(m.group()) if m else None\\n\\n\\ndef propose(q):\\n    \\"\\"\\"模型提出答案：问一次 LLM，解析出数字。\\"\\"\\"\\n    reply = client.chat([{\\"role\\": \\"user\\", \\"content\\": f\\"{q}，只输出数字。\\"}])\\n    return extract_number(reply)\\n\\n\\nproposal = {q: propose(q) for q in questions}\\n\\n# 候选答案池：mock 的提案 + 干扰项；乘法题的正确答案不在池里\\ncandidate_pool = {}\\nfor q in questions:\\n    cands = []\\n    if proposal[q] is not None:\\n        cands.append(proposal[q])\\n    cands += [answers[q] + 3, answers[q] - 7]   # 干扰项\\n    candidate_pool[q] = list(dict.fromkeys(cands))\\n\\nfor q in questions:\\n    print(f\\"{q}: 提案 {proposal[q]}, 候选池 {candidate_pool[q]}\\")\\nprint(\\"关键观察：前三题的提案就是正确答案；乘法题的提案是 None，正确答案不在池中。\\")\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-24",
   "source": "rng = np.random.default_rng(3)\\nlr = 0.2\\nlogits_map = {q: np.zeros(len(candidate_pool[q])) for q in questions}\\nq_fail = questions[-1]\\n\\n\\ndef run_round():\\n    \\"\\"\\"跑一轮 RL：每个问题采样答案，环境核对，REINFORCE 更新。返回逐题是否成功。\\"\\"\\"\\n    per_q = {}\\n    for q, cands in candidate_pool.items():\\n        p = softmax(logits_map[q])\\n        idx = int(rng.choice(len(cands), p=p))\\n        a = cands[idx]\\n        r = 1.0 if a == answers[q] else -1.0\\n        logits_map[q] += lr * r * (np.eye(len(cands))[idx] - p)\\n        per_q[q] = (a == answers[q])\\n    return per_q\\n\\n\\ndef correct_prob():\\n    \\"\\"\\"正确答案的平均策略概率（只统计正确答案在池里的问题）。\\"\\"\\"\\n    vals = []\\n    for q, cands in candidate_pool.items():\\n        if answers[q] in cands:\\n            vals.append(float(softmax(logits_map[q])[cands.index(answers[q])]))\\n    return np.mean(vals)\\n\\n\\ndef block_means(x):\\n    \\"\\"\\"把逐轮序列切成每 15 轮一块，返回块均值。\\"\\"\\"\\n    return [round(float(np.mean(x[i * 15:(i + 1) * 15])), 2)\\n            for i in range(len(x) // 15)]\\n\\n\\n# 不改造失败任务：连续 60 轮，观察成功率与正确答案概率\\nrates, probs, fail_rate = [], [], []\\nfor it in range(60):\\n    per_q = run_round()\\n    rates.append(np.mean(list(per_q.values())))\\n    probs.append(correct_prob())\\n    fail_rate.append(per_q[q_fail])\\n\\nprint(\\"整体采样成功率（每 15 轮）:\\", block_means(rates))\\nprint(\\"正确答案平均概率（每 15 轮）:\\", block_means(probs))\\nprint(\\"乘法问题采样成功率（每 15 轮）:\\", block_means(fail_rate))\\nprint(\\"关键观察：前 3 题正确答案概率逐块上升；乘法题的正确答案不在池中，成功率恒为 0——任务被丢弃。\\")\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-25",
   "source": "import matplotlib.pyplot as plt\\n\\n# 命运 B：改造——把乘法拆成连加，用 mock 逐个核对子目标，环境验证终值\\nacc = 12\\nverified = None\\nfor step in range(6):\\n    target = acc + 12\\n    reply = client.chat([{\\"role\\": \\"user\\",\\n                          \\"content\\": f\\"检查 {acc} 加 12 是否等于 {target}，只输出数字。\\"}])\\n    if extract_number(reply) != target:\\n        break\\n    acc = target\\nverified = acc\\n\\npool_before = sum(len(c) for c in candidate_pool.values())\\nprint(\\"改造得到的验证答案:\\", verified)\\nassert verified == answers[q_fail]\\n\\n# 正确答案进入候选池，策略重新初始化\\ncandidate_pool[q_fail] = list(dict.fromkeys(candidate_pool[q_fail] + [verified]))\\nlogits_map[q_fail] = np.zeros(len(candidate_pool[q_fail]))\\npool_after = sum(len(c) for c in candidate_pool.values())\\nprint(\\"改造后乘法问题候选池:\\", candidate_pool[q_fail])\\nprint(\\"数据池大小: 改造前\\", pool_before, \\"→ 改造后\\", pool_after)\\n\\n# 继续跑 100 轮 RL，观察乘法问题随数据池膨胀而学会\\nrates2, prob2, fail2 = [], [], []\\nfor it in range(100):\\n    per_q = run_round()\\n    rates2.append(np.mean(list(per_q.values())))\\n    prob2.append(float(softmax(logits_map[q_fail])[candidate_pool[q_fail].index(84)]))\\n    fail2.append(per_q[q_fail])\\n\\nprint(\\"改造后整体采样成功率（每 20 轮）:\\", block_means(rates2))\\nprint(\\"改造后乘法问题正确概率（每 20 轮）:\\", block_means(prob2))\\nprint(\\"改造后乘法问题采样成功率（每 20 轮）:\\", block_means(fail2))\\n\\nplt.figure(figsize=(6.2, 3.8))\\nplt.plot(rates, label=\\"overall (before curriculum)\\")\\nplt.plot(rates2, label=\\"overall (after curriculum)\\")\\nplt.plot(prob2, label=\\"correct prob of failed task\\")\\nplt.xlabel(\\"round\\")\\nplt.ylabel(\\"success rate / prob\\")\\nplt.title(\\"Environment feedback flywheel\\")\\nplt.legend()\\nplt.tight_layout()\\nplt.show()\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-26",
   "source": "## 4. 演进路线图：信号来源的迁移\\n\\n把四个阶段画到一条时间线上，信号来源的迁移就一目了然。2021-2022 的 SFT 用人类示范，2022-2023 的 RLHF 用人类偏好，2024-2025 的 RLVR 用规则判据，2024-2026 的 Agent 后训练用环境结果。每一个新阶段没有淘汰前面的，而是叠加在它之上——Agent 模型也要先 SFT、再对齐、再 RLVR，最后才做轨迹级 RL。\\n\\n下图把各阶段的代表工作标在时间线上，信号来源写在节点下方。\\n"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-27",
   "source": "import matplotlib.pyplot as plt\\n\\nstages = [\\n    (2021.0, \\"SFT\\\\nhuman demos\\", \\"FLAN / InstructGPT-SFT\\"),\\n    (2022.6, \\"RLHF\\\\nhuman prefs\\", \\"InstructGPT / ChatGPT\\"),\\n    (2024.1, \\"RLVR\\\\nverifiable rules\\", \\"DeepSeek-R1 / DAPO\\"),\\n    (2025.3, \\"Agent RL\\\\nenvironment\\", \\"RLEF / WebRL / MiRA\\"),\\n]\\n\\nfig, ax = plt.subplots(figsize=(6.6, 3.0))\\nax.axhline(0, color=\\"gray\\", lw=1)\\nfor x, label, work in stages:\\n    ax.scatter(x, 0, s=90, zorder=3)\\n    ax.annotate(label, (x, 0), xytext=(0, 14), textcoords=\\"offset points\\",\\n                ha=\\"center\\", fontsize=9)\\n    ax.annotate(work, (x, 0), xytext=(0, -20), textcoords=\\"offset points\\",\\n                ha=\\"center\\", fontsize=7, color=\\"dimgray\\")\\nax.set_xlim(2020, 2026.8)\\nax.set_ylim(-0.4, 0.4)\\nax.axis(\\"off\\")\\nplt.title(\\"Post-training evolution: reward moves from humans to environment\\")\\nplt.tight_layout()\\nplt.show()\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-28",
   "source": "把这条时间线映射回课程地图。04 讲的 RLEF 与 Constitutional AI 是 Agent 反馈的种子；06 讲的 GRPO 与 DAPO 是 Agent 后训练的算法引擎；08 讲的深度研究是\\"环境当验证器\\"的一种具体形态。本讲把它们收拢成一条主线。后面 13 讲的 SWE 智能体、14 讲的记忆、17 讲的评测，都会反复用到轨迹、环境反馈与评测这三个词。\\n\\n一句话收束：后训练的历史，就是奖励信号从人类手里、交到验证器手里、最后交到环境手里的历史。\\n"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-29",
   "source": "## 小结\\n\\n这一讲所学的内容：\\n\\n- [ ] 预训练模型是续写器，指令对它是待续写文本；后训练把优化目标从下一个 token 改成用户意图\\n- [ ] SFT 用人类示范，损失是交叉熵，只推高示范过的行为，无法超越示教者\\n- [ ] RLHF 用人类偏好训练 RM 再优化 RM 分数，加 KL 约束防走远；RM 是代理目标，可能被 reward hacking\\n- [ ] RLVR 用规则判据当奖励，GRPO 用组内 advantage 归一化，不需要价值网络\\n- [ ] 全对组与全错组的 advantage 全为 0，这批样本不产生梯度、算力白费\\n- [ ] PPO 用价值网络做基线，GRPO 用组均值做基线，两者收敛行为不同\\n- [ ] Agent 后训练把训练单位从文本升级为轨迹，奖励来自环境执行结果，可能稀疏且延迟\\n- [ ] 轨迹级奖励的梯度方差大、收敛慢；里程碑奖励提供过程信号，方差小\\n- [ ] 环境反馈飞轮：模型输出 → 环境判定 → 信号当奖励 → 过滤成功轨迹再采样；失败任务可丢弃或改造\\n- [ ] 四个阶段信号来源：人类示范 → 人类偏好 → 规则判据 → 环境执行结果，成本下降、更难被 hack\\n"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-30",
   "source": "## 作业\\n\\n> 可以让 AI 帮忙解释思路，但不建议直接让 AI \\"做完这道题\\"。\\n"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-31",
   "source": "**作业 1：后训练四阶段分类**\\n\\n下面 6 条训练设置描述分别属于哪个阶段。补全 classify 函数，让每条描述返回 \\"SFT\\" / \\"RLHF\\" / \\"RLVR\\" / \\"Agent RL\\"。参考答案已填好，请先在草稿上自己补全一遍，再运行对照。\\n\\n小提示：先找描述里的信号来源——示范、偏好排序、规则判据、还是环境执行结果。每条描述只对应一个阶段。\\n"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-32",
   "source": "def classify(desc):\\n    \\"\\"\\"根据描述里的信号来源判断训练阶段。\\"\\"\\"\\n    if \\"轨迹\\" in desc or \\"沙箱\\" in desc:\\n        return \\"Agent RL\\"\\n    if \\"标准答案\\" in desc or \\"规则函数\\" in desc or \\"通过测试\\" in desc:\\n        return \\"RLVR\\"\\n    if \\"示范\\" in desc or \\"期望回复\\" in desc:\\n        return \\"SFT\\"\\n    return \\"RLHF\\"\\n\\n\\ndescs = [\\n    \\"用人工书写的 (指令, 期望回复) 做交叉熵微调\\",\\n    \\"让标注员对 6 个候选回复排序，训练奖励模型后做 PPO\\",\\n    \\"用规则函数判断答案与标准答案是否一致，做 GRPO\\",\\n    \\"在沙箱环境里跑完整 Agent 轨迹，用测试是否通过当奖励\\",\\n    \\"让 AI 依据原则列表自我批评与修订，用偏好做 PPO\\",\\n    \\"用规则函数判断代码是否通过测试，奖励 +1/-1\\",\\n]\\nstages = [classify(d) for d in descs]\\nprint(\\"分类结果：\\", stages)\\n\\nassert stages[0] == \\"SFT\\"\\nassert stages[1] == \\"RLHF\\"\\nassert stages[2] == \\"RLVR\\"\\nassert stages[3] == \\"Agent RL\\"\\nassert stages[4] == \\"RLHF\\"\\nassert stages[5] == \\"RLVR\\"\\nprint(\\"6 条全部判对。抓住信号来源，就能定位训练阶段。\\")\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-33",
   "source": "**作业 2：组内 advantage 与零梯度组**\\n\\nrewards 是一个组内奖励数组。补全 grpo_advantage，返回 (r - mean) / std；再补全 has_zero_signal，判断一组奖励是否产生零梯度（全对或全错）。参考答案已填好，请先在草稿上自己补全一遍，再运行对照。\\n\\n小提示：先算均值、减均值、再除标准差；std 分母加一个微小量（如 1e-9），防止全对/全错组除零。\\n"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-34",
   "source": "import numpy as np\\n\\n\\ndef grpo_advantage(rewards):\\n    \\"\\"\\"组内归一化 advantage；std 分母加微小量防除零。\\"\\"\\"\\n    mean = rewards.mean()\\n    std = rewards.std() + 1e-9\\n    return (rewards - mean) / std\\n\\n\\ndef has_zero_signal(rewards):\\n    \\"\\"\\"全对或全错时 advantage 全为 0，返回 True。\\"\\"\\"\\n    return bool(rewards.max() == rewards.min())\\n\\n\\nr2 = np.array([1.0, 1.0, -1.0, -1.0])\\na2 = grpo_advantage(r2)\\nprint(\\"r =\\", r2, \\"-> advantage =\\", np.round(a2, 3))\\nassert np.allclose(a2, [1.0, 1.0, -1.0, -1.0])\\nassert has_zero_signal(np.array([1.0, 1.0, 1.0])) is True\\nassert has_zero_signal(np.array([-1.0, -1.0])) is True\\nassert has_zero_signal(r2) is False\\nassert np.allclose(grpo_advantage(np.array([1.0, 1.0, 1.0])), 0.0)\\n\\nprint(\\"advantage 归一化与零梯度组判定都正确。\\")\\nprint(\\"全对/全错组的 advantage 全为 0，不产生任何梯度；大量这样的组就是算力浪费，DAPO 用动态采样解决它。\\")\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-35",
   "source": "**作业 3：轨迹奖励与里程碑奖励**\\n\\ntoy 三步工具调用任务：查数据库 → 过滤 → 返回结果，每步有子目标（可判定是否达成）。补全 trajectory_reward（只有三步全成功给 +1）与 milestone_reward（每步子目标达成 +0.3，最后成功再 +0.1），并打印两种奖励对同一轨迹的差异。参考答案已填好，请先在草稿上自己补全一遍，再运行对照。\\n\\n小提示：里程碑奖励其实是在给模型过程信号，这是 MiRA 的动机；先判断每步子目标是否达成，再累加。\\n"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "id": "09-36",
   "source": "def trajectory_reward(steps_ok):\\n    \\"\\"\\"steps_ok 是三个布尔值。只有全部成功给 +1，否则 -1。\\"\\"\\"\\n    return 1.0 if all(steps_ok) else -1.0\\n\\n\\ndef milestone_reward(steps_ok):\\n    \\"\\"\\"每步子目标达成 +0.3，最后成功再 +0.1（合计 1.0），保留一位小数。\\"\\"\\"\\n    per_step = 0.3 * sum(steps_ok)\\n    final = 0.1 if all(steps_ok) else 0.0\\n    return round(per_step + final, 1)\\n\\n\\nok_all = [True, True, True]\\nok_partial = [True, False, True]\\n\\nassert trajectory_reward(ok_all) == 1.0\\nassert trajectory_reward(ok_partial) == -1.0\\nassert milestone_reward(ok_all) == 1.0\\nassert milestone_reward(ok_partial) == 0.6\\n\\nprint(\\"轨迹奖励：全对\\", trajectory_reward(ok_all), \\"，部分对\\", trajectory_reward(ok_partial))\\nprint(\\"里程碑奖励：全对\\", milestone_reward(ok_all), \\"，部分对\\", milestone_reward(ok_partial))\\nprint(\\"同一轨迹，轨迹奖励只给 -1，里程碑奖励给了 +0.6 的过程信号——模型知道第一步做对了。\\")\\n",
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "09-37",
   "source": "## 参考资料\\n\\n- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022 — InstructGPT：SFT→RM→PPO 三段式 RLHF 的标杆，ChatGPT 的技术前身，1.3B 打平 175B 的参数差\\n- Christiano et al., [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741), 2017 — RLHF 的思想源头：用人类偏好学奖励模型，而非手写奖励\\n- Bai et al., [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073), 2022 — 用 AI 反馈压缩人工标注成本；04 讲已精读\\n- Wei et al., [Finetuned Language Models are Zero-Shot Learners](https://arxiv.org/abs/2109.01652), 2021 — FLAN：instruction tuning 的代表，SFT 时代泛化到新指令的证据\\n- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024 — GRPO 的提出之处，组内 advantage 的思想来源；06 讲已精读\\n- Guo et al., [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948), 2025 — RLVR 的标志：R1-Zero 无 SFT 纯 RL、thinking 自发涌现；\\"没有可 hack 的 RM\\"原文出处\\n- Yu et al., [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476), 2025 — RLVR 的工程化：全对/全错组零梯度、动态采样与 Clip-Higher；06 讲已精读\\n- Chen et al., [RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning](https://arxiv.org/abs/2410.02089), 2024 — 执行反馈 + RL 教会模型修代码，Agent 轨迹 RL 的最小原型；04 讲已精读\\n- Yao et al., [WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://arxiv.org/abs/2207.01206), 2022 — 早期 IL+RL 网页购物 Agent，Agent 后训练的对照起点\\n- Qin et al., [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789), 2023 — 工具调用训练的早期代表，把调用哪个 API 当作动作\\n- Xu et al., [WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning](https://arxiv.org/abs/2411.02337), 2024 — 网页 Agent 的自演化在线课程 RL；Llama-3.1-8B 在 WebArena-Lite 上 4.8%→42.4%\\n- Wang et al., [A Subgoal-driven Framework for Improving Long-Horizon LLM Agents](https://arxiv.org/abs/2603.19685), 2026 — MiRA：里程碑式密集奖励解决 Agent RL 的稀疏延迟奖励；Gemma3-12B 在 WebArena-Lite 上 6.4%→43.0%\\n- OpenAI, [Advancing RL for agentic systems](https://openai.com/index/advancing-rl-for-agentic-systems/), 2025 — 远程沙箱环境里做全轨迹级 RL 的前沿案例，奖励来自测试与终端成败\\n"
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
