const n=`{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-01",
   "source": "# Test-time Compute 缩放：训练完成后仍可投入的算力"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-02",
   "source": "> 第 1 讲把 Agent 定义为\\"一次调用之外还能做更多计算\\"的结构，那还是一个定性的描述。这一讲把\\"更多计算\\"变成一条可度量的缩放轴：模型训练结束后，不更新权重，只在推理时多花算力，能力还能涨多少。\\n>\\n> 我们从最简单的重复采样开始，从零实现 pass@k 无偏估计、多数投票与 best-of-n，再研究给定固定算力预算时如何按题目难度分配最划算，最后把多种技术组合成分层系统并用自动搜索寻找最优配置。全部实验使用合成数据，离线即可复现。"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-03",
   "source": "训练结束后的模型是一个固定的函数：给定一个问题，它输出一个候选答案。想让它更强，传统的路径是继续训练，但那要重跑昂贵的训练流程。这一讲关注另一条不动权重的路径——只改变推理阶段怎么花算力。这类算力统称 test-time compute，重复采样、自我修订、候选验证、多数投票都属于它。\\n\\n先体会\\"不动权重也能变强\\"。假设某道题单次做对的概率是 0.3，独立尝试 k 次，至少一次做对的概率是 $1 - (1 - 0.3)^k$。k 取 1、2、5、10 时，概率分别是 0.30、0.51、0.83、0.97。同一个模型，只是多花了推理算力，一道题的通过概率就从 0.3 涨到接近 1。\\n\\n这个最小实验说明推理阶段花算力的方式能直接改变准确率，剩下的问题是这种改变有多大、预算怎么花最划算。第一节先建立一个合成基准，把\\"一道题能否被解出\\"变成可测量的数量。"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-04",
   "source": "## 1. 训练完成后的算力：test-time compute"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-05",
   "source": "为给\\"花算力\\"一个统一视角，Snell 等人的工作把全部 test-time 方法归到两个旋钮上。第一个旋钮是提议分布（proposer），决定模型生成什么样的候选，重复采样、引导自我修订都改这里。第二个旋钮是输出后处理（verifier），拿到一批候选后怎么挑，best-of-n 挑选、多数投票都改这里。后面几节的每个演示都落在两个旋钮的某个组合里。\\n\\n先建立贯穿全讲的合成实验环境。我们用 GSM8K 风格的数学应用题做参照：一个模型在每道题上有一个真实的单次正确率 $p@1$，记为 $p_i$。题目有难有易，$p_i$ 在题目之间呈重尾分布——大多数题很难，少量题很容易。之后的每个实验都以这批题目和它们的 $p_i$ 为输入。先手动算一遍\\"多试几次\\"的收益，再生成这批题目。"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-06",
   "source": "import numpy as np\\n\\np = 0.3\\nprint(\\"单次正确率 0.3 的题，独立尝试 k 次，至少一次做对的概率\\")\\nfor k in (1, 2, 5, 10, 20):\\n    prob = 1 - (1 - p) ** k\\n    print(\\"k = %2d -> %.3f\\" % (k, prob))"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-07",
   "source": "def make_benchmark(n=200, alpha=0.25, scale=0.15, seed=42):\\n    \\"\\"\\"生成合成基准：n 道题，每道题一个单次正确率 p@1。\\n\\n    p@1 取自 Kumaraswamy(alpha, 1) 再乘 scale，密度在 0 附近\\n    正比于 p^(alpha-1)，形成重左尾：少量难题几乎解不出，\\n    少量易题很快能解。返回形状 (n,) 的数组。\\n    \\"\\"\\"\\n    rng = np.random.default_rng(seed)\\n    u = rng.random(n)\\n    return scale * u ** (1.0 / alpha)\\n\\np1 = make_benchmark()\\nprint(\\"题目数:\\", p1.shape[0])\\nprint(\\"p@1 均值: %.4f  中位数: %.4f\\" % (p1.mean(), np.median(p1)))\\nn = p1.shape[0]\\nprint(\\"最易 10%% 的平均 p@1: %.4f\\" % np.sort(p1)[-n // 10:].mean())\\nprint(\\"最难 10%% 的平均 p@1: %.4f\\" % np.sort(p1)[:n // 10].mean())"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-08",
   "source": "## 2. 重复采样：Large Language Monkeys"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-09",
   "source": "Large Language Monkeys 用一句话回答了第 1 节的问题：重复采样能把覆盖率拉得很高。对每个问题独立采样 k 个候选解，再用验证器挑一个，这就是 \`pass@k\`。论文定义了配套的两个量：覆盖率（coverage）是 k 个候选里至少有一个正确的题目占比，回答\\"模型到底能不能解出这道题\\"；精度（precision）是挑出的候选里正确的比例，回答\\"验证器能不能从干草堆里找针\\"。\\n\\n论文给出大量实证数字。SWE-bench Lite 上，DeepSeek-Coder-V2-Instruct 单次只解出 15.9% 的真实 GitHub issue，采 250 次后涨到 56%，超过当时的单次 SOTA 十三个百分点。Gemma-2B 在 CodeContests 上 pass@1 只有 0.02%，pass@10k 涨到 7.1%，提升了 300 倍。采样次数确实能换到覆盖率，而且收益形态有规律可循。先处理测量口径，再看规律。"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-10",
   "source": "但\\"采样 k 次\\"里有一个测量细节。如果每道题生成 N 个样本、其中 C 个正确，直接报告\\"有正确样本的题目占比\\"，会系统性高估 pass@k——因为 N 个样本都被用上了，等价于 pass@N。Chen 等人给出无偏估计：\\n\\n$$\\\\mathrm{pass@k} = \\\\frac{1}{P}\\\\sum_{i=1}^{P}\\\\left[1 - \\\\frac{\\\\binom{N-C_i}{k}}{\\\\binom{N}{k}}\\\\right]$$\\n\\n分子是从 N 个样本里选出 k 个、一个正确的都没选到的组合数。先手算一个例子。某道题采样 N = 10 个样本，其中 C = 3 个正确。k = 1 时，$\\\\binom{7}{1}/\\\\binom{10}{1} = 0.7$，pass@1 = 0.3，等于 C/N，符合直觉；k = 5 时，$\\\\binom{7}{5}/\\\\binom{10}{5} = 21/252 \\\\approx 0.083$，pass@5 ≈ 0.917；k = 10 时 $\\\\binom{7}{10} = 0$，pass@10 = 1。这三个值留给下面的代码验证。"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-11",
   "source": "def estimate_pass_at_k(num_correct, num_samples, k):\\n    \\"\\"\\"Chen 等人的 pass@k 无偏估计，数值稳定形式。\\n\\n    num_correct：num_samples 个样本里正确的个数；k：要估计的采样次数。\\n    等价于 1 - C(num_samples-num_correct, k) / C(num_samples, k)。\\n    返回 (0, 1] 的估计值。\\n    \\"\\"\\"\\n    if num_correct == 0:\\n        return 0.0\\n    denom = np.arange(num_samples - num_correct + 1, num_samples + 1)\\n    return 1.0 - np.prod(1.0 - k / denom)\\n\\nfor k in (1, 5, 10):\\n    print(\\"N=10, C=3, k=%2d -> pass@k = %.4f\\" % (k, estimate_pass_at_k(3, 10, k)))\\n\\nfrom math import comb\\n\\ndef estimate_pass_at_k_comb(num_correct, num_samples, k):\\n    \\"\\"\\"用组合数写出的同一估计，用于交叉验证。\\"\\"\\"\\n    if k > num_samples - num_correct:\\n        return 1.0\\n    return 1.0 - comb(num_samples - num_correct, k) / comb(num_samples, k)\\n\\nfor k in (1, 5, 10):\\n    assert abs(estimate_pass_at_k(3, 10, k) - estimate_pass_at_k_comb(3, 10, k)) < 1e-12\\nprint(\\"数值稳定形式与组合数形式完全一致\\")"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-12",
   "source": "p1 = make_benchmark()\\nN = 50\\nrng = np.random.default_rng(7)\\ncorrect = rng.binomial(N, p1)                  # 每道题 N 个样本里正确的个数\\n\\nks = np.array([1, 5, 10, 25, 50])\\ntrue_pass = np.array([(1 - (1 - p1) ** k).mean() for k in ks])\\nunbiased = np.array(\\n    [[estimate_pass_at_k(c, N, k) for c in correct] for k in ks]\\n).mean(axis=1)\\nnaive = (correct >= 1).mean()                  # 只要 N 个里出现过正确就算对\\n\\nprint(\\"k   true   unbiased   naive\\")\\nfor i, k in enumerate(ks):\\n    print(\\"%3d  %.3f    %.3f    %.3f\\" % (k, true_pass[i], unbiased[i], naive))\\nprint(\\"关键观察：k=1 时朴素估计 0.43，真值只有 0.03；\\"\\n      \\"朴素估计把 N 个样本全用上，等价于 pass@50，严重高估小 k。\\")"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-13",
   "source": "把合成基准的覆盖率画成 k 的函数，会得到一条典型的推理期缩放曲线。论文里 Llama-3-8B-Instruct 在 MATH 上的覆盖率先从 100 样本的 82.9% 涨到 10000 样本的 98.44%，拟合出 $\\\\text{coverage} = \\\\exp(a\\\\,k^b)$，其中 a = -1.33、b = -0.43。下面在合成基准上复现这条曲线，并用最小二乘拟合同一形式的幂律。"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-14",
   "source": "def coverage_at_k(p1, k):\\n    \\"\\"\\"合成基准的覆盖率：k 个样本里至少一个正确的题目占比。\\"\\"\\"\\n    return 1 - (1 - p1) ** k\\n\\nks = np.logspace(1, 2.7, 40).astype(int)       # k = 10 .. 500，对数 40 点\\ncovs = np.array([coverage_at_k(p1, k).mean() for k in ks])\\n\\ndef fit_power_law(ks, covs):\\n    \\"\\"\\"拟合 coverage = exp(a * k^b)。\\n\\n    对 log(-log covs) 关于 log ks 做线性回归：斜率即 b，截距即 log(-a)。\\n    要求 0 < covs < 1。\\n    \\"\\"\\"\\n    slope, intercept = np.polyfit(np.log(ks), np.log(-np.log(covs)), 1)\\n    return -np.exp(intercept), slope\\n\\na, b = fit_power_law(ks, covs)\\npred = np.exp(a * ks ** b)\\nrelerr = np.abs(pred - covs) / covs\\nprint(\\"拟合 coverage = exp(a * k^b): a = %.3f, b = %.3f\\" % (a, b))\\nprint(\\"平均相对误差: %.4f\\" % relerr.mean())\\n\\nimport matplotlib.pyplot as plt\\n\\nfig, ax = plt.subplots(figsize=(6, 4))\\nax.loglog(ks, covs, \\"o-\\", label=\\"simulated coverage\\")\\nax.loglog(ks, pred, \\"--\\", label=\\"fit exp(a*k^b)\\")\\nax.set_xlabel(\\"k (samples per problem)\\")\\nax.set_ylabel(\\"coverage\\")\\nax.set_title(\\"Coverage scaling on synthetic benchmark\\")\\nax.legend()\\nplt.show()\\nprint(\\"关键观察：log-log 上近似直线，平均相对误差约 %.1f%%，重现论文的幂律形态。\\"\\n      % (100 * relerr.mean()))"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-15",
   "source": "拟合出来的幂律 $\\\\exp(a\\\\,k^b)$ 有一个出人意料的来源。单独看一道题，失败率是 $(1-p_i)^k$，随 k 指数衰减。但把整个基准合起来看，聚合失败率只按幂律缓慢下降。同一个采样过程里，单题指数衰减与聚合幂律并存，原因在于 $p@1$ 在题目之间的分布是重尾的。定理 3.1 说，如果 $p@1$ 的密度在 0 附近像 $C\\\\,p^{b-1}$ 那样发散，聚合失败率就按 $k^{-b}$ 缩放；反过来，聚合呈幂律也要求 $p@1$ 分布有这种重左尾。少量几乎解不出的题把整体曲线拖成了幂律。下面同时画出单题与聚合两条失败率曲线，再移除重尾分布看幂律是否消失。"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-16",
   "source": "ks = np.logspace(0, 3, 80)                     # k = 1 .. 1000\\n\\np_hard = 0.01                                  # 单道\\"难\\"题的 p@1\\nfail_single = (1 - p_hard) ** ks               # 单题失败率：指数衰减\\n\\np1 = make_benchmark()\\nfail_agg = np.mean((1 - p1) ** ks[:, None], axis=1)\\n\\ndef loglog_r2(ks, fail):\\n    \\"\\"\\"log(fail) 关于 log(ks) 线性回归的 R^2，衡量幂律拟合程度。\\"\\"\\"\\n    x = np.log(ks)\\n    y = np.log(fail)\\n    slope, intercept = np.polyfit(x, y, 1)\\n    pred = slope * x + intercept\\n    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)\\n\\nprint(\\"单题失败率 k=1 -> k=1000 下降 %.0f 倍\\" % (fail_single[0] / fail_single[-1]))\\nprint(\\"聚合失败率 k=1 -> k=1000 下降 %.1f 倍\\" % (fail_agg[0] / fail_agg[-1]))\\nprint(\\"聚合失败率 log-log 拟合 R^2: %.4f\\" % loglog_r2(ks, fail_agg))\\n\\nrng = np.random.default_rng(1)\\np_unif = rng.uniform(0.2, 0.4, 200)            # 无重左尾的 p@1 分布\\nfail_unif = np.mean((1 - p_unif) ** ks[:, None], axis=1)\\nprint(\\"无重尾(均匀 0.2~0.4)聚合失败率 R^2: %.4f\\" % loglog_r2(ks, fail_unif))\\n\\nfig, axes = plt.subplots(1, 2, figsize=(11, 4))\\naxes[0].loglog(ks, fail_single, label=\\"single problem (exp)\\")\\naxes[0].loglog(ks, fail_agg, \\".-\\", label=\\"aggregated (power law)\\")\\naxes[0].set_xlabel(\\"k\\")\\naxes[0].set_ylabel(\\"failure rate\\")\\naxes[0].set_title(\\"Exponential vs power law\\")\\naxes[0].legend()\\naxes[1].loglog(ks, fail_agg, \\".-\\", label=\\"heavy tail p@1\\")\\naxes[1].loglog(ks, fail_unif, \\".-\\", label=\\"uniform p@1\\")\\naxes[1].set_xlabel(\\"k\\")\\naxes[1].set_ylabel(\\"failure rate\\")\\naxes[1].set_title(\\"Heavy tail makes the power law\\")\\naxes[1].legend()\\nplt.tight_layout()\\nplt.show()\\nprint(\\"关键观察：单题失败率陡降，聚合失败率在 log-log 上近似直线；\\"\\n      \\"去掉重尾分布后直线形态消失。\\")"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-17",
   "source": "## 3. 从采样到投票：self-consistency 与 best-of-n"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-18",
   "source": "覆盖率回答\\"模型能不能解出这道题\\"，最终答对多少还取决于怎么从候选里挑一个，也就是 precision。最简单的选择器有两个。第一个是多数投票：对每个问题生成 k 个候选，取出现次数最多的答案，这是 self-consistency（Wang 等人）的做法。多数投票不需要验证器，但它要求正确解成为众数；自由形式答案几乎不重合时，正确解至少要出现两次。第二个是 best-of-n：给每个候选打分，取分数最高的。Oracle 验证器永远挑得到正确解，真实验证器会有噪声。\\n\\n论文里有一个反直觉的数字：MATH 上覆盖率涨到 95% 以上，多数投票却只从 40.50% 涨到 41.41%，reward model 加 best-of-n 也在约 100 个样本处封顶。覆盖率与真实成功率之间的 gap 随样本数扩大。下面的模拟重现这三条曲线：候选里有多类错误答案，错误类的扎堆会带偏多数投票。"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-19",
   "source": "W = 50                                       # 错误答案类别数\\npool = 10000                                 # 每道题预生成的样本池\\nP = p1.shape[0]\\nrng = np.random.default_rng(5)\\n\\ncorrect = rng.random((P, pool)) < p1[:, None]       # 每个样本是否正确\\nwrong = rng.integers(1, W + 1, size=(P, pool))      # 错误样本的类别\\nA = np.where(correct, 0, wrong)                     # 0 表示正确类\\n\\ndef majority_accuracy(A, k):\\n    \\"\\"\\"用前 k 个样本做多数投票，返回基准上的答对率。\\n\\n    每道题统计各类别出现次数，正确类 0 严格多于其他类才算对。\\n    \\"\\"\\"\\n    hits = 0\\n    for row in A:\\n        counts = np.bincount(row[:k], minlength=W + 1)\\n        if counts[0] > counts[1:].max():\\n            hits += 1\\n    return hits / A.shape[0]\\n\\nks = np.unique(np.logspace(0, np.log10(5000), 40).astype(int))\\ncoverage = np.array([coverage_at_k(p1, k).mean() for k in ks])\\nmaj = np.array([majority_accuracy(A, k) for k in ks])\\nprecision = 0.55                             # 固定精度的验证器\\nweak_bon = coverage * precision              # best-of-n = coverage x precision\\n\\nfor k, c, m, w in zip(ks[::5], coverage[::5], maj[::5], weak_bon[::5]):\\n    print(\\"k=%5d  coverage=%.3f  majority=%.3f  weak-bo=%.3f\\" % (k, c, m, w))\\n\\nfig, ax = plt.subplots(figsize=(7, 4))\\nax.semilogx(ks, coverage, label=\\"coverage (oracle)\\")\\nax.semilogx(ks, weak_bon, label=\\"best-of-n, precision 0.55\\")\\nax.semilogx(ks, maj, label=\\"majority vote\\")\\nax.set_xlabel(\\"k (samples per problem)\\")\\nax.set_ylabel(\\"accuracy\\")\\nax.set_title(\\"Coverage vs realized accuracy\\")\\nax.legend()\\nplt.show()"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-20",
   "source": "def gap_at(k_target, xs, ys):\\n    \\"\\"\\"在 log-spaced 的 ks 里找最接近 k_target 的位置，返回两条曲线的差。\\"\\"\\"\\n    i = int(np.argmin(np.abs(ks - k_target)))\\n    return xs[i] - ys[i]\\n\\nprint(\\"k=100  时 coverage - majority = %.3f\\" % gap_at(100, coverage, maj))\\nprint(\\"k=5000 时 coverage - majority = %.3f\\" % gap_at(5000, coverage, maj))\\nprint(\\"k=5000 时：coverage %.3f，weak best-of-n %.3f，majority %.3f\\"\\n      % (coverage[-1], weak_bon[-1], maj[-1]))\\nprint(\\"关键观察：coverage 逼近 1，majority 明显落后，gap 随 k 扩大；\\"\\n      \\"best-of-n 的收益 = coverage x precision，验证器的 precision 决定上限。\\")"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-21",
   "source": "## 4. Compute-optimal 缩放：Snell 定律"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-22",
   "source": "重复采样把所有预算花在同一个选择器上。Snell 等人的问题更进一步：给定一笔固定的推理算力预算，不同难度的题目应该用不同方法。容易的题适合顺序修订（revision）——先给一个草稿，再顺着错误逐步修改；难的题适合并行重采样——让多个独立候选互相竞争。把预算按题目难度分配，就是 compute-optimal 缩放，论文报告同等精度下比均匀 best-of-n 少花约 4 倍算力。\\n\\n难度的划分需要一个可操作的标准。论文用 base LLM 对每道题采 2048 个样本估出 pass@1，按 5 分位数切成 5 个难度档。我们沿用这个做法：把合成基准按 $p_i$ 升序排好，均分成 5 档，档 1 最难、档 5 最易。下面用一套简化的修订模型模拟\\"并行条数 × 顺序步数\\"的预算拆分，观察各档的最优拆分方式。"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-23",
   "source": "def revision_gain(p, thresh=0.03, mult=3.0, cap=0.9):\\n    \\"\\"\\"修订的有效性：容易题能修，难题修不动。\\n\\n    修订模型只在容易的修正上学过，对 p@1 很低的难题，\\n    一连串修订往往在原错误上打转，不产生新信息。\\n    \\"\\"\\"\\n    gain = np.where(p >= thresh, mult * p, 0.0)\\n    return np.minimum(gain, cap)\\n\\ndef chain_success(p, s):\\n    \\"\\"\\"长度为 s 的顺序修订链的通过概率。\\n\\n    第一步以 p 做对；失败后每步以增益 revision_gain(p) 修正。\\n    \\"\\"\\"\\n    gain = revision_gain(p)\\n    return 1 - (1 - p) * (1 - gain) ** (s - 1)\\n\\ndef budget_split_success(p, t, budget=16):\\n    \\"\\"\\"把预算拆成并行与顺序：N_seq = budget**t 个顺序步、N_par 条并行链。\\n\\n    总预算 = N_par * N_seq = budget。返回任一链成功的通过率。\\n    \\"\\"\\"\\n    n_seq = max(1, int(round(budget ** t)))\\n    n_par = budget // n_seq\\n    single = chain_success(p, n_seq)\\n    return 1 - (1 - single) ** n_par\\n\\norder = np.argsort(p1)                       # 按 p@1 升序，最难在前\\nbins = np.array_split(order, 5)              # 切成 5 个难度档，档 1 最难\\nbin_p = [p1[i] for i in bins]                # 每档的 p@1 数组\\nts = np.linspace(0, 1, 17)                   # t=0 全并行，t=1 全顺序\\n\\nbest_t = []\\nfor q in bin_p:\\n    vals = [budget_split_success(q, t).mean() for t in ts]\\n    best_t.append(ts[int(np.argmax(vals))])\\nfor bi, t in enumerate(best_t):\\n    print(\\"bin %d 最优 t = %.2f\\" % (bi + 1, t))\\nprint(\\"趋势：最难题偏并行(t->0)，易题偏顺序(t->1)\\")"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-24",
   "source": "success = np.array([[budget_split_success(q, t).mean() for t in ts] for q in bin_p])\\nrel = success / success.max(axis=1, keepdims=True)     # 每档按自身最优归一化\\n\\nfig, axes = plt.subplots(1, 2, figsize=(12, 4))\\nim = axes[0].imshow(rel, aspect=\\"auto\\", origin=\\"lower\\", cmap=\\"viridis\\",\\n                    extent=[1, 5, ts[0], ts[-1]])\\nfor bi, t in enumerate(best_t):\\n    axes[0].plot(bi + 1, t, \\"o\\", ms=6, mfc=\\"white\\", mec=\\"k\\")\\naxes[0].set_xlabel(\\"difficulty bin (1 = hardest)\\")\\naxes[0].set_ylabel(\\"t (1 = all sequential)\\")\\naxes[0].set_title(\\"Optimal budget split (per-bin normalized)\\")\\nplt.colorbar(im, ax=axes[0])\\n\\nfor label, j in [(\\"hardest\\", 0), (\\"middle\\", 2), (\\"easiest\\", 4)]:\\n    axes[1].plot(ts, success[j], label=label)\\naxes[1].set_xlabel(\\"t (1 = all sequential)\\")\\naxes[1].set_ylabel(\\"success rate\\")\\naxes[1].set_title(\\"Success vs split, by difficulty\\")\\naxes[1].legend()\\nplt.tight_layout()\\nplt.show()\\n\\ntotal = [budget_split_success(p1, t).mean() for t in ts]\\nt_unif = ts[int(np.argmax(total))]\\nacc_co = np.mean([budget_split_success(q, bt).mean()\\n                  for q, bt in zip(bin_p, best_t)])\\nprint(\\"统一分配（所有题共用一个 t）：最佳 t=%.2f，成功率 %.3f\\" % (t_unif, np.max(total)))\\nprint(\\"compute-optimal（每档一个 t）：成功率 %.3f\\" % acc_co)\\n\\ndef uniform_succ(budget):\\n    \\"\\"\\"统一分配下，给定预算能达到的最佳成功率。\\"\\"\\"\\n    return max(budget_split_success(p1, t, budget=budget).mean() for t in ts)\\n\\nfor b in (16, 24, 32):\\n    print(\\"统一分配预算 %3d -> 成功率 %.3f\\" % (b, uniform_succ(b)))\\nprint(\\"关键观察：同样 16 单位预算，compute-optimal 达到 %.3f；\\"\\n      \\"统一分配要到 24 单位预算才追平，约 1.5 倍算力差距。\\" % acc_co)"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-25",
   "source": "还有一个问题：test-time compute 与训练算力之间如何兑换。Snell 的答案是它取决于推理负载比例 $R = D_{infer}/D_{train}$。训练算力约 $6ND_{train}$，推理算力约 $2ND_{infer}$。把模型参数放大 M 倍，总算力变成 M 倍的训练加推理；用小模型加 test-time compute 匹配这笔总预算时，可用的采样次数变为\\n\\n$$S = M + 3\\\\,\\\\frac{D_{train}}{D_{infer}}(M-1)$$\\n\\n取 M = 14，R 取 0.16（自举式训练，推理负载低）、0.79（典型负载）、22（大规模部署，推理负载高）时，S 分别约 258、63、16。下面画小模型的采样曲线，把\\"14 倍大模型贪心解码\\"的精度当作一条水平线，看三个 R 下的可用样本数能把小模型推到线上方还是下方。"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-26",
   "source": "p_small = make_benchmark(alpha=0.8, scale=0.5, seed=11)    # 一组\\"小模型\\"题目\\nM = 14                                        # 大模型是 14 倍参数\\nm = 20                                        # 大模型 greedy 等价于小模型采 20 次\\nbig_acc = (1 - (1 - p_small) ** m).mean()\\n\\nS = np.arange(1, 301)\\nsmall_acc = np.array([(1 - (1 - p_small) ** s).mean() for s in S])\\n\\nfig, ax = plt.subplots(figsize=(7, 4))\\nax.plot(S, small_acc, label=\\"small model + repeated sampling\\")\\nax.axhline(big_acc, color=\\"gray\\", ls=\\"--\\", label=\\"14x big model (greedy)\\")\\nfor R in (0.16, 0.79, 22.0):\\n    s_star = M + 3 * (M - 1) / R              # FLOPs-matched 的可用样本数\\n    acc = (1 - (1 - p_small) ** s_star).mean()\\n    winner = \\"test-time\\" if acc > big_acc else \\"pretrain\\"\\n    ax.plot(s_star, acc, \\"o\\")\\n    ax.annotate(\\"R=%.2f -> %s\\" % (R, winner), (s_star, acc),\\n                textcoords=\\"offset points\\", xytext=(0, 8), fontsize=9)\\nax.set_xscale(\\"log\\")\\nax.set_xlabel(\\"samples per problem\\")\\nax.set_ylabel(\\"accuracy\\")\\nax.set_title(\\"Small model + test-time vs 14x pretrain\\")\\nax.legend()\\nplt.show()\\n\\nprint(\\"R=0.16（自举）-> 可用样本 %.0f，test-time 占优\\" % (M + 3 * 13 / 0.16))\\nprint(\\"R=0.79（典型）-> 可用样本 %.0f，test-time 占优\\" % (M + 3 * 13 / 0.79))\\nprint(\\"R=22.0（部署）-> 可用样本 %.0f，pretraining 占优\\" % (M + 3 * 13 / 22.0))\\nprint(\\"关键观察：R 小（自举/推理负载低）时 test-time 更划算，R 大（部署）时扩训练更划算。\\")"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-27",
   "source": "## 5. 方法组合与架构搜索：Archon"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-28",
   "source": "前面几节的方法各有适用场景：重复采样对覆盖率有效，顺序修订对易题有效，best-of-n 依赖验证器。Archon 观察到没有哪个单一技术在所有任务上最优，于是把这些技术组织成分层的 LLM 系统，再用自动搜索找出最优组合。系统的每个组件都是 text-to-text 操作，没有可训练的权重：Generator 生成候选，Fuser 把多个候选合并，Ranker 两两比较排序，Critic 先列优缺点再交给排序与融合，Verifier 先给推理再给判定，Unit-Test 生成器产出测试语句并打分。\\n\\n结构有若干硬规则：Generator 只能放在第一层，每层只放一类组件，Critic 必须在 Ranker 或 Fuser 之前，最后一层输出第一个字符串。去掉无效配置后，搜索空间有 9576 个配置，论文用贝叶斯优化在约两成数据上搜索，最佳架构平均超过当时 frontier 模型 15.1%。下面在一个 96 配置的简化空间里，从零实现网格搜索与随机搜索，观察最优架构是否随任务变化。"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-29",
   "source": "configs = []\\nfor top_k in (2, 4, 6, 8):\\n    for layers in (1, 2, 3):\\n        for critic in (0, 1):\\n            for verifier in (0, 1):\\n                for unit_test in (0, 1):\\n                    configs.append(dict(top_k=top_k, layers=layers, critic=critic,\\n                                        verifier=verifier, unit_test=unit_test))\\nprint(\\"配置总数:\\", len(configs))\\n\\nrng_eval = np.random.default_rng(5)\\nscore_table = {}\\n\\ndef architecture_accuracy(cfg, task):\\n    \\"\\"\\"合成评估器：读缓存分数，没有则计算并缓存。\\n\\n    cfg：配置字典；task：'instruct' 或 'code'。\\n    融合层对指令跟随有用、单测对代码有用、验证器对推理有用。\\n    \\"\\"\\"\\n    key = (task, cfg[\\"top_k\\"], cfg[\\"layers\\"], cfg[\\"critic\\"],\\n           cfg[\\"verifier\\"], cfg[\\"unit_test\\"])\\n    if key in score_table:\\n        return score_table[key]\\n    acc = 0.55 if task == \\"instruct\\" else 0.30\\n    acc += 0.008 * cfg[\\"top_k\\"]\\n    acc += 0.030 * cfg[\\"layers\\"] * (1.0 if task == \\"instruct\\" else 0.35)\\n    acc += 0.035 * cfg[\\"critic\\"] * (1.0 if task == \\"instruct\\" else 0.10)\\n    acc += 0.025 * cfg[\\"verifier\\"] * (1.0 if task == \\"instruct\\" else -0.20)\\n    acc += 0.090 * cfg[\\"unit_test\\"] if task == \\"code\\" else -0.025 * cfg[\\"unit_test\\"]\\n    acc += rng_eval.normal(0, 0.012)          # 评估噪声\\n    score_table[key] = min(acc, 1.0)\\n    return score_table[key]\\n\\ndef grid_best(configs, task):\\n    \\"\\"\\"遍历全部配置，返回（最高分，配置）。\\"\\"\\"\\n    scores = [architecture_accuracy(c, task) for c in configs]\\n    i = int(np.argmax(scores))\\n    return scores[i], configs[i]\\n\\ndef random_search_trace(configs, task, rng, n=40):\\n    \\"\\"\\"不放回随机采 n 个配置，记录每一步的当前最优分。\\"\\"\\"\\n    order = rng.permutation(len(configs))\\n    best = []\\n    for step in range(min(n, len(configs))):\\n        sc = architecture_accuracy(configs[order[step]], task)\\n        best.append(sc if step == 0 else max(best[-1], sc))\\n    return np.array(best)\\n\\nfor task in (\\"instruct\\", \\"code\\"):\\n    score = grid_best(configs, task)[0]\\n    trace = random_search_trace(configs, task, np.random.default_rng(9))\\n    print(\\"%s: grid 最优 %.3f，随机 40 次评估达到 %.3f（%.1f%%）\\"\\n          % (task, score, trace[-1], 100 * trace[-1] / score))"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-30",
   "source": "fig, ax = plt.subplots(figsize=(7, 4))\\nfor task, color in [(\\"instruct\\", \\"C0\\"), (\\"code\\", \\"C3\\")]:\\n    grid_score = grid_best(configs, task)[0]\\n    trace = random_search_trace(configs, task, np.random.default_rng(9))\\n    ax.plot(range(1, len(trace) + 1), trace, \\"-o\\", ms=3, color=color,\\n            label=\\"%s random\\" % task)\\n    ax.axhline(grid_score, color=color, ls=\\"--\\", label=\\"%s grid best\\" % task)\\nax.set_xlabel(\\"number of evaluations\\")\\nax.set_ylabel(\\"best accuracy found\\")\\nax.set_title(\\"Architecture search: random vs grid\\")\\nax.legend(fontsize=8)\\nplt.show()\\n\\ninstruct_cfg = grid_best(configs, \\"instruct\\")[1]\\ncode_cfg = grid_best(configs, \\"code\\")[1]\\nprint(\\"指令跟随任务的最优配置:\\", instruct_cfg)\\nprint(\\"代码任务的最优配置:\\", code_cfg)\\nprint(\\"关键观察：两个任务的最优配置不同，没有通吃的单一架构，\\"\\n      \\"这正是自动搜索的动机。\\")"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-31",
   "source": "import sys, os\\n_root = os.path.abspath(os.getcwd())\\nwhile not os.path.exists(os.path.join(_root, 'llm_client.py')):\\n    _root = os.path.dirname(_root)\\n    if _root == os.path.dirname(_root):\\n        break\\nif _root not in sys.path:\\n    sys.path.insert(0, _root)\\nfrom llm_client import get_llm\\nclient = get_llm()\\nimport os\\nmock_requested = bool(os.environ.get(\\"LLM_MOCK\\")\\n                      or os.environ.get(\\"AGENT_LLM_MOCK\\"))\\neffective_mock = client.is_mock or mock_requested\\nprint(\\"LLM 客户端就绪，有效模式:\\",\\n      \\"mock（确定性占位）\\" if effective_mock else \\"real API\\")"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "id": "code-32",
   "source": "import re\\n\\ndef generate_candidates(client, question, n=3):\\n    \\"\\"\\"用 n 种提示模板让模型各自作答，返回候选答案字符串。\\"\\"\\"\\n    templates = [\\n        \\"请计算 %s，只输出最终答案。\\",\\n        \\"%s 等于多少？只输出数字。\\",\\n        \\"求解：%s。只输出最终答案。\\",\\n    ]\\n    candidates = []\\n    for i in range(n):\\n        reply = client.chat([{\\"role\\": \\"user\\", \\"content\\": templates[i] % question}])\\n        nums = re.findall(r\\"\\\\d+\\", reply)\\n        candidates.append(nums[0] if nums else \\"N/A\\")\\n    return candidates\\n\\ndef majority(answers):\\n    \\"\\"\\"多数投票：返回出现次数最多的答案。\\"\\"\\"\\n    votes = {}\\n    for a in answers:\\n        votes[a] = votes.get(a, 0) + 1\\n    return max(votes, key=votes.get)\\n\\nquestion = \\"15 加 27\\"\\nmock_requested = bool(os.environ.get(\\"LLM_MOCK\\")\\n                      or os.environ.get(\\"AGENT_LLM_MOCK\\"))\\nif mock_requested or not getattr(client, \\"api_key\\", \\"\\"):\\n    client = get_llm(force_mock=True)        # 离线验证一律走 mock，避免真实调用\\ntry:\\n    answers = generate_candidates(client, question, n=3)\\nexcept RuntimeError:\\n    client = get_llm(force_mock=True)\\n    answers = generate_candidates(client, question, n=3)\\n    print(\\"未检测到可用 API key，已切换到 mock 模式\\")\\n\\nprint(\\"三个候选答案:\\", answers)\\nprint(\\"多数投票结果:\\", majority(answers), \\"（正确答案 42）\\")\\nif client.is_mock:\\n    print(\\"说明：mock 模式下回复为确定性占位；真实 API 下温度采样会给候选带来多样性。\\")"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-33",
   "source": "## 小结\\n\\n这一讲沿着\\"给一笔推理算力预算怎么花\\"的决策链走了一遍：\\n\\n- [ ] 训练完成后仍可花推理算力换取能力，所有 test-time 方法都落在 proposer 与 verifier 两个旋钮上\\n- [ ] 重复采样把覆盖率推高，pass@k 需要无偏估计，直接统计会系统性高估\\n- [ ] 覆盖率呈幂律 $\\\\exp(a\\\\,k^b)$，用 log-log 线性回归即可拟合\\n- [ ] 单道题失败率指数衰减，聚合到整个基准变成幂律；去掉重尾分布，幂律消失\\n- [ ] 多数投票与 best-of-n 的真实收益由 precision 决定，无可靠验证器时收益封顶\\n- [ ] 固定预算下按题目难度分配并行与顺序算力，compute-optimal 优于统一分配\\n- [ ] 小模型加 test-time compute 与更大模型谁的算力更值，由推理负载比例 R 决定\\n- [ ] 多种 test-time 技术可组合成分层系统，用自动搜索替代手工搭架构"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-34",
   "source": "## 作业\\n\\n> 可以让 AI 帮忙解释思路，但不建议直接让 AI \\"做完这道题\\"。"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-35",
   "source": "**作业 1：pass@k 无偏估计**\\n\\n补全下面的函数，完成 pass@k 的数值稳定形式。已给出 N = 20、C = 5 的验证，预期 pass@5 ≈ 0.806，pass@20 = 1。\\n\\n\`\`\`python\\ndef estimate_pass_at_k(num_correct, num_samples, k):\\n    if num_correct == 0:\\n        return 0.0\\n    denom = np.arange(num_samples - num_correct + 1, num_samples + 1)\\n    return 1.0 - np.prod(____)      # 补全\\n\\nassert abs(estimate_pass_at_k(5, 20, 5) - 0.8063) < 1e-3\\nassert abs(estimate_pass_at_k(5, 20, 20) - 1.0) < 1e-12\\nprint(\\"无偏估计实现正确：k 越接近 N，朴素估计的高估越明显。\\")\\n\`\`\`\\n\\n小提示：\`np.prod(1 - k / denom)\` 就是组合数之比 $\\\\binom{N-C}{k}/\\\\binom{N}{k}$，二者在 k > N-C 时都等于 0。"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-36",
   "source": "**作业 2：从重尾分布推出幂律指数**\\n\\n给定合成 p@1（Kumaraswamy(0.4, 1) 乘 0.15），模拟聚合覆盖率，在 log-log 上拟合出指数 b，并与左尾理论 -alpha 对照。\\n\\n\`\`\`python\\np = make_benchmark(alpha=0.4, scale=0.15, seed=3)\\nks = np.logspace(1, 2.7, 40).astype(int)\\ncovs = np.array([(1 - (1 - p) ** k).mean() for k in ks])\\nslope, intercept = np.polyfit(np.log(ks), np.log(-np.log(covs)), 1)\\nb = ____                                    # 补全：斜率即 b\\n\\nassert abs(b - (-0.4)) < 0.25\\nprint(\\"聚合幂律指数 b 由 p@1 分布左尾指数决定：理论值 -0.4，拟合值 %.3f。\\" % b)\\n\`\`\`\\n\\n小提示：Kumaraswamy(alpha, 1) 的密度在 0 附近正比于 $p^{\\\\alpha-1}$，定理 3.1 给出聚合失败率按 $k^{-\\\\alpha}$ 缩放。有限 k 区间上拟合值会略偏离渐近值，断言因此放宽到 0.25。"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-37",
   "source": "**作业 3：多数投票的 plateau**\\n\\n合成数据上，多数投票需要正确类严格成为众数。补全条件，验证 majority 的提升明显小于 coverage。\\n\\n\`\`\`python\\nW = 50\\npool = 10000\\np = make_benchmark(seed=5)\\nP = p.shape[0]\\nrng = np.random.default_rng(5)\\ncorrect = rng.random((P, pool)) < p[:, None]\\nwrong = rng.integers(1, W + 1, size=(P, pool))\\nA = np.where(correct, 0, wrong)\\n\\ndef majority_accuracy(A, k):\\n    hits = 0\\n    for row in A:\\n        counts = np.bincount(row[:k], minlength=W + 1)\\n        if counts[0] > ____:                 # 补全：正确类严格多于其他类\\n            hits += 1\\n    return hits / A.shape[0]\\n\\nc_100 = (1 - (1 - p) ** 100).mean()\\nc_5000 = (1 - (1 - p) ** 5000).mean()\\nm_100 = majority_accuracy(A, 100)\\nm_5000 = majority_accuracy(A, 5000)\\nassert (m_5000 - m_100) < (c_5000 - c_100)\\nprint(\\"k 从 100 到 5000：coverage 提升 %.3f，majority 只提升 %.3f——\\"\\n      \\"多数投票对难题无能为力，gap 随样本数扩大。\\" % (c_5000 - c_100, m_5000 - m_100))\\n\`\`\`\\n\\n小提示：每类错误答案被\\"扎堆\\"时，正确类必须严格多于任何一类才能成为众数，即 \`counts[0] > counts[1:].max()\`。"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "id": "md-38",
   "source": "## 参考资料\\n\\n- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787)（Brown 等人，2024）— 重复采样覆盖率的实证与幂律，coverage/precision 两轴的出处\\n- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)（Snell 等人，2024）— compute-optimal 缩放：按难度分配 test-time 算力，revision 加 PRM 搜索\\n- [Archon: An Architecture Search Framework for Inference-Time Techniques](https://arxiv.org/abs/2409.15254)（Saad-Falcon 等人，ICML 2025）— 分层 LLM 系统加贝叶斯优化架构搜索；注意正确 ID 是 2409.15254\\n- [How Do Large Language Monkeys Get Their Power (Laws)?](https://arxiv.org/abs/2502.17578)（Schaeffer 等人，ICML 2025）— 单题指数衰减加重尾 p@1 分布推出聚合幂律；注意正确 ID 是 2502.17578\\n- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)（Chen 等人，2021）— pass@k 无偏估计器的出处\\n- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)（Wang 等人，2023）— 多数投票 / self-consistency，与 precision 的讨论直接相关\\n- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)（Cobbe 等人，2021）— GSM8K 与最早验证器训练\\n- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)（Lightman 等人，2023）— PRM 训练与 MATH 难度分档，Snell 论文沿用其数据划分\\n- [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814)（Li 等人，2022）— 大规模重复采样的先驱，CodeContests 数据集出处\\n- [Beyond Chinchilla-Optimal: Accounting for Inference in LM Scaling Laws](https://arxiv.org/abs/2401.00448)（Sardana 与 Frankle，2023）— 把推理 FLOPs 计入缩放定律，Snell 论文 FLOPs 公式的依据"
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
