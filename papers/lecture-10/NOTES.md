English: [NOTES.en.md](NOTES.en.md)

# Lecture 10 — 软件工程智能体（Agentic Frameworks for Software Engineering）研读笔记

> 本文件是 CS329A 第 10 讲的论文研读笔记，是编写对应 notebook 的素材。
> 来源：https://cs329a.stanford.edu/（Autumn 2025 课程大纲）
>
> 对应 OUTLINE 章节：10-swe-agents.ipynb（代码任务中的 test-time compute：CodeMonkeys / 让 LLM 写高效内核：KernelBench / Agent-System 接口设计 / SWE-Agent 循环的构建）。

## 课程主题

这一讲要解决的核心问题：**如何把 Agent 用到真实的软件工程任务上，并度量它的价值**。前几讲已经把 Agent 循环、验证器、工具使用讲清楚了，但那些大多在"合成任务"（数学、搜索、问答）上。软件工程是 Agent 最有经济价值的真实场景：代码库巨大（SWE-bench 一个仓库几百万 token）、反馈来源丰富（编译错误、测试失败、profiler）、成功标准多元（既要正确也要快）。本讲回答三个递进的问题：一是 test-time compute 在代码任务里怎么缩放（CodeMonkeys 给出"串行修复循环 + 并行采样 + 选择"的完整配方）；二是评测标准怎么设计——当"对"不够、还要"快"时怎么办（KernelBench 引入 fast_p 指标）；三是 Agent 和底层系统之间要靠什么样的"接口"才能对齐（Agent-System Interface）。

为什么安排在这个位置：L10 是 Part 3（Agent Engineering）的第一讲，前面 L1-L9 建立了"循环 + 验证 + 工具 + 搜索 + 后训练"的底层能力，本讲把这些能力组装成一个能解决真实 GitHub issue 的完整系统，并示范"为评测而设计的任务"如何反过来牵引系统设计。L11（记忆）和 L14（评测与长程任务）紧接着它，分别补足记忆系统和更系统化的评测方法。

教学上本讲用一个"把工程问题切成三段"的主线：**找上下文 → 生成候选 → 选择答案**（context → generation → selection）。这个三段切法（CodeMonkeys 明确提出）是 SWE-agent 系统设计的通用脚手架，也贯穿 SWE-Agent、Agentless 等所有主流方法。

## 论文精读

### 论文 1：CodeMonkeys: Scaling Test-Time Compute for Software Engineering（arXiv:2501.14723，codemonkeys.pdf）
- **核心思想**：把 test-time compute 的缩放（L2 的主题）真正用到 SWE-bench Verified 上，并回答"如果一个系统从一开始就是为了吃 test-time compute 设计的，它该怎么搭"。答案是把一个 issue 的解决拆成三个子任务：找相关文件（context）、生成候选编辑（generation）、从候选里挑对的（selection），然后分别缩放：**串行缩放** = 让模型和测试脚本一起迭代修复（每增加一次迭代，就是多花一串 compute）；**并行缩放** = 每个 issue 独立采样多条修复轨迹。关键洞察：因为并行采样了很多条轨迹，找上下文的成本可以在下游样本间摊薄，于是可以用最朴素的方法——让一个便宜模型（Qwen2.5-Coder-32B，本地跑）把代码库里每个文件都读一遍来判断相关性。CodeMonkeys 在 SWE-bench Verified 上拿到 57.4%，花约 2300 美元。
- **关键公式/算法**：
  - 三段式评测指标：Context 用 recall（所有需要改的文件都被放进上下文的题目占比，128k 上限下 92.6%）；Generation 用 coverage（至少一条采样编辑正确的题目占比，最优 69.8%）；Selection 用最终 score（提交的编辑被官方测试判对的题目占比，57.4%）。三者的关系是：coverage 是"生成的天花板"（oracle selection 的上界），selection 决定能从天花板收回来多少。
  - 上下文流水线：Qwen2.5-Coder 逐文件扫描判断相关（只扫 Python 文件、排除 test 目录，平均处理 294 万 token/题）→ 对相关文件生成摘要 → 用 Claude 排序（temperature 0，重复 3 次取平均 rank）→ 装进最多 128k token 的上下文窗口。平均上下文 74,570 token，相对全量压缩 50.5x；扫描+排序总共只占成本 15%（因为摊薄到了 10 条下游轨迹上）。
  - 生成阶段 = 两个背靠背的**状态机**（state machine 抽象来自 Moatless Tools）：①Testing State Machine 先独立迭代出一个能复现 issue 的测试脚本（standalone Python，exit 0 表示修复、exit 2 表示未修复）；②Editing State Machine 以该测试为种子，用 aider 风格 diff 迭代生成编辑，且允许同时改测试——这就是论文强调的"two-sided debugging"：测试要在未编辑代码库上失败、在编辑后通过。每次迭代都把执行输出喂回模型。每 issue 跑 10 对状态机，每台最多 8 次迭代，temperature 0.5。
  - 选择阶段（4 种方法对比）：①用测试多数投票（把 10 个生成的测试各跑在 10 条编辑上，选通过最多的）；②纯模型选择；③模型选择但先按测试通过数过滤出 top-3；④**选择状态机** + top-3 过滤：让模型写"能区分候选"的新测试，跑了再决定选谁或再写测试，最多 10 次迭代。最优的是④，得 57.4%。
  - 关键数字：随机选择只有 45.8%，oracle（=coverage）69.8%——选择方法收回了随机与 oracle 之间大约一半的差距。Barrel of Monkeys（把 CodeMonkeys 的编辑和 SWE-bench Verified 排行榜 top-4 提交合并成 5 样本池，coverage 80.8%）用选择状态机做到 66.2%，超过池子里最好的单个方法（Blackbox 62.8%），只比 o3 报告的 71.7% 低 5.5pt。
  - 成本拆解（表 1）：总 2291.90 美元。生成编辑占 59.6%、生成测试 19.2%、相关性扫描 14.6%、选择 5.8%、排序 0.9%。
  - 缩放规律（图 5）：coverage 随"状态机数 × 每台迭代数"增长，前几个迭代收益最大；**相同总预算下，串行/并行的不同分配常得到相近的 coverage**——但两者不完全可互换：并行多采样会让选择更难，串行深轨迹则绕开选择问题，只是当模型"错误地批准"自己的编辑时，增加迭代上限救不回来，而并行加一条新轨迹总能"重新开始"。
  - 附录：用 DeepSeek-V3 替换 Claude 做生成/选择，在 100 题子集上拿到 Claude 成绩的 86.8%，但 API 便宜一个数量级——"便宜模型大量采样 + 强选择器"是成本敏感的选择。
- **关键实验结论**：57.4%（SWE-bench Verified，约 2300 美元/500 题）；coverage 69.8%；oracle selection 时 69.8% 超过了当时排行榜上除 o3 外的所有提交。选择阶段证明了"选得对不对"对最终分数至关重要：随机 45.8% → 简单投票 53.0% → 模型选择 52.0% → top-3+模型 55.6% → **top-3+选择状态机 57.4%**。
- **与课程主题的关系**：本讲第一块砖——把 L2 的"采样+验证"升级成能在真实 GitHub 仓库上运转的完整 agent 循环。三段式分解（context/generation/selection）是整讲的教学骨架；"让模型写测试"的强制要求同时服务了两件事：给迭代提供更细的执行反馈、给选择提供（不完美的）验证器——这正是 L3 验证器思想在代码域的落地。状态机是"SWE-Agent 循环的构建"（OUTLINE 第 4 点）最直接的可实现范本。
- **可演示的代码点**：从零搭一个迷你状态机（生成编辑+测试 → 跑测试 → 拿 exit code 反馈 → 迭代）；实现 coverage 公式并画"采样数-覆盖率"曲线；实现三类选择器（测试多数投票 / 模型选择 / 选择状态机）并在合成候选池上对比；复现"摊薄上下文成本"的成本账。

### 论文 2：KernelBench: Can LLMs Write Efficient GPU Kernels?（arXiv:2502.10517，kernelbench.pdf）
- **核心思想**：把评测从"代码对不对"推进到"代码对不对且快不快"。给 LLM 一个 PyTorch 参考实现（`Model` 类），要求它输出一个同样接口但内部换成自定义 CUDA/Triton 内核的 `ModelNew` 类，然后自动评测两条轴：功能正确（随机输入对比输出）和性能（wall-clock 加速比）。250 个任务按算子数分三级：Level 1 单算子（100）、Level 2 算子序列（100，考融合）、Level 3 完整架构（50，AlexNet/MiniGPT）。动机来自现实：FlashAttention 这类关键内核在 Transformer 提出 5 年后才出现、换硬件还要再花 2 年——如果 LLM 能自动写高效内核，价值直接兑现为推理/训练成本和能耗的下降。
- **关键公式/算法**：
  - fast_p 指标：$\text{fast}_p = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\text{correct}_i \land \text{speedup}_i > p]$，其中 $\text{speedup}_i = T_{\text{Model}}/T_{\text{ModelNew}}$。p 是加速比阈值：fast_0 就是纯正确率，fast_1 是"正确且比 PyTorch Eager 快"。把 p 当旋钮就能画整条"加速比分布"曲线，也便于将来把基线升级（如对 torch.compile）。评测细节：正确性用 5 组随机输入；性能预热 3 次、100 次计时取均值（CV<3%）；只允许同一时间一个 kernel 在 GPU 上跑。
  - 任务格式：输入是含 `get_inputs()`/`get_init_inputs()` 的 PyTorch `Model`；输出 `ModelNew` 用 `torch.utils.cpp_extension.load_inline` 内联 CUDA（论文给出完整 add/matmul 例子）。LM 要自己决定优化哪些算子、用什么技术（融合、tiling、共享内存、tensor core/wmma、PTX）。
  - one-shot 基线：单一 in-context 示例（add 算子）+ greedy 解码。所有前沿模型平均在 <20% 任务上快于 PyTorch Eager（表 1）：DeepSeek-R1 在 L2 36%、L1 12%；o1 在 L2 24%、L1 10%；其余多是个位数；对 torch.compile 基线更低。
  - 两种测试时方法（同为 10 次调用预算，表 2）：**repeated sampling**（高温度并行采样 k 条，fast_p@k = 至少一条达标）与 **iterative refinement**（多轮把生成 G + 执行反馈 E + profiler 反馈 P 喂回模型）。迭代更优（6 例中 5 例），且反馈组合 G+E+P 最强：DeepSeek-R1 在 Level 2 从 one-shot 的 36% 升到 72%（图 6），L1 12%→43%、L3 2%→18%。纯正确率（fast_0，表 9）用 E 后 R1 在 L1/L2 超 90%。重复采样的例子：DeepSeek-V3 L2 从 4%（one-shot）到 37%（k=100），但若模型对某类任务（34 个卷积变体）内在概率过低，采样再多也救不回来。
  - 失败模式归因（图 2）：执行失败（编译/越界/运行错误）vs 功能不正确（shape/value 错）。推理模型（o1、R1）执行失败明显更少（总错误 <55% vs 其他 >70%），但**所有模型在功能正确性上同样挣扎**。作者归因于 CUDA 是低资源语言——在 The Stack v1.2 里只占 0.073%。
  - 有意思的内核（第 6 节）：13x 的 diag-matmul（Claude，把对角线矩阵乘改为按行缩放，纯算法优化）；2.9x 的 GELU 融合（DeepSeek-V3，含常数折叠）；2.8x 的 cosine similarity（o1，用共享内存做规约）；2.6x 的 matmul+divide+sum+scale 融合（Sonnet）。但从未出现可用的 tensor core 指令。
  - 跨硬件（表 14/图 8-9）：one-shot 内核不跨硬件泛化，R1 在 L2 的 fast1 是 L40S 36% vs A10G 47%。提供硬件规格（TFLOPS/带宽）在上下文中帮助有限；few-shot 示范（融合/tiling/FlashAttention）反而降低整体 fast1（模型更激进 → 更多执行失败），但在正确子集里 o1 对 77% 的 GEMM 应用了 tiling。
  - 评测系统（附录 H）：三段流水线（并行生成 → CPU 上 nvcc 预编译缓存 → GPU 上单核逐一计时）；迭代实验用"GPU orchestrator + 有限状态机 + 信号量抢 GPU"，还有可视化轨迹的 UI。
- **关键实验结论**：one-shot 下最强推理模型也只匹配 PyTorch 基线 <20%；迭代反馈（E+P）把 R1 的 L2 fast1 从 36% 拉到 72%；重复采样和迭代都比 one-shot 强，但天花板由基座模型决定。benchmark 设计上"把 p 调高就变难"的特性，让它可以随模型进步持续演化。
- **与课程主题的关系**：本讲第二块砖——示范"为评测设计环境"如何驱动 agent 系统设计。KernelBench 的回环（生成→编译→执行→profiler 反馈→再生成）就是 SWE-agent 循环在"性能优化"这一目标下的特例；fast_p 展示了如何把"多目标（正确+快）"压成一个标量指标。也点出 Agent 的真实瓶颈往往不在"想得对不对"而在"数据稀缺"（CUDA 语料 0.073%）。
- **可演示的代码点**：用 numpy 实现 fast_p 并画 p 从 0 升高时的正确+加速曲线；CPU 上做一个"伪 kernel 评测"（对比逐算子实现 vs 融合实现的计时与正确性，感受融合收益）；搭一个模拟"生成→编译错误反馈→再生成"的迭代循环，观察 fast1@N 随轮次上升；模拟 repeated sampling 的覆盖率曲线。

### 论文 3：On the Need to Align Intent and Implementation in Uncertainty Quantification（arXiv:2506.03037，agent-system-interfaces.pdf）
- **⚠️ 文件问题**：`papers/lecture-10/agent-system-interfaces.pdf` 里的实际内容不是本讲预期的 "Improving Parallel Program Performance with LLM Optimizers via Agent-System Interfaces"。那个真正对应 Agent-System 接口设计的论文 arXiv 编号应为 **2410.15625**（Wei, Nie, Teixeira, Yadav, Lee, Wang, Aiken，ICML 2025，斯坦福）。本 PDF 装的是另一篇立场论文：Trivedi & Nord（Fermilab/芝加哥大学）的 *On the Need to Align Intent and Implementation in Uncertainty Quantification for Machine Learning*。**写 notebook 时若要覆盖 "Agent-System 接口设计" 一节，请重新下载 2410.15625**；以下对实际文件内容的笔记仍值得保留，因为它把 "intent 与 implementation 对齐" 这个主题做了可迁移的剖析。
- **核心思想**：立场论文，论证"量化的不确定性必须锚定在它的推理目标上"。核心病态叫 **construct drift**：不确定度是为 A 算出来的，却被拿来支撑关于 B 的结论（例如把"模型输出的方差"当作"模型不知道什么"）。作者给出一个诊断框架：先分类"估计目标"（在估计什么），再分类"不确定性构造"（用什么逻辑表达不确定），然后要求二者通过一条"认识论契约"（epistemic contract）对齐，并用三根"可信轴"来检验。
- **关键公式/算法**：
  - 估计目标分类（六种）：estimation（纯数值摘要，无承诺）、prediction（预测未来可观测值）、inference（恢复潜在参数 θ）、predictive inference（对未来观测做无模型覆盖保证，如 conformal）、indirect inference（无似然时用辅助统计量）、simulation-based inference（SBI，用模拟器替代似然，NPE/NLE/NRE）。
  - 不确定性构造家族（四种，各对应一种概率解释）：frequentist（长期重复保证，如置信区间、conformal 集）、Bayesian（信念一致更新，credible 区域）、fiducial/hybrid（Fisher，从数据生成机制导出分布）、logical probability（Keynes/Carnap，证据到假设的蕴含度）。
  - 认识论契约（表 2）：目标 → warrant（正当性）→ 构造。例：prediction → 长期覆盖 → conformal 集；parameter inference → 信念一致性 → credible 区域；SBI 参数 → 学习到的近似 → NPE 后验。
  - 三根可信轴（第 5 节）：①formal guarantees（有没有理论保证：覆盖定理、后验一致性）；②empirical reliability（实践中是否成立：SBC、posterior predictive checks、校准曲线）；③model correspondence（是否尊重领域结构：对称性、守恒律、因果）。三根轴不可互换——一个方法可能覆盖有效、benchmark 好看，却完全无视物理结构。
  - SBI checklist（把三根轴落实到模拟推断）：theory check（方法声称保证什么）、forward checks（后验经模拟器采样能否重现观测数据分布）、inverse checks（能否从模拟数据找回已知参数，检验可辨识性与校准）、degeneracy mapping（参数空间是否有观测等价方向）、global structure comprehension（联合分布 p(θ,x) 的全面检查）。
  - 典型错位案例（第 6 节，每条对应违反哪根轴）：把 deep ensemble 的跨模型方差当"epistemic uncertainty"（无校准保证，违反轴 1&2）；conformal 区间只保证 marginal coverage 但缺少领域本体（违反轴 2&3）；NRE 在快速代理模拟器上训练导致过自信后验、覆盖差（Delaunoy et al.，违反轴 1&3）；对同一检测同时报 frequentist p 值与 Bayesian credible 区域引起"矛盾"假象（warrant 混淆）；conformal 在亚群上失效（校准了覆盖却没校准解释力）。
  - 建议（第 7 节）：显式声明推理链（目标/损失/构造/warrant）、按用途选构造、同时做前向与逆向校验、不要混用构造（prediction interval ≠ credible region）、用模拟器当测试仪器、超越 i.i.d. 评测、研究模型稳定性、跨学科澄清术语。
- **关键实验结论**：这是一篇立场论文，无实验曲线；最有价值的产出是诊断清单（表 3 的 cross-cutting diagnostics：marginal coverage、conditional coverage、sharpness、OOD 行为、aleatoric/epistemic 分离、composability、decision alignment 等）和"先声明推理链再做宣称"的可操作纪律。
- **与课程主题的关系**：这是本讲第三块的**概念基座**。它虽然讲的是统计不确定性，但骨架完全可以平移到 agent-system 接口设计：Agent 的"意图"（想做什么、声称什么）必须与系统的"实现语义"（实际上怎么执行、报什么数）对齐，否则就会发生工程版的 construct drift——例如 Agent 把测试通过率当"修复成功"、把覆盖率当"正确率"。2410.15625 正是把同一句"对齐 intent 与 implementation"落实到系统层：给 LLM 优化器一个 DSL（把映射决策显式化成可搜索空间）加一个 AutoGuide（把原始执行输出翻译成可行动的自然语言反馈），而不是让模型直接面对 C++ 裸系统。这也呼应 L14（Agent 评测）：评测指标本身也是一种"接口"，指标定义错了，系统再好也白搭。
- **可演示的代码点**：用合成数据演示"方差不是不确定性"（跨模型方差没有覆盖保证）；写一个迷你 conformal-style 校准检查对比原始 std 与校准后的区间覆盖率；把"declare the inference chain"做成一个模板函数，让学生给自己的 脚本化 评测声明目标与指标。

> 注：真正对应本讲 OUTLINE 第 3 节"Agent-System 接口设计"的论文（arXiv:2410.15625，正确文件缺失）要点速记，供重下载后参考：框架名 **Agent-System Interface (ASI)**，由 DSL + AutoGuide 两部分组成。DSL 把任务到处理器/数据到存储的"mapper"编写从 C++ 变成声明式，代码量平均缩小约 14x，并显式定义可搜索的结构化搜索空间；AutoGuide 把原始执行输出（如 "assertion failed: stride mismatch"）翻译成可行动建议（"内存布局不对，调整 layout 约束"）。结果：10 次迭代的生成式优化器就超过跑了 1000 次迭代的 OpenTuner（快 3.8x；同样 10 次则快 11x）；在 9 个基准（Circuit/Stencil/Pennant + 6 种 matmul 算法）上相对专家 mapper 最高 1.34x 加速；调优时间从数天降到约 10 分钟；消融显示 DSL 单次生成成功率 80%（C++ 为 0%），完整反馈（执行+解释+建议）显著优于任何降级反馈。

## 教学主线（想象 Stanford 老师会怎么教）

按"真实任务 → 拆解 → 评测 → 接口"组织，三篇论文是同一个 SWE 故事的四步：

1. **先立靶子：SWE-bench 与 test-time compute 的相遇**。回顾 L2 的结论——LLM Monkey 工作发现 coverage 随采样数 log-linear 增长，但"能采到正确解"不等于"能提交正确解"。抛出一个刺眼数字：CodeMonkeys 随机选择只有 45.8%，oracle 选择（即 coverage）69.8%——正确解生成了，却选不出来。这就是 SWE 版"生成-验证差距"（呼应 L3）。这里给出本讲脚手架：把修 issue 切成**找上下文 → 生成候选 → 选择答案**。
2. **CodeMonkeys：怎么把 test-time compute 真正吃进系统**。逐段讲：①上下文（因为要采样 10 条轨迹，逐文件扫描的成本被摊薄，朴素方法反而划算，这是"为 test-time compute 反推设计"的第一个例子）；②生成（两个背靠背状态机：先单独迭代出测试，再让编辑与测试互相纠错，强调"测试要在未编辑库上失败、编辑后通过"的两侧调试）；③选择（多数投票 vs 模型选择 vs 选择状态机，逐步逼近 oracle）。**读者容易卡住的地方**：串行缩放（更多迭代）与并行缩放（更多轨迹）到底差在哪——用图 5 的"前沿曲线"讲：同预算下 coverage 相近，但并行会让选择变难、串行会撞上"模型错误批准"的天花板。
3. **KernelBench：把"对"和"快"同时考**。老师会问一个反问式的问题：CodeMonkeys 的评测是二元的（修好/没修好），但如果任务的目标是"性能"呢？引出 fast_p：一个标量同时编码正确性和加速比，p 是旋钮。先给打击性的 one-shot 结果（前沿模型 <20% 快于 PyTorch），再给解法：执行反馈 E + profiler 反馈 P 的迭代让 R1 的 L2 从 36%→72%。**读者容易卡住的地方**：为什么单纯加执行反馈（E）效果显著，再加 profiler（P）只在部分模型有用——因为"正确性反馈不如编译错误那么有信息量"，模型能不能消化 profiler 输出取决于基座。可以放一段真实的 wmma 内核（图 10）展示"模型会尝试它没吃透的硬件指令"。
4. **Agent-System 接口：意图要与实现对齐**。从 UQ 论文借概念：construct drift = 为 A 算的不确定度被拿来支撑关于 B 的结论。平移到系统层：Agent 的意图（声称修好了、声称变快了）必须与系统的实现语义（测试怎么判、计时怎么计）对齐，否则指标漂移。真正的 ASI 论文（2410.15625）给出落地方案：用 DSL 把"映射决策"显式化成一个可搜索空间、用 AutoGuide 把原始执行输出翻译成可行动建议——接口是 Agent 与系统之间的一层"翻译契约"。
5. **收尾与预告**：一条轴串起整讲——真实任务的 agent 系统 = 上下文检索 + 修复循环 + 选择器 + 面向目标的评测接口；并预告 L11（记忆：Cartridges/MemGPT 解决上下文管理）、L14（评测：SWE-bench 系列与长程任务评测）。

关键对照表（可用于 notebook 的总结 cell）：

| 论文 | 解决什么 | 核心机制 | 关键指标 | 一句话工程启示 |
|---|---|---|---|---|
| CodeMonkeys | 修真实 GitHub issue | 串行状态机修复 + 并行采样 + 选择状态机 | 57.4%（oracle 69.8%） | 上下文成本靠并行摊薄，选择决定一切 |
| KernelBench | 写高效 GPU 内核 | 生成-编译-执行-profiler 反馈环 | fast_p；R1 L2 36%→72% | 评测接口（正确+加速）反推系统设计 |
| UQ 立场论文 | 对齐不确定度的意图与实现 | 目标×构造×可信轴诊断框架 | 无实验；诊断清单 | 声明推理链，防止指标漂移 |
| (ASI 2410.15625) | Agent 优化并行程序 | DSL + AutoGuide | 10 迭代超 OpenTuner 千迭代 | 接口把系统语义翻译成 Agent 能用的形式 |

## 代码演示点子（3-6 个）

1. **迷你 SWE 修复循环（串行 test-time compute）**：从零实现一个最简状态机——循环体是"LLM 生成编辑 + 测试脚本 → 在 脚本化 代码库上跑测试 → 拿 exit code/输出当反馈 → 再生成"，直到模型批准或达到迭代上限。用 `llm_client.py` 的 脚本化 分支提供脚本化的"越来越好的编辑"，跟踪每轮测试通过情况。期望输出：一个能可视化"迭代次数 × 修复质量"的曲线，体现串行缩放收益递减（前几次迭代收益最大，对应论文图 5）。
2. **并行采样覆盖率曲线**：给定每题的"单次采样成功概率"（合成数据，难度各异），实现 coverage = 1 − ∏(1 − p_i)，画"采样数 N 对覆盖率"的 log-linear 曲线；再加一个串行+并行的预算分配版（把总预算 B 分成 B_serial 轮 × B_parallel 条），验证"同预算不同分配 coverage 相近"。期望输出：复现 LLM Monkeys / CodeMonkeys 图 5 的直觉。
3. **三种选择器对比（test 投票 / 模型选择 / 选择状态机）**：合成一批候选编辑（每个带隐藏的正确性标签 + 一批生成测试的通过情况），实现"测试多数投票"、"直接模型选择"、"先过滤 top-3 再选择"，比较各自的分数与 oracle（coverage）之差。期望输出：选择状态机 + 过滤 > 纯投票 > 随机，量化"选择回收了随机与 oracle 之间多少差距"。
4. **fast_p 与加速比分布**：用 numpy 造一批"候选内核"（每个带 `is_correct` 和 `speedup`），实现 fast_p 公式并画 p 从 0 到 2 的下降曲线；再用一个简化版"正确性检查"（随机输入对输出容差）说明 5 组随机输入怎么近似判定正确。期望输出：fast_0 = 纯正确率、fast_1 = 达标率、p 越高越难，直观展示"阈值是旋钮"。
5. **CPU 上的融合演示（伪 KernelBench）**：选一个简单算子序列（如 `matmul → ReLU → sum(dim=1)`），分别用 numpy 的逐算子写法与手工融合写法实现，计时并断言数值一致。期望输出：融合版更快的量级对比，让学生不用 GPU 也能体会 KernelBench Level 2 的"融合"动机。**实验语料**：固定 seed、固定形状、多次计时取中位数。
6. **意图-实现对齐：方差不是不确定性**：合成一个简单回归任务，训练（或模拟）多个模型，对比两种"不确定性"：①跨模型输出的方差（deep ensemble 的常见做法）；②一个带边际覆盖保证的校准区间。在留出集上计算方差解释不了多少真实误差、而校准区间有近似名义覆盖。期望输出：一张"覆盖 vs 名义水平"的校准曲线，直观演示 construct drift（呼应 UQ 论文的三根可信轴）。

## 作业点子（3 个）

1. **覆盖率与采样数**：给定每题单次采样成功概率 `p` 与采样数 `n`，填空实现 coverage 公式。`assert abs(coverage(0.2, 10) - (1 - 0.8**10)) < 1e-9`、`assert coverage(0.2, 100) > coverage(0.2, 10)`。小提示：覆盖率 = 至少一条成功的概率 = 1 − 全部失败的概率；题目之间假设独立。
2. **fast_p 计算**：给定候选列表（每项 `(is_correct, speedup)`），填空实现 `fast_p(candidates, p)`。`assert fast_p(cands, 0) == 正确条数/总数`、`assert fast_p(cands, 1) <= fast_p(cands, 0.5)`。小提示：把"正确"与"speedup > p"两个条件用 and 组合再取均值。
3. **用测试做多数投票的选择器**：给定一个 `passes` 布尔矩阵（行=生成测试，列=候选编辑），填空实现"选出通过测试最多的编辑，平手时选 diff 最短的"。`assert select(passes, lengths) == 期望的下标`。小提示：`passes.sum(axis=0)` 是每列通过数；平手时在通过数最大的一组里按 `lengths` 取最小。

## 参考资料

- Ehrlich et al., *CodeMonkeys: Scaling Test-Time Compute for Software Engineering*（arXiv:2501.14723）— 本讲第一主力；代码与轨迹在 scalingintelligence.stanford.edu/pubs/codemonkeys
- Ouyang et al., *KernelBench: Can LLMs Write Efficient GPU Kernels?*（arXiv:2502.10517）— 本讲第二主力；框架开源（github.com/scaling-intelligence/KernelBench）
- Trivedi & Nord, *On the Need to Align Intent and Implementation in Uncertainty Quantification for Machine Learning*（arXiv:2506.03037）— 实际装在本讲文件夹里的立场论文，intent/implementation 对齐的概念基座（⚠️ 与预期文件不符，见论文 3 开头说明）
- Wei et al., *Improving Parallel Program Performance with LLM Optimizers via Agent-System Interfaces*（arXiv:2410.15625，ICML 2025）— 真正对应"Agent-System 接口设计"一节的论文（DSL + AutoGuide），**本讲缺此 PDF，写 notebook 前请重下载**
- Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*（arXiv:2310.06770）— SWE-bench 基准本身；SWE-bench Verified 见 OpenAI 官方博客（2024-08）
- Brown et al., *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*（arXiv:2407.21787）— CodeMonkeys 的直接前作，"coverage 随采样 log-linear 增长"的出处
- Yang et al., *SWE-Agent: Agent-Computer Interfaces Enable Automated Software Engineering*（arXiv:2405.15793）— 另一个 SWE-agent 框架，强调 agent-computer 接口，与 L10 第 4 点互补
- Xia et al., *Agentless: Demystifying LLM-based Software Engineering Agents*（arXiv:2407.01489）— 同样采用"定位-修复-验证"三段式，CodeMonkeys 的对照系
- Tillet et al., *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*（2019）— KernelBench 允许使用的高层内核语言之一
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*（arXiv:2205.14135）— 论文反复引用的人类写高效内核标杆
