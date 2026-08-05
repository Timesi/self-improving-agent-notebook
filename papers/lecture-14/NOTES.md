# Lecture 14 — Augmenting Agents with Memory（给 Agent 加记忆）研读笔记

> 本文件是 CS329A 第 14 讲的论文研读笔记，是编写对应 notebook 的素材。
> 来源：https://cs329a.stanford.edu/（Autumn 2025 课程大纲）

## 课程主题

这一讲要解决的核心问题是：**Agent 的上下文窗口是有限的，但任务（长对话、长文档、长期个性化）需要的信息是无限的。怎么办？**

位置安排：在 13 讲 SWE 智能体（agent 在长代码库/长 issue 里反复工作，强烈暴露 context 瓶颈）之后、17 讲评测之前，专门讲"记忆"。前三篇论文正好把"记忆"拆成三个互补的层面：

1. **MemGPT——记忆的分层组织（软件层/控制层）**：把 context window 当作稀缺内存，仿照操作系统的虚拟内存分页，用函数调用让 LLM 自己"换页"——把当前最相关的信息放在 main context，把旧信息逐出到外部存储，需要时再检索回来。
2. **Cartridges——记忆的可学习压缩（表示层）**：与其决定"哪些文本放进 context"，不如把整篇语料蒸馏进一个离线训练的 KV cache（可学习参数）。这是把"记忆"从离散 token 变成连续向量表示，是 MemGPT 的"token 搬运"思路的另一种极致。
3. **CacheBlend——记忆的跨请求复用（系统层/工程层）**：多个请求共享同一批文本 chunk 时，如何复用它们预计算好的 KV cache 而不损失质量。它处理"复用时的 cross-attention 丢失"问题，是让上述记忆机制真正跑得快起来的工程基础设施。

一个贯穿性的直觉：**记忆不是"塞更多 token"，而是"用更聪明的存储/表示/复用，让有限 context 发挥无限能力"**。整讲从"软件控制"到"可学习表示"再到"系统复用"，构成一个从抽象到工程的三级台阶。

## 论文精读

### 论文 1：MemGPT: Towards LLMs as Operating Systems（arxiv:2310.08560，memgpt.pdf）

- **核心思想**：把 LLM 的固定 context window 类比为操作系统的"物理内存"，在它外面再加一层"磁盘"（外部存储），然后用 LLM 的函数调用能力实现"虚拟内存分页"——LLM 自己决定把哪些信息换入（main context）哪些换出（external context），从而在有限窗口上提供"无限上下文"的假象。系统被命名为 MemGPT（MemoryGPT），采用 OS 式的分层记忆架构，评估领域是长文档分析和多会话对话。

- **关键公式/算法**：
  - **两层记忆层级**：*main context*（类比 RAM，即 LLM 的 prompt tokens，推理时可见）与 *external context*（类比磁盘，在 context 之外，必须显式换入 main context 才能被推理使用）。
  - **main context 分三段**（固定顺序、连续拼接）：
    - *system instructions*：只读、静态，包含控制流说明、各记忆层级的用途、函数使用说明。
    - *working context*：定长、可读写的一段非结构化文本，只能通过函数写；对话场景用来存用户关键事实、偏好、persona。
    - *FIFO queue*：滚动消息历史（含用户/助手消息、系统消息、函数调用入参出参）；队列第一个 index 存一条"递归摘要"（对已逐出消息的递归总结）。
  - **Queue Manager（队列管理器）**：新消息到达 → 追加进 FIFO queue → 拼接 prompt tokens → 触发 LLM 推理 → 把输入消息和生成输出写进 *recall storage*（消息数据库）。
  - **逐出（eviction）策略**：当 prompt tokens 超过 *warning token count*（默认约 context 的 70%）时，向队列插入"内存压力"系统消息，提示 LLM 用函数把重要信息保存到 working context 或 *archival storage*；超过 *flush token count*（100%）时，flush 队列——逐出约 context 50% 的消息、用现有递归摘要+被逐消息生成新的递归摘要。被逐消息不再在 context 内可见，但永久存在 recall storage，可随时通过函数检索回来。
  - **Function executor（函数执行器）**：解析 LLM 输出的 completion tokens 为函数调用，校验参数后执行，把结果（含运行时错误）回喂给 LLM。自定向的读写：函数 schema 与自然语言描述写进 system instructions，LLM 自主决定何时 append/replace 自己的 memory。典型函数：`working_context.append`、`working_context.replace`、`recall_storage.search`、`archival_storage.search`/`insert`。检索带分页（pagination），防止结果溢出窗口。
  - **控制流 / 函数链（function chaining）**：事件（用户消息、系统消息、用户交互如"登录/上传完成"、定时事件）触发推理；LLM 可在输出里带 `request_heartbeat=true` 特殊参数，请求立即再做一次推理，从而串起多步检索；不带该 flag（yield）则暂停，等下一个外部事件。
  - archival storage 用 PostgreSQL + pgvector（HNSW 索引）做向量检索（cosine），"检索器"天然从外部存储的 search 功能涌现出来。

- **关键实验结论**：
  - **Deep Memory Retrieval（DMR，跨会话一致性）**：问及先前会话内容的窄答案问题。MemGPT 显著超过固定上下文基线：GPT-3.5 Turbo 38.7%→66.9%（ROUGE-L 0.394→0.629）；GPT-4 32.1%→92.5%（0.296→0.814）；GPT-4 Turbo 35.3%→93.4%（0.359→0.827）。基线能看到过去 5 段对话的"有损摘要"，MemGPT 则持有完整历史但只能分页检索。
  - **Conversation Opener（engagement）**：开场的个性化程度。MemGPT 能接近甚至超过人类手写开场（SIM-H：Human 1.000，GPT-3.5 0.817，GPT-4 0.773，GPT-4 Turbo 0.767）。把信息存进 working context 是 key。
  - **文档 QA（NaturalQuestions-Open）**：固定上下文基线的准确率被检索器性能"封顶"（只有塞进 context 的文档才可见），且随截断加剧而下降；MemGPT 不受检索文档数量影响，能反复查询 archival storage 并翻页，有效上下文不再受窗口限制。
  - **Nested KV retrieval（多跳）**：140 对 UUID（约 8k tokens，即 GPT-4 的窗口大小），嵌套层数 0-4。GPT-3.5 在第 1 层嵌套就掉到 0%；GPT-4 / GPT-4 Turbo 到第 3 层掉到 0%；MemGPT+GPT-4 对嵌套层数不敏感（反复用函数查 main context 里的键值对）。
  - 动机数字：context 很短（Table 1：Llama 2k、Llama 2 4k、GPT-4 发布时 8k、Claude 2 100k、GPT-4 Turbo 128k、Yi-34B-200k 200k），并且长上下文模型"只在开头和结尾记得好，中间丢失"（lost in the middle，引 Liu et al. 2023）。发布数据：扩充版 MSC、nested KV、20M 维基百科文章 embedding。

- **与课程主题的关系**：本文给出"记忆"的第一个答案——**分层的、由 Agent 自己管理的记忆**。它定义了 agent 记忆的标准词汇表（main/external context、working memory、recall/archival storage、eviction、递归摘要、内存压力警告），是整讲的概念地基。它与课程先前的工具调用（L04）和 ReAct 循环（L05）自然衔接：函数调用从"操作外部世界"延伸到"操作自己的记忆"。

- **可演示的代码点**：
  - 从零实现记忆层级数据结构（working context / FIFO queue / recall storage）与队列管理器，用 mock LLM 走完"内存压力 → 保存 → flush → 递归摘要"。
  - 实现函数执行器 + 简单的函数 schema，模拟 `working_context.replace("Boyfriend named James", ...)` 这类自定向改写。
  - 复现"memory pressure"阈值逻辑：70% 警告、100% flush、逐出 50%、生成递归摘要，可视化 main context 恒定有界。
  - 用 MockLLM + 脚本化函数调用演示多会话一致性问题（DMR 的简化版）。

### 论文 2：Cartridges: Lightweight and general-purpose long context representations via self-study（arxiv:2506.06266，cartridges.pdf）

- **核心思想**：大量应用反复把同一份大语料（代码库、财报、病历、聊天记录）塞进 context（ICL）。ICL 的代价是 KV cache 随输入线性增长，服务成本极高。Cartridges 提出**离线为每份语料训练一个"小 KV cache"**（可训练参数，本质是简化版 prefix-tuning），推理时把这个训练好的 KV cache（叫 *Cartridge*）加载进 LLM，再拼上用户 query 解码。训练成本可被"反复查询同一语料"摊薄。关键发现：直接用 next-token prediction（NTP）在语料上训练不行（只会背诵、不泛化）；他们提出 **Self-study** 配方：让模型自己生成关于语料的合成对话，再用 **context-distillation 目标**训练，从而复刻 ICL 的功能。

- **关键公式/算法**：
  - **Cartridge 参数化**：`Z = {zk, zv} ∈ R^{p×d}`（每层 p 个可训练 key/value 向量），内存占用等价于"p 个 token 的 KV cache"。做法：把 ICL 的 KV cache 中对应语料 C 的那 n_C 对 K/V 替换成 Z，其余参数全部冻结，只把 loss 反传到 Z 的 key/value 向量上。
  - **初始化（关键技巧）**：把 Z 初始化为"语料 C 前 p 个 token 的 KV cache"。消融（LongHealth 准确率）：随机向量 29.9%，随机 token 的 K/V 51.3%，用语料前 p 个 token 55.3%。随机初始化会不稳定、效果差（呼应 prefix-tuning 原文结论）。
  - **Self-study 合成数据（Algorithm 1）**：① 把语料 chunk（512–4096 token，支持超长语料）；② 取一个 seed prompt；③ 让同一个 LLM 扮演 A、B 两个角色交替采样 k 轮对话（A 的历史含 seed prompt + 语料子块在 system prompt；B 的历史不含 seed prompt，但角色对调），拼接成训练序列 `x = a1 ⊕ b1 ⊕ … ⊕ ak ⊕ bk`。主实验单轮对话 k=1。
  - **Seed prompts**：5 种通用类型（structuring 结构化、summarization 总结、question 提问、use cases 用例、creative 创作），所有语料共用同一套、不含语料特化信息。用 5 种随机采样优于单一 seed：LongHealth +4.8 准确率（43.6→48.4）、MTOB +7.9 chrF（24.1→32.0）。
  - **Context-distillation 目标**：
    `argmin_Z Σ_{(x,c̃)∈D} Σ_{i=1}^{|x|} D_KL( F(·|c̃ ⊕ x[:i]) || F_Z(·|x[:i]) )`
    教师 = 把语料子块 c̃ 放 context 的模型分布，学生 = 带 Cartridge 的同一模型分布。优于同数据量下的 NTP：MTOB +8.6 chrF（24.9→33.5）、LongHealth +3.7 准确率。
  - 消融：**KV cache 参数化优于 LoRA**——MTOB 上同内存 prefix-tuning 高 4.5 chrF；更关键的是对无关查询（MMLU）的破坏：LoRA 随 size 增大从 54.7 掉到 45.3，prefix-tuning 只从 54.7 掉到 54.3。冻结 attention sink（首个 token 的 K/V）提升训练稳定性。训练成本：LLaMA-8B 的 ICL 质量 Cartridge 约 30 分钟/单台 8×H100 节点。

- **关键实验结论**：
  - **核心数字**：跨基准平均，Cartridges 匹配 ICL 质量，同时内存少 **38.6×**、峰值吞吐高 **26.4×**。LongHealth 最多省 10×、QASPER 最多省 100× 内存；压缩类基线（截断、GPT-4o 摘要、DuoAttention）在 >2× 压缩时质量快速崩坏，而 Cartridge 是训练出的压缩。
  - **NTP 的失败**：用 107× 更少内存完美背诵语料，但只会在"背诵类"查询上表现好，其他查询类型（推理/结构/创作）不泛化。
  - **上下文长度外推（MTOB）**：LLaMA-8B（128k 窗口）用 Self-study 的 chunking 处理 484k token 的卡朗语（Kalamang）教材，前 130k token 上比 ICL 高 **11.0 chrF**，并匹配手工精选 60k 子集上的 ICL——窗口装不下的语料也能用。
  - **可组合性**：两个独立训练的 Cartridge（AMD 10-K、Pepsi 10-K）直接拼接即可回答跨文档问题，无需联合训练；显著优于"只放一个 Cartridge"和"截断版 ICL"（后者要 39.8 GB context）。数据集：LongHealth、MTOB、QASPER、GENCONVO（源自 FinanceBench）。

- **与课程主题的关系**：本文给出"记忆"的第二个答案——**把记忆做成可学习的连续表示**。MemGPT 在"控制层"搬 token；Cartridges 在"表示层"把整份语料压缩进 KV cache。它也展示了 ICL 的本质：ICL 质量其实可以被"蒸馏"进参数里，从而把"运行时昂贵的上下文"换成"离线便宜的表示"。这对"agent 的长期记忆"（如记住用户全部聊天记录）和"coding agent 的 full-repo 上下文"有直接启发，也是后续 15/16 讲把推理/数学与压缩结合的前瞻。

- **可演示的代码点**：
  - 用一个小 transformer（torch 从零实现因果自注意力，或加载小开源模型）冻结权重，训练 p 个可训练 K/V 向量，在玩具语料上对比 NTP objective 与 context-distillation objective 的泛化差异。
  - 手算/可视化初始化重要性：random / random-token / 语料前 p-token 三种初始化下的训练曲线。
  - 在 toy 语料上演示"背诵 vs 泛化"：NTP 训练出的 Cartridge 能续写但答不了问题；Self-study 数据 + KL 蒸馏后能回答。
  - 演示 Cartridge 组合：两个独立训练的 Cartridge 拼接后回答需要两者信息的跨文档问题。

### 论文 3：CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion（arxiv:2405.16444，cacheblend.pdf）

- **核心思想**：RAG 输入由多个文本 chunk 拼成，prefill（对整段输入算 KV cache）很慢，决定了 time-to-first-token（TTFT）。已有两类 KV 复用方案各有缺陷：*prefix caching*（vLLM/SGLang/RAGCache）只复用前缀 chunk 的 KV，其他 chunk 照常 prefill，对多 chunk 场景几乎没用；*full KV reuse*（PromptCache）复用所有 chunk 但忽略了 chunk 之间（以及 chunk 与前面文本）的 **cross-attention**，质量崩坏。CacheBlend 提出 **selective KV recompute（选择性 KV 重算）**：按层只重算一小部分 token 的 KV（其余复用），从而同时拿到 full KV reuse 的速度和 full KV recompute 的质量；重算的微小额外延迟还能与"从慢速存储加载 KV cache"流水线重叠，因此 KV cache 可以放到更慢更便宜的设备上而不增延迟。

- **关键公式/算法**：
  - **动机数字**：4000 token 输入在单张 A40 上 prefill 要 3s（Llama-34B）/ 6s（Llama-70B）。
  - **记号**：`KV_full`（全量重算的 cache）、`KV_pre`（预计算 cache）、`KV_new`（CacheBlend 更新后的 cache）；每层 i 产生前向注意力矩阵 `A_i`。定义 **KV deviation** `Δkv(KV_i, KV_full_i)[j] = |KV_i[j] − KV_full_i[j]|`（衡量某个 token 某层的 K/V 偏离全量重算多少），以及 **attention deviation** `Δattn(A_i, A_full_i)`（前向注意力矩阵的 L-2 范数差）。目标：快速把 KV_pre 更新到 KV_new，使各层 `Δattn(A_new_i, A_full_i)` 最小。
  - **选择性重算的工作流（per layer）**：对每层输入套 mask 只保留选中的 token → 只对选中 token 算 Q/K/V → 把未选中 token 的 K/V 从预计算 cache 里补回（这样注意力矩阵仍包含选中 token 与所有 token 的注意力）→ 跑同一注意力模块得到下一层输入。计算开销 ∝ 选中 token 数：重算 r% 的 token，开销就是全量 prefill 的 r%。
  - **选哪些 token 重算（HKVD）**：
    - Insight 1：重算 KV deviation 更高的 token，对降低 attention deviation 贡献更大 → 每层选 KV deviation 最高的约 10-20%（High-KV-Deviation，HKVD）token。
    - Insight 2：token 的 KV deviation 在相邻层高度相关（Spearman 秩相关高，因为 transformer 相邻层输入 embedding 变化缓慢）→ 不必每层都全量算 deviation，采用**渐进筛选**：第 1 层全量重算并选 r1%（略高于目标 r）个 token，下一层只在这 r1% 里选 r2%（略低于 r1%）继续重算，逐层收敛。
  - **为什么够**：注意力稀疏性——约 10-15% 的 token 拥有远超其他 token 的 KV deviation；只有与别的 chunk 有高注意力（高 cross-attention）的 token 才需要重算。
  - **RoPE 位置恢复**：预计算 chunk 的 K 向量只需乘一个旋转矩阵（`[cos mθ, −sin mθ; sin mθ, cos mθ]`），因为 RoPE 下注意力分数只依赖相对位置（附录给了证明）。这一步开销可忽略。
  - **流水线与装载控制（Loading Controller）**：第 i 层的重算在第 i−1 层 KV 加载完后即可开始 → 用 KV 加载延迟去"藏"重算延迟。两个延迟估计器（`T_recompute(r%, LLM, L) = r% × Prefill(LLM, L)`，`T_load(LLM, L, device) = PerTokenKVSize × L / Throughput(device)`），选取 r 使两者相等，再与质量下限 r* = 15% 取 max；或固定 15% 重算率，选"最便宜且 T_recompute ≥ T_load"的存储设备。例：Llama-7B 重算 15% 每层 3ms，从 NVME SSD 加载一层 KV 需 16ms → 延迟被完全隐藏。
  - **KV cache store**：把输入按文本 chunk 切分（应用相关，论文沿用 PromptCache 的策略）、对 chunk 文本 hash 查 cache，LRU 逐出；hash 表放 CPU（100 万 chunk 仅 16MB）。

- **关键实验结论**：
  - **TTFT 减 2.2–3.3×、吞吐增 2.8–5×**（相对 full KV recompute），质量损失 ≤0.01–0.03（F1/Rouge-L）。相对 prefix caching 也是 TTFT 2.2–3.3× 降、吞吐提升。
  - 相对 full KV reuse：几乎相同 TTFT，但 QA 任务 F1 高 0.1–0.2、摘要任务 Rouge-L 高 0.03–0.25（full KV reuse 质量常被超过 2× 优势拉开）。
  - 5%–18% 重算率下，相对 full KV recompute 质量损失 ≤0.002，可换算成 TTFT 减 4.1–6.6×。
  - 质量随 chunk 数量提升（F1 从约 0.15 到 0.35 区间），full KV reuse 与 full prefill 的差距随 chunk 变多而扩大 → 说明 cross-attention 在真实多 chunk 场景普遍存在。RAG 之外，CacheBlend 也优于 LangChain 的 MapReduce（TTFT 低 2–5×、F1 更高）和 MapRerank（质量高很多）。
  - 模型：Mistral-7B、Yi-34B、Llama-70B（后两者 8-bit 量化）；数据集：2WikiMQA、Musique（QA，F1）、SAMSum、MultiNews（摘要，Rouge-L）；chunk 512 token，取 top-6 chunk（L2 距离）。

- **与课程主题的关系**：本文给出"记忆"的第三个答案——**在工程/系统层让记忆复用变快**。前面两篇关心"记忆里存什么、怎么组织/压缩"，这一篇关心"同一段记忆被反复用时，怎么省 prefill"。它是让 MemGPT/Cartridges 这类机制真正落地的 serving 基建：reused context 的 KV 缓存 + 跨 chunk 的 cross-attention 恢复 + 慢速廉价存储。课程里它把"记忆"和"成本/吞吐"连起来，与 14 讲工程向（agent 要规模化服务）的定位吻合。

- **可演示的代码点**：
  - 手算 2-chunk 注意力矩阵：构造一个"消息/罗纳尔多进球数"式的两 chunk 问题，对比 full recompute、full KV reuse（无 cross-attention）、CacheBlend（只重算 HKVD token）三种情况下的前向注意力矩阵差（numpy）。
  - 从零实现 HKVD 选择与渐进筛选：给定相邻两层 KV deviation 矩阵，算 top-r% token 与 Spearman 秩相关，验证"层间高相关"。
  - 复现 attention sparsity：在玩具注意力上画出 KV deviation 的 CDF，观察 10-15% token 占大部分 deviation。
  - 用延迟模型演示流水线：模拟 `T_recompute(r)` 与 `T_load(device)`，让学生选 r 和存储设备（复制论文的 Loading Controller 决策）。

## 教学主线（想象 Stanford 老师会怎么教）

建议的讲授顺序与直觉类比：

1. **失败案例建立动机**：展示一个"金鱼记忆"的 assistant——上一会话记住的用户生日、偏好，这一会话全忘了；或一份 10-K 财报远超 context 装不下。引出问题：agent 要长期运行，context 窗口是硬约束，而且长上下文模型"中间丢"（lost in the middle，MemGPT 引言里引的证据）。**此处点出第一句话：Agent 缺的不是参数，是记忆。**
2. **OS 类比：分层记忆**：操作系统如何用"物理内存 + 磁盘 + 分页"假装有无限内存？把 LLM 的 context 比作内存、外部存储比作磁盘，函数调用比作分页指令（page in / page out）。这就是 MemGPT 的全部骨架。用图 3 的记忆层级图讲清 main context 三段（system/working/FIFO）与 external context 两库（recall/archival）。
3. **控制机制：谁决定搬什么**：MemGPT 的精髓是"自定向"——LLM 自己决定何时 append/replace/evict/retrieve。讲清 memory pressure warning（70%）→ flush（100%）→ 递归摘要的闭环，以及 request_heartbeat 函数链。读者最容易卡住的地方：**为什么要递归摘要而不是直接删？为什么 eviction 要让 LLM 参与而不是 LRU 硬逐出？**（答案：记忆里藏着语义重要性，LLM 判断比固定策略好；但也可以指出这是代价——每次都烧 token。）
4. **从"搬运 token"到"学习表示"**：指出 MemGPT 局限——记忆操作全走文本、耗 token、受窗口约束。转向 Cartridges：能不能不搬文本，而是把整份语料压缩成可训练 KV cache？这里要讲清前缀调优（prefix-tuning）的本质：往输入前面加 p 个"虚拟 token"，但这次是每份语料训练一份，训练成本跨查询摊薄。
5. **self-study 的三个设计决策**：为什么 NTP 不行（背诵不泛化）→ 合成对话数据（两个角色互问互答、5 种 seed prompt）→ context-distillation 目标（教师=语料在 context，学生=带 Cartridge）。每个决策配一张消融图（Figure 3/6）。此处可手算一页 KL 蒸馏公式，让读者看到"对齐分布"而不是"背文本"。读者容易卡住：**为什么蒸馏而不是 NTP 就能泛化？**（因为蒸馏教的是"面对语料时的答题行为"，NTP 只教"语料本身长什么样"。）
6. **回到系统层：复用 KV 就是复用记忆**：如果 Cartridge 是一份"训练好的记忆"，那么 RAG 里反复出现的 chunk 其实共享同一份"记忆"（预计算 KV cache）。引出 CacheBlend 的问题：全量复用会丢 cross-attention。用一个 Messi/Ronaldo 对比题当例子，让读者亲眼看"两个 chunk 分开算 KV 再拼起来，模型答错"。
7. **手算/可视化理解**：在一页上画出三种方案的注意力矩阵（full recompute / full KV reuse / CacheBlend 选择性重算），直观看到黄色 cross-attention 块的有无。然后讲 HKVD 选择（Insight 1、Insight 2、渐进筛选）与"注意力稀疏→10-15% token 够用"。
8. **系统工程**：Loading Controller 流水线（重算一层时加载下一层），让慢速 SSD 变得可行。最后把三篇收拢成一句：**MemGPT 决定记忆放哪、Cartridges 决定记忆长什么样、CacheBlend 让记忆复用不花钱**——并预告它们都会出现在后续讲（如 15/16 的推理、17 的评测）里。

## 代码演示点子（3-6 个）

1. **从零实现 MemGPT 式分层记忆循环**：用 Python 实现 memory hierarchy（working context + FIFO queue + recall storage）+ 函数执行器，走 llm_client（mock 模式给脚本化轨迹）。关键思路：队列管理器按 token 计数插入 memory-pressure 系统消息（70%）、超限 flush（100%）并生成递归摘要；函数调用按 schema 校验后执行并回喂。期望输出：多轮对话后 main context 的 token 数始终 ≤ 预算，且旧信息可经 recall_storage.search 找回。
2. **可视化"逐出 + 递归摘要"的预算演化**：给一个固定 context budget，逐步 append 消息，画"main context 占用 vs 轮次"曲线，标注 warning/flush 点，展示每一轮 flush 后占用回落；把递归摘要的内容打印出来，让学生看到信息被压缩而非删除。数据用玩具脚本消息即可，不依赖 LLM。
3. **从零实现最小 Cartridge（prefix-tuning）训练**：用 torch 从零实现一个小的因果自注意力语言模型（或加载小模型），冻结全部权重，只训练 p 个可训练 K/V 向量。在玩具语料（如几段"人物档案"）上对比两种目标：NTP on corpus（只会背诵）vs context-distillation（KL 对齐教师分布，能回答合成问题）。期望输出：背诵损失更低的 Cartridge 答不了问题，而蒸馏出的 Cartridge 能答——直观复现论文 Figure 3 左图。
4. **手算 KV cache 复用与选择性重算**：构造两 chunk 输入（类似 Messi/Ronaldo），用 numpy 手算三种方案的注意力矩阵/前向注意力偏差（full recompute vs full reuse vs 只重算 HKVD token）。再实现 HKVD 选择 + 渐进筛选：给定相邻两层 KV deviation 矩阵，选 top-r% token，计算 Spearman 秩相关验证层间一致性（Insight 2）。期望输出：一张注意力热图 + 三条 deviation 曲线。
5. **Loading Controller 决策模拟**：复现 CacheBlend 的延迟模型——给定模型 prefill 速度、存储设备吞吐，函数 `T_recompute(r)` 与 `T_load(device)`，让学生求解"不增延迟的最大重算率"以及"给定 15% 重算率下的最廉价存储设备"。期望输出：与论文 Figure 10 一致的决策表/曲线，直观看到流水线为何能"藏"重算延迟。
6. **Cartridge 组合实验**：训练两个独立 Cartridge（两个 toy 文档），直接拼接后问一个需要两者信息的跨文档问题，对比"只放一个"与"ICL 全量塞 context"。期望输出：组合出的回答正确，且内存远小于 ICL。

## 作业点子（3 个）

1. **实现 MemGPT 队列逐出策略**：填空实现 `should_warn(tokens, budget)`、`should_flush(tokens, budget)` 与递归摘要更新，assert 每轮后 `len(main_context) <= budget`。考察：70%/100% 阈值、50% 逐出、摘要替换的语义。小提示：把"逐出的消息"接在旧摘要后面再喂给 mock LLM 生成新摘要。
2. **HKVD token 选择与层间相关性**：填空实现 `top_r_percent(kv_dev, r)` 选出每层 HKVD token，并用 `spearman` 计算相邻两层 deviation 秩相关；assert 相关性高于给定阈值、且重算 HKVD 后 attention deviation 低于重算低 deviation token。考察：Insight 1/2 的直接落地。
3. **context-distillation 目标填空**：填空补全 `d_kl` 项：`D_KL(F_teacher || F_student)`，其中教师 logits 来自"语料在 context"、学生 logits 来自"Cartridge 版本"，然后与 NTP 损失对比梯度方向；assert 蒸馏损失在"面对问题"时低于 NTP 损失。考察：对"对齐分布"而非"背诵文本"的理解。

## 参考资料

- MemGPT: Towards LLMs as Operating Systems（https://arxiv.org/abs/2310.08560）— OS 式分层记忆与虚拟上下文管理，本讲"记忆组织"的奠基论文。
- Cartridges: Lightweight and general-purpose long context representations via self-study（https://arxiv.org/abs/2506.06266）— 离线训练 KV cache 表示语料，self-study 合成数据 + context distillation。
- CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion（https://arxiv.org/abs/2405.16444，EuroSys '25）— 跨 chunk 复用预计算 KV cache 的选择性重算系统，代码 https://github.com/LMCache/LMCache。
- Lost in the Middle: How Language Models Use Long Contexts（https://arxiv.org/abs/2307.03172）— 长上下文模型"两头记得好、中间丢"的实证，MemGPT 的动机来源。
- Prefix-Tuning: Optimizing Continuous Prompts for Generation（https://arxiv.org/abs/2101.00190）— Cartridge 参数化的理论来源（可训练 K/V 向量）。
- PromptCache: Modular Attention Reuse for Low-latency Inference（https://arxiv.org/abs/2311.04934）— CacheBlend 的 full KV reuse 基线（忽略 cross-attention 的那一种）。
- H2O: Heavy-Hitter Oracle for Efficient Generative Inference（https://arxiv.org/abs/2306.14048）— KV cache 压缩的代表作，可与 Cartridge 的"训练式压缩"对照。
- Titans: Learning to Memorize at Test Time（https://arxiv.org/abs/2501.00663）— 用梯度下降式记忆更新的架构，Cartridges 引言里对标的最相关工作。
- Letta（原 MemGPT，https://github.com/letta-ai/letta）— MemGPT 的开源工程化产物，agent 长期记忆框架。
- CS329A 课程主页（https://cs329a.stanford.edu/）— Autumn 2025 课程大纲与本讲定位。
