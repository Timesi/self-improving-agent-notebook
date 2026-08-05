# Lecture 19 — 多模态机器人智能体（Multimodal AI Agents in Robotics）研读笔记

> 本文件是 CS329A 第 19 讲的论文研读笔记，是编写对应 notebook 的素材。
> 来源：https://cs329a.stanford.edu/（Autumn 2025 课程大纲）。本讲嘉宾 Danny Driess（Physical Intelligence）。

## 课程主题

这一讲回答一个核心问题：Agent 不能只"想"，还要"动"——当 Agent 拥有一个物理身体、需要在真实世界里完成操作任务时，感知、语言、动作如何融进一个统一的模型？答案是**视觉-语言-动作模型（Vision-Language-Action Model，VLA）**：把机器人连续动作离散化成 token，直接接在 VLM 后面做 next-token 预测，让一个端到端模型既能理解语言指令和图像场景，又能输出末端执行器的控制指令。

这一讲在课程里是"Agent 的终极形态"：前面 L1-L18 的 Agent 都在**数字世界**里行动（工具调用、代码、搜索、推理），这一讲把 Agent 循环放到**物理世界**——感知 → 决策 → 动作 → 观察反馈，正是最完整的 Agent 循环。同时它回扣课程反复出现的两大主线：**规模（scale）**（VLM 的 web 预训练知识迁移到机器人，弥补机器人数据只有几十万到一百万条的量级鸿沟）和**从物理世界学习**（遥操作数据采集、真实机器人闭环评测、sim-to-real 反馈）。本讲还安排在 L20 未来方向之前，为"通用具身智能体"收尾。

## 论文精读

### 论文 1：RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control（arXiv:2307.15818，rt-2.pdf）

- **核心思想**：机器人数据太少（13 台机器人、17 个月才几万条轨迹），而 web 上有数十亿的图文 token 预训练。能不能不新造架构，直接把**现成的 VLM** 微调成能输出低层控制指令的策略？RT-2 的做法简单到出奇：**把动作当成"另一种语言"**——把连续动作离散化成 token，放进 VLM 的词汇表，与自然语言一起参与训练。这一模型类别就叫 VLA（vision-language-action），RT-2 是其第一个大规模实例。核心洞察：模型不用学会新动作，web 知识赋予它的是"把已学会的动作以新方式派上用场"的能力（语义理解、符号识别、跨语言、基础推理）。

- **关键公式/算法**：
  - **动作空间**：7DoF 移动操作臂。动作 = 6 维末端执行器位姿位移（$\Delta\text{pos}_{x,y,z}$、$\Delta\text{rot}_{x,y,z}$）+ 夹爪开度 + 1 个终止指令。除终止指令外，每个连续维度**均匀离散成 256 个 bin**，用 bin 序号表示，共 8 个整数。
  - **动作字符串格式**：`"terminate Δpos_x Δpos_y Δpos_z Δrot_x Δrot_y Δrot_z gripper_extension"`，例如 `"1 128 91 241 5 101 127"`。把动作向量拼成字符串，动作 token 与自然语言 token 一起进训练集，模型以标准 VQA 格式回答 `"Q: what action should the robot take to [task]? A:"`。
  - **token 分配**：PaLI-X 对 0-1000 的整数各有独立 token，直接绑定 bin 序号与对应整数 token；PaLM-E 没有该便利，直接**覆盖 256 个最不常用 token**。覆盖现有 token 是一种 symbol tuning。
  - **co-fine-tuning（关键细节）**：不是只在机器人数据上微调，而是把机器人轨迹数据与原始 web 视觉-语言数据**一起**微调，并提高机器人数据在每批里的采样权重。这防止模型遗忘 web 学到的抽象视觉概念，是泛化性能提升的关键。
  - **输出约束（output constraint）**：机器人任务解码时把词汇表屏蔽成**只允许采样合法的动作 token**；普通 VQA 任务仍可输出全部自然语言 token。
  - **实时推理**：55B 模型放多 TPU 云端、经网络查询，1-3 Hz；5B 版约 5 Hz。
  - **CoT 变体**：额外微调几百步，数据增广成 `"Instruction: I'm hungry. Plan: pick rxbar chocolate. Action: 1 128 124 136 121 158 111 255."`，让模型先生成自然语言计划再输出动作。

- **关键实验结论**（约 6000 次真实机器人评测）：
  - **seen 任务**与 RT-1 相当；**泛化（unseen 物体/背景/环境）**平均约为 RT-1 和 MOO 的 **2 倍**、约为 VC-1/R3M 的 **6 倍**。
  - **涌现能力**分三类：符号理解（"move apple to 3"）、推理（"move the apple to the cup with same color"、数学、多语言）、人类识别（"move coke can to the person with glasses"）；最佳 RT-2-PaLI-X 平均成功率是 RT-1 的 **3 倍以上**。
  - **消融**（图 6b）：55B/5B 从零训练效果很差（5B 从零训练已崩，55B 直接不测）；co-fine-tuning 优于仅机器人数据微调；**模型越大泛化越好**。
  - Language-Table 仿真：RT-2-PaLI-3B 达 **90 ± 10**，远超 BC-Zero 72±3、RT-1 74±13、LAVA 77±4。
  - 局限：web 数据不带来**新动作**（物理技能仍限于机器人数据分布）；大模型推理成本高。

- **与课程主题的关系**：RT-2 是 VLA 的奠基论文，确立了两条贯穿本讲的原则：(1) **动作即 token**——动作空间离散化 + 复用现有 VLM 词汇表；(2) **web 知识迁移**——用规模弥补机器人数据稀缺，模型能泛化到训练中从未见过的指令与场景。它也是"一个模型同时当 VLM 和策略"的范式起点，OpenVLA 正是它的开源化与工程化。

- **可演示的代码点**：
  - 从零实现 256-bin 均匀离散化与 `"1 128 91 241 5 101 127"` 动作字符串的编解码。
  - 实现解码时的**动作 token 词汇表屏蔽**（output constraint）。
  - 用 `llm_client` 演示 RT-2 的 `Instruction → Plan → Action` 链式推理格式并解析出动作 token。

### 论文 2：OpenVLA: An Open-Source Vision-Language-Action Model（arXiv:2406.09246，openvla.pdf）

- **核心思想**：RT-2 有两个阻碍普及的问题：模型关闭不开源、且没研究如何高效微调到新任务。OpenVLA 是**第一个开源通用 VLA**：7B 参数，在 Open X-Embodiment 数据集 970k 条真实机器人轨迹上微调，开箱即用控制多种机器人，并系统研究了参数高效微调（LoRA）与量化推理，让 VLA 在消费级 GPU 上也能微调和服务。它证明"小的、开源的、可微调的"可以打败"大的、封闭的"：比 RT-2-X（55B）**少 7 倍参数**却在 29 个任务上平均成功率高 **16.5%**。

- **关键公式/算法**：
  - **架构**：Prismatic-7B VLM = 视觉编码器 + 2 层 MLP projector + Llama 2 7B 语言主干。视觉编码器是**融合双编码器**：SigLIP（高层语义）+ DINOv2（低层空间细节）的特征按通道拼接，DINOv2 的加入显著改善空间推理（对机器人控制尤其重要）。
  - **动作 token 化**：沿用 RT-2 的 256-bin 离散化，但 bin 宽度用动作在训练数据上的**第 1-99 分位数**区间均匀切分（RT-2 用 min-max，容易被离群动作把区间撑大、降低有效分辨率）。N 维动作 → N 个 $[0,255]$ 整数。
  - **token 分配**：Llama tokenizer 只预留 100 个特殊 token，不够 256 个动作 token，因此覆盖词汇表**最后 256 个最不常用 token**（即 last 256 tokens）。
  - **训练目标**：标准 next-token prediction，交叉熵**只在动作 token 上计算**。在 BridgeData V2 上先做小规模设计实验（跑得快），再上全量数据训练。
  - **数据**：Open X-Embodiment（70+ 子数据集、>2M 轨迹）经筛选：只留含至少 1 个第三人称相机、单臂末端控制的**操作**数据集，用 Octo 的混合权重配平；DROID 曾以 10% 权重加入，因动作 token 准确率上不去而在最后 1/3 训练中移除。最终 970k 轨迹。
  - **设计决策**（小规模实验得出）：微调视觉编码器很关键（冻结则性能明显下降——预训练视觉特征不够细粒度）；图像 224×224 与 384×384 无性能差异（后者训练慢 3 倍）；LR 固定 2e-5；训练跑 **27 个 epoch**（LLM/VLM 通常 1-2 个），真实性能一直涨到动作 token 准确率过 95%。
  - **基础设施**：64 张 A100 训练 14 天（共 21,500 A100 小时），batch 2048；bf16 推理占 15GB、RTX 4090 上约 6 Hz。

- **关键实验结论**：
  - **开箱即用评测**（BridgeData V2 170 次 rollout / Google robot 60 次 rollout，A/B 同条件）：OpenVLA 在 BridgeData V2 上显著优于 RT-2-X（55B），在 Google robot 上与 RT-2-X 相当；两者都远超 RT-1-X（35M）与 Octo（93M）。RT-2-X 仅在"语义泛化"类别更高（它 co-fine-tuning 保留了更多 web 知识，OpenVLA 只做机器人数据微调）。
  - **数据与组件功劳**：OpenVLA 用 970k 轨迹（vs RT-2-X 的 350k）、清洗了 Bridge 数据集里的全零动作、用融合视觉编码器。
  - **微调到新机器人**（Franka，10-150 条演示）：全参微调下 OpenVLA 聚合性能最高，是唯一所有任务都 ≥50% 的方法；Diffusion Policy 在窄的单指令任务上更精细，但多物体、多指令、需语言 grounding 的任务上 OpenVLA 占优，比 Diffusion Policy 平均高 **20.4%**。
  - **LoRA 参数高效微调**（表 1）：LoRA rank=32 只需微调 **1.4% 参数**就追平全参微调（68.2% vs 69.7%），单张 A100 10-15 小时完成（全参的 1/8 计算量）；只微调最后一层（30.3%）或冻结视觉编码器（47.0%）都不行。
  - **量化推理**（表 2）：int4 量化性能与 bf16 相当（71.9% vs 71.3%），显存从 16.8GB 降到 7.0GB；int8 反而变慢（量化开销）。
  - 局限：只支持单图、无本体感知/历史；推理吞吐限制了 ALOHA 这类 50Hz 高频任务；成功率仍 <90%。

- **与课程主题的关系**：OpenVLA 是 VLA 的**开源化与工程化**收尾。它把 RT-2 的范式变成一个可复现、可下载、可在消费级 GPU 上微调的生态，直接呼应课程"亲手实现 + 可实验"的目标：notebook 里读者可以加载 OpenVLA（HuggingFace）或从零实现它的核心（动作 token 化、mini VLA 前向、LoRA）。它也是 Danny Driess 所在 Physical Intelligence 后续工作（π0）之前，社区最通用的 VLA 基线。

- **可演示的代码点**：
  - 从零实现**分位数 binning**（OpenVLA 式）并对比均匀 binning（RT-2 式）对离群动作的敏感度。
  - 用 torch 搭一个 **mini VLA**（vision patch embed + 指令 token + 小 transformer，预测 256 类 × 8 维动作 token）。
  - 从零实现 **LoRA** 低秩微调，数 trainable 参数占比，对比全参微调。

## 教学主线（想象 Stanford 老师会怎么教）

1. **动机：Agent 的终极形态是有身体的。** 开场放一个失败案例：一个能写代码、能搜资料的 Agent 面对一台真实机械臂却不会抓杯子。由此引出"具身智能 = Agent 放进物理世界"：感知 → 决策 → 动作 → 观察反馈，这是最完整的 Agent 循环，前面所有讲的工具调用、验证、推理在这里都变成"如何用身体做事"。老师的类比：语言的 Agent 像在棋盘上指指点点，具身 Agent 才是亲手落子。

2. **卡点：机器人数据太少。** 给数字对比——web 有几十亿图文 token，机器人最大数据集（Open X-Embodiment）只有约 100 万条轨迹。任何从零训练的策略都无法获得 web 模型的语义常识。由此提出核心问题：能不能直接复用 VLM？引出"VLM 到 VLA"的飞跃。

3. **方法 A：RT-2——动作是另一种语言。** 核心一步是**动作空间离散化**：连续 7 维动作（6 维位姿位移 + 夹爪 + 终止）每维切 256 个 bin，拼成字符串 `"1 128 91 241 5 101 127"`，动作 token 就混在自然语言 token 里做 next-token 预测。手算一个连续动作 → bin 序号 → token 字符串，读者最容易卡在"为什么 256？为什么覆盖词表 token？"。接着讲 **co-fine-tuning**：机器人数据 + web 数据一起训、加机器人采样权重，防止遗忘。最后用涌现能力（move apple to 3、move to cup with same color、pick rock as hammer）说明 web 知识真的迁移进来了。卡点预警：读者容易以为 web 数据带来新动作，其实只带来新语义——动作技能仍受机器人数据分布限制。

4. **方法 B：OpenVLA——开源 + 高效微调。** 先指出 RT-2 的两个问题：封闭、不可微调。然后展示 OpenVLA 的架构取舍：Prismatic = SigLIP+DINOv2 融合视觉 + 2 层 MLP + Llama 2 7B；分位数 binning 比 min-max 抗离群；训练 27 个 epoch 才收敛。再讲"小模型为什么能赢大模型"：更大的数据量（970k vs 350k）+ 更干净的清洗 + 融合视觉特征。最后落到**可落地性**：LoRA 只训 1.4% 参数追平全参，int4 量化显存减半不降性能——让 VLA 从云端进消费级 GPU，呼应课程的"可复现"精神。

5. **从物理世界学习的反馈闭环。** 收尾强调整个循环不是一次训练完事：遥操作采集数据 → 训练 VLA → 真实机器人闭环评测 → 失败样本回流。补一个未来视角：Danny Driess 所在的 Physical Intelligence 的 π0 用 flow-matching 连续动作头，是"离散动作 token"之外的另一条路线；而 RT-1（35M 离散化 transformer）、RT-X（跨本体）构成了这条脉络的上下游。

## 代码演示点子（4-6 个）

1. **手写动作 token 离散化**：从零实现 RT-2 式均匀 binning 与 OpenVLA 式分位数 binning。用 numpy 生成一组 7 维连续动作，分别用 min-max 与 1-99 分位数定 bin 区间，比较两者在混入离群动作后有效分辨率的差别；再把 bin 序号拼成 `"1 128 91 241 5 101 127"` 并写反向 de-tokenize。期望输出：两个方法的 bin 区间、一个动作的 token 字符串、反解回连续动作的误差。

2. **mini VLA 前向（torch 从零搭）**：用一个极小的 transformer-decoder 充当"VLM 主干"：图像经过一个小 patch embed 变成视觉 token，指令文本经 embedding 层变文本 token，拼接后因果解码，最后一层预测 8 个位置的 256 类动作 token。用合成 episodes 训几个 step，画动作 token 准确率上升曲线；再在解码端实现**词汇表屏蔽**（只采样动作 token）。期望输出：loss/acc 曲线 + 采样出的动作 token 字符串。

3. **遥操作数据格式与批量处理**：模拟 Open X-Embodiment 式 episode 数据（dict 或 npz：每 episode 含一组图像、一条语言指令、一组 7 维动作），演示从原始轨迹 → 离散化 → 动作 token 字符串 → 按 (图像, 指令, 动作 token) 打批（padding + attention mask）的整条数据管线，统计一个 epoch 里的动作 token 数。期望输出：数据量统计、首个 batch 的 shape、一条被解析出的 `Plan/Action` 样本。

4. **co-fine-tuning 反遗忘小实验**：在 mini VLA 上先做一轮"web 任务"（图像分类/描述）预训练，然后对比两种微调：(a) 只训机器人数据；(b) 机器人数据 + 少量 web 数据混合（提高机器人采样权重）。测两类任务的准确率，直观看到只训机器人数据时 web 任务准确率崩塌、co-fine-tuning 则保住。期望输出：两条"遗忘"曲线，复现 RT-2 图 6b 的结论。

5. **LoRA 低秩微调从零实现**：在 mini VLA 的线性层上加 LoRA（$W = W_0 + BA$，低秩 $A,B$），只训练 $B$ 和 $A$，统计可训练参数占比并对比全参微调在一个新任务上的表现与显存/参数开销。期望输出：参数量对比表，复现 OpenVLA"1.4% 参数追平全参"的数量级。

6. **Plan + Action 链式推理（llm_client）**：解析 RT-2 的 CoT 数据格式 `"Instruction: ... Plan: ... Action: 1 128 ..."`，用 `llm_client`（mock 模式给脚本化轨迹）让模型先生成计划再生成动作 token，演示"语言推理桥接到低层控制"。期望输出：解析出的 Plan 文本与解码后的 7 维动作，并说明动作 token 需经词汇表屏蔽。

## 作业点子（3 个）

1. **实现动作离散化**：考察 binning 与抗离群。给定连续动作数组与 bin 数 256，写 `quantile_binning(actions, lo=1, hi=99)` 返回每个维度的 bin 区间，并用它实现 `discretize(a, bins) -> int` 与 `detokenize(i, bins) -> float`；补全后用 assert 检查：反向误差在 bin 宽度以内、混入离群动作后分位数 bin 的区间宽度显著小于 min-max。小提示：先按维度取分位数再切，注意对越界值 clip。

2. **实现动作 token 字符串编解码（RT-2 格式）**：考察序列格式。给定 7 维 bin 序号（含终止位），写 `to_action_string(ids) -> str` 与 `parse_action_string(s) -> list[int]`，使 `"1 128 91 241 5 101 127"` 双向成立；补全后用 assert 检查：字符串以空格分隔、长度与维度数一致、roundtrip 后序号相同。小提示：`str.join` 与 `str.split`，注意终止位放在最前。

3. **实现解码时的动作 token 词汇表屏蔽**：考察 output constraint。给定完整词表 logits（形状 `(V,)`）与动作 token 集合 `action_ids`，写 `mask_for_robot_task(logits, action_ids)` 返回只保留动作 token 的对数概率分布；补全后用 assert 检查：屏蔽后其余位置的 logits 为 `-inf`、softmax 后非动作 token 概率为 0。小提示：用 `torch.full_like` 填 `-inf` 再 `scatter` 放回动作位置。

## 参考资料

- RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control（arXiv:2307.15818；https://robotics-transformer2.github.io）— VLA 的开创性工作：动作即 token + co-fine-tuning + web 知识迁移。
- OpenVLA: An Open-Source Vision-Language-Action Model（arXiv:2406.09246；https://openvla.github.io）— 第一个开源通用 VLA，LoRA 参数高效微调与量化的系统研究。
- RT-1: Robotics Transformer for Real-World Control at Scale（arXiv:2212.06817）— 35M 参数离散化动作 transformer，RT-2 的基石与数据来源。
- Open X-Embodiment: Robotic Learning Datasets and RT-X Models（arXiv:2310.08864；https://robotics-transformer-x.github.io）— 70+ 子数据集、跨本体的 RT-1-X / RT-2-X，OpenVLA 的训练数据源。
- PaLM-E: An Embodied Multimodal Language Model（arXiv:2303.03378）— RT-2-PaLM-E 的 VLM 主干，具身多模态语言模型的早期代表。
- π0: A Vision-Language-Action Flow Model（Physical Intelligence，2024）— 本讲嘉宾所在团队的工作：用 flow-matching 连续动作头取代离散 token，VLA 的另一条路线。
- Prismatic VLMs（Karamcheti et al., 2024，https://prismatic-vlms.github.io）— OpenVLA 的 VLM 主干，SigLIP+DINOv2 融合视觉编码器。
- LoRA: Low-Rank Adaptation of Large Language Models（arXiv:2106.09685）— OpenVLA 参数高效微调所依赖的低秩适配方法。
- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion（arXiv:2303.04137）— OpenVLA 微调实验对比的从零学习基线。
- CS329A 课程大纲 Lecture 19：Multimodal AI Agents in Robotics（https://cs329a.stanford.edu/）— 本讲在课程地图中的定位。
