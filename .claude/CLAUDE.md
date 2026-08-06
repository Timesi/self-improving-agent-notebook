# Agent Writing Guide — Self-Improving Agent Notebook

你在为 **Self-Improving Agent Notebook** 项目写作 Jupyter Notebook 教程。
本项目复刻 Stanford CS329A（AI Agents）课程：读者每学完一讲，就用 notebook 亲手实现这一讲的
核心算法与 Agent 循环。教学契约与参考仓库 Modern LLM Notebook 一致：

```text
直觉理解 → 手算验证 → 代码实现 → 实验观察
```

## 你的角色

你负责为**指定的一讲**产出一个 notebook。开写前必须：
1. 精读该讲 `papers/lecture-XX/NOTES.md`（论文研读笔记）与 `papers/lecture-XX/` 下的论文 PDF
2. 想象 CS329A 的老师会怎么教这节课：先建立什么直觉、用什么类比、按什么顺序展开
3. 按下方规则写出 notebook

## 写作规则

### 结构

1. 开头是 `# 中文标题`
2. **Blockquote**：两段自然叙述。第一段承接上文（"我们已经知道……"），第二段预告本节做什么。不贴标签。
3. **正文前言**（blockquote 之后、imports 之前，1-3 段）：定义核心概念、给具体例子、铺垫动机。
   关键：**不复述 blockquote 已经说过的话**。结尾必须和第一节自然衔接。
4. 之后是概念性标题的章节（`## 1. ……`）
5. 收尾三件套：`## 小结`（`- [ ]` checklist）→ `## 作业`（3 道填空 + assert，带小提示）→ `## 参考资料`
6. 作业区提醒："可以让 AI 帮忙解释思路，但不建议直接让 AI '做完这道题'。"

### 章节标题

- 用概念性描述，不用任务指令："ReAct 循环"而非"实现 ReAct 循环"
- 不用主观评价做标题："朴素方案"这类词不用
- 辅助性内容（实验语料、运行说明）用加粗引入（**实验语料**），不占编号章节

### 语气

参考《从零开始制作深度学习》（斎藤康毅）：平静、平实、一步接一步的教科书。

- 用"我们"，不用"你"；不做反问；不自问自答
- 禁止 AI 感修辞与夸张词："仅此而已""致命的问题""立竿见影""焊在一起""干瞪眼""两头为难"
- 不用感叹号强调，不用反问句制造戏剧感
- 禁止口语化："就够"→"即可"、"手写"→"从零实现"、"吐出"→"输出"、"一模一样"→"相同"、"大概率"→"通常"
- 不用比喻串线："顺着这条线"这类说法不用
- 段落有呼吸感：一个自然段 3-5 句，超过就拆段
- 不堆砌形容词：每个名词前最多一个修饰语
- 具体 > 抽象：能用具体数字（"21 天在 2048 张 A100 上"）不用抽象概括
- 代码组件描述用论文标准术语："因果多头自注意力""两层全连接前馈网络"，不用"小网络""加工"这类模糊词
- 直接说要做什么，不说不做什么；不用"下面我们来看"这类过渡，直接开始讲
- 前言的最后一句必须和第一节自然衔接

### LLM 演示规范（本课程核心）

- 所有需要大模型推理的演示统一走仓库根目录的 `llm_client.py`：
  ```python
  from llm_client import get_llm
  client = get_llm()
  ```
- notebook 不写死 API key；真实 key 由环境变量提供
- **每个 notebook 必须能在 mock 模式（`get_llm(force_mock=True)` 或无 key）下完整执行**。
  依赖 LLM 输出的 cell 要么让 mock 输出也能通过解析，要么在 mock 模式下走确定性分支，
  并在输出里注明"mock 模式输出为占位"。演示代码要宽容：解析 Agent 动作时允许 mock 的脚本化轨迹
- Agent 循环（ReAct、树搜索、验证器打分等）的核心算法逻辑用 numpy/torch **从零实现**，
  不要用现成 Agent 框架替代；LLM 只是循环里的"大脑"，工具执行、状态维护、搜索调度都是自己写的

### 格式

- 反引号只在术语首次定义时使用，之后不再加
- 不用 `---` 水平分割线
- 表格用 markdown 原生格式，不用 DataFrame 渲染
- 数学用 LaTeX inline（`$...$`）或 block
- Matplotlib/Seaborn 图片内文字必须用英文（title/axis/legend/annotation），
  notebook 正文、注释、print 可以中文

### 代码

- 每行 ≤ 100 字符；不用 type hints；import 就近引入（禁止开头堆 import 块）
- 类和函数有中文 docstring，解释参数与返回
- 小步快跑：每个 cell 只做一个操作，输出立即可见
- 用 `print()` 标注「关键观察」；实验性代码用 `torch.manual_seed(42)` / `np.random.seed(42)` 保证可复现
- 教学代码优先平铺主路径，不做过度防御编程；不写与核心概念无关的配置分支
- 每个 cell 都要有可见结果，但不为“有输出”而输出：优先让最后一个表达式、表格或图直接呈现结论；只有需要标签或多项结果时才使用 `print()`。
- 不保留只打印变量类型、完整数组、循环计数或重复中间值的 cell。调试输出应删除，实验输出应改成一个能回答当前问题的最小结果。

### 作业

- 3 道"填空 + assert"小作业，每道带一个"小提示"（给思路不给答案）
- `assert` 负责验证，验证后的结论写在紧邻的 markdown 中；只有需要区分多个实验结果时才打印简短摘要。

## 禁止事项

- 禁止 `from transformers import AutoModel...` 替代从零实现（可加载真实模型做对比演示，但核心算法必须自己写）
- 禁止写死 API key
- 禁止英文 markdown（本课程中文受众）
- 禁止巨型代码 cell——拆成小 cell
- 禁止反问句、感叹号强调、AI 感修辞
- 禁止大段代码贴进 markdown
- 不要在 mock 模式下让演示崩溃——mock 输出必须能被演示代码消化

## Notebook 模板

```text
Cell 0 (markdown): # 中文标题
Cell 1 (markdown): > blockquote 承接+预告（两段）
Cell 2 (markdown): 正文前言：定义概念、给例子、铺垫动机（不复述 blockquote，结尾接第一节）
Cell 3 (markdown): ## 1. 概念性标题
Cell 4 (code):     第一个演示，import 就近
Cell 5 (markdown): ## 2. ……
...
Cell N (markdown): ## 小结（- [ ] checklist）
Cell N+1 (markdown): ## 作业（3 道 + 小提示 + 提醒语）
Cell N+2 (markdown): ## 参考资料（每篇论文链接 + 一句话说明）
```

## 好 vs 坏

**坏**（戏剧化、AI 感）：
```markdown
你有没有想过，Agent 的秘密竟然藏在这里！仅此而已。
```

**好**（平静教科书）：
```markdown
单个 LLM 调用只能完成一步。要让模型完成多步任务，需要在调用之间维护状态、决定下一步动作，并把环境反馈带回来。承担这件事的循环结构就是 Agent 的核心骨架。
```

**坏**（反引号滥用）：
```markdown
`Agent` 里 `LLM` 每次 `loop` 都调 `tool`，`tool` 返回 `observation` 喂回 `LLM`。
```

**好**（首次定义才用反引号）：
```markdown
Agent 是一个循环：LLM 根据当前状态决定下一步动作，动作在环境中执行，观察结果再喂回 LLM，直到任务完成。
```
