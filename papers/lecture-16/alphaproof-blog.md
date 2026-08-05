# AlphaProof / AlphaGeometry 2 — DeepMind 博客摘要

来源：https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/
（"AI achieves silver-medal standard solving International Mathematical Olympiad problems", 2024-07-25）

## AlphaProof：形式化语言 Lean + 强化学习

- 用形式语言 Lean 证明数学命题的"能自我训练的系统"，把 AlphaZero 强化学习算法从棋盘博弈扩展到数学证明搜索
- 形式语言的好处：推理过程可被机器正式验证（验证器是可靠的）
- 瓶颈：人类书写的形式化数据极少。解决办法：微调 Gemini，把自然语言问题自动翻译成形式化陈述，
  构建约一百万道不同难度数学问题的大型形式化题库
- 训练循环：约 100 万道非正式问题 → 形式化器网络翻译成形式语言 → 求解器网络在 Lean 中搜索证明或反例 →
  AlphaZero 逐步自我强化，题目难度递增。竞赛期间系统持续运行，对题目自生成变体反复证明强化

## AlphaGeometry 2：神经-符号混合

- 前代 AlphaGeometry 的大幅改进：语言模型从零训练、合成数据多一个数量级；符号引擎快两个数量级
- 知识共享机制：组合不同搜索树解决更难问题
- 能处理对象运动、角度/比例/距离方程等难题
- 赛前可解过去 25 年 83% 的 IMO 历史几何题（前代 53%）

## IMO 2024 成绩

- 6 题解出 4 题（AlphaProof：2 道代数 + 1 道数论；AlphaGeometry 2：1 道几何），2 道组合题未解出
- 每道满分 7 分，共 28 分 = 银牌顶尖水平；金牌门槛 29 分（609 人中 58 人达到）
- 最难一题仅 5 名人类选手解出，AlphaProof 也解决了
- 成绩由 Fields 奖得主 Timothy Gowers 教授等按 IMO 规则评定
- 解题时间：几分钟到最长三天（人类是两场 4.5 小时）

## 与前代联系

- AlphaProof 直接复用 AlphaZero 强化学习范式：搜索可能的证明步骤 → 每个被验证的证明反过来强化语言模型
- 自我对弈式迭代改进是 AlphaZero 家族的核心特征

## 教学要点（对 16 讲）

- 验证器（verifier）在这里是"形式的、可靠的"——Lean 检查器，不是 LLM-as-judge
- 搜索（search）+ 强化学习（RL）是两条主线
- 神经-符号结合：神经网络提供候选步骤，符号引擎提供确定性推理
- 与 02/03 讲（test-time compute、robust verification）一脉相承
