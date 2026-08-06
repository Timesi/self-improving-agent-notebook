# AlphaGeometry — GitHub README 要点

来源：https://github.com/google-deepmind/alphageometry
论文发表在 Nature（Trinh, Wu, Le, He, Luong, 2024, DOI: 10.1038/s41586-023-06747-5），无 arxiv 版本。

## 系统架构（神经-符号）

- **DDAR**：符号引擎，组合 Deductive Database（DD，Horn 子句式演绎规则）与 Algebraic Reasoning（AR，角度/比例/距离方程）。
- **AlphaGeometry** = DDAR + 语言模型。LM 是 150M 参数的 decoder-only transformer
  （JAX + beam search），在 DDAR 卡住时提出辅助构造（auxiliary constructions）。
- 辅助构造是解竞赛几何题的关键难点，等价于"生成外部项"。

## 合成数据

- 约 10 亿随机定理前提采样，符号引擎生成 1 亿条合成定理与证明（许多证明 >200 步，4 倍于平均竞赛证明）。
- LM 在合成数据上预训练，再在需要辅助构造的约 900 万证明（约 9%）上微调。

## 结果

| Solver | IMO-AG-30 | jgex_ag_231 |
|:---|:---|:---|
| DDAR 单独 | 14 | 198 |
| AlphaGeometry | 25 | 228 |

- IMO-AG-30 解出 25/30，超过此前最好的 Wu 氏方法（10/30），接近平均金牌水平。
- 输出人类可读的证明；解出 IMO 2000 与 2015 全部几何题；发现 IMO 2004 定理的推广（找出未用前提）。

## 复现参数与硬件

- 论文结果：`BATCH_SIZE=32, BEAM_SIZE=512, DEPTH=16`
- 硬件：4 张 V100 + 250 CPU workers（满足 IMO 时限）
- 公开代码为清晰性去掉了部分优化（并行 GPU 推理、多 CPU DDAR、LM/DDAR 并行）

## 教学要点（对 16 讲）

- 神经-符号结合：LM 提供候选步骤（辅助构造），符号引擎提供确定性验证与搜索
- 与 03 讲答案验证一脉相承：这里的验证器是形式的、可靠的
- 与 AlphaProof（Lean + RL）对比：几何用符号引擎，数论/代数用形式化证明
