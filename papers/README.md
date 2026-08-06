# papers/ 目录说明

每讲一个子目录 `lecture-XX/`，包含论文 PDF 与研读笔记：

- **NOTES.md** — 研读笔记（已纳入版本控制）。每份笔记包含：论文核心思想、关键公式/算法、
  关键实验数字、教学主线（想象 Stanford 老师怎么教）、代码演示点子、作业点子、参考资料。
  notebook 由这些笔记驱动编写。
- **\*.pdf** — 论文原文（未纳入版本控制，共约 150MB）。用脚本一键重新下载：

```bash
python scripts/download_papers.py           # 下载全部
python scripts/download_papers.py --lecture 02   # 只下载某一讲
python scripts/download_papers.py --dry-run      # 只查看计划
```

- **模板**：`NOTES_TEMPLATE.md` 规定了研读笔记的结构。

## 说明

- 个别讲座无 arxiv 论文（如 AlphaGeometry 发在 Nature、AlphaProof 是 DeepMind 博客、
  嘉宾讲座无指定论文），参考材料以 `.md` 文件形式入库（如 `lecture-13/alphageometry-readme.md`）。
