"""下载 CS329A 各讲的论文 PDF 到 papers/lecture-XX/。

优先用已知 arxiv ID；解析不到或 ID 为 None 时用 arxiv API 按标题搜索。
用法：python scripts/download_papers.py [--lecture 02] [--dry-run]
"""

import argparse
import os
import re
import sys
import time
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPERS = os.path.join(REPO, "papers")

# lecture -> [(slug, title, arxiv_id or None)]
LECTURES = {
    "02": [
        ("large-language-monkeys", "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling", "2407.21787"),
        ("archon", "Archon: An Architecture Search Framework for Inference-Time Techniques", "2409.15254"),
        ("snell-test-time-compute", "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters", "2408.03314"),
        ("monkey-laws", "How Do Large Language Monkeys Get Their Power (Laws)?", "2502.17578"),
    ],
    "03": [
        ("shrinking-gen-verif-gap", "Shrinking the Generation-Verification Gap with Weak Verifiers", "2506.18203"),
        ("cobbe-verifiers", "Training Verifiers to Solve Math Word Problems", "2110.14168"),
        ("lightman-verify-step-by-step", "Let's Verify Step by Step", "2305.20050"),
        ("math-shepherd", "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations", "2312.08935"),
    ],
    "04": [
        ("react", "ReAct: Synergizing Reasoning and Acting in Language Models", "2210.03629"),
        ("rlef", "RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning", "2410.02089"),
        ("constitutional-ai", "Constitutional AI: Harmlessness from AI Feedback", "2212.08073"),
    ],
    "05": [
        ("swirl", "SWiRL: Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use", "2504.04736"),
        ("lats", "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models", "2310.04406"),
        ("sprint", "SPRINT: Enabling Interleaved Planning and Parallelized Execution in Reasoning Models", "2506.05745"),
        ("adapt", "ADaPT: As-Needed Decomposition and Planning with Language Models", "2311.05772"),
        ("wider-or-deeper", "Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search", "2503.04412"),
    ],
    "06": [
        ("star", "STaR: Bootstrapping Reasoning With Reasoning", "2203.14465"),
        ("deepseek-math", "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", "2402.03300"),
        ("dapo", "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", "2503.14476"),
    ],
    "07": [
        ("adas", "Automated design of agentic systems", "2408.08435"),
        ("ai-scientist", "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery", "2408.06292"),
        ("alphaevolve", "AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms", "2506.13131"),
    ],
    "08": [
        ("alphacode", "Competition-Level Code Generation with AlphaCode", "2203.07814"),
        ("search-o1", "Search-o1: Agentic Search-Enhanced Large Reasoning Models", "2501.05366"),
    ],
    "13": [
        ("codemonkeys", "CodeMonkeys: Scaling Test-Time Compute for Software Engineering", "2501.14723"),
        ("kernelbench", "KernelBench: Can LLMs Write Efficient GPU Kernels?", "2502.10517"),
        ("agent-system-interfaces", "Improving Parallel Program Performance with LLM Optimizers via Agent-System Interfaces", "2506.03037"),
    ],
    "14": [
        ("cartridges", "Cartridges: Lightweight and general-purpose long context representations via self-study", "2506.06266"),
        ("memgpt", "MemGPT: Towards LLMs as Operating Systems", "2310.08560"),
        ("cacheblend", "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion", "2405.16444"),
    ],
    "17": [
        ("measuring-long-tasks", "Measuring AI Ability to Complete Long Tasks", "2503.14499"),
        ("gdpval", "GDPVal: Evaluating AI Model Performance on Real-World Economically Valuable Tasks", "2510.04374"),
        ("deepscholar-bench", "DeepScholar-Bench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis", "2508.20033"),
    ],
    "15": [
        ("cot", "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "2201.11903"),
        ("self-consistency", "Self-Consistency Improves Chain of Thought Reasoning in Language Models", "2203.11171"),
        ("emergent-abilities", "Emergent Abilities of Large Language Models", "2206.07682"),
    ],
    # 16 讲：AlphaGeometry 发表在 Nature（10.1038/s41586-023-06747-5）无 arxiv 版，
    # 参考材料见 alphageometry-readme.md 与 alphaproof-blog.md
    "16": [
    ],
    "19": [
        ("openvla", "OpenVLA: An Open-Source Vision-Language-Action Model", "2406.09246"),
        ("rt-2", "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control", "2307.15818"),
    ],
}


def resolve_id_by_title(title):
    """用 arxiv API 按标题搜索，返回第一个结果的 ID 或 None。"""
    query = urllib.parse.quote(f'ti:"{title}"')
    url = f"http://export.arxiv.org/api/query?search_query={query}&max_results=1"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml = resp.read().decode("utf-8")
        m = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", xml)
        if m:
            return m.group(1)
    except Exception as exc:
        print(f"  标题搜索失败 {title}: {exc}")
    return None


def slugify(slug):
    return re.sub(r"[^a-z0-9-]", "", slug.lower())


def download_pdf(arxiv_id, dest):
    """下载 arxiv 论文 PDF。arxiv_id 可能是 'vX' 后缀。"""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "self-improving-agent-notebook/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 10000:  # 小于 10KB 通常不是有效 PDF
            print(f"  !! 文件过小（{len(data)} bytes），可能不是 PDF: {arxiv_id}")
            return False
        with open(dest, "wb") as f:
            f.write(data)
        print(f"  OK  {os.path.basename(dest)}  ({len(data)//1024} KB)  <- arxiv:{arxiv_id}")
        return True
    except Exception as exc:
        print(f"  !! 下载失败 {arxiv_id}: {exc}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lecture", help="只下载指定讲，如 02")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划不下载")
    args = ap.parse_args()

    lectures = {k: v for k, v in LECTURES.items() if not args.lecture or k == args.lecture}
    if not lectures:
        print("没有匹配的讲座")
        sys.exit(1)

    total = fail = 0
    for lecture, papers in sorted(lectures.items()):
        dest_dir = os.path.join(PAPERS, f"lecture-{lecture}")
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\n=== lecture-{lecture} ===")
        for slug, title, arxiv_id in papers:
            total += 1
            if args.dry_run:
                print(f"  [dry] {slug}: {arxiv_id or '标题搜索'}")
                continue
            dest = os.path.join(dest_dir, f"{slugify(slug)}.pdf")
            if os.path.exists(dest) and os.path.getsize(dest) > 10000:
                print(f"  跳过（已存在）{os.path.basename(dest)}")
                continue
            if not arxiv_id:
                arxiv_id = resolve_id_by_title(title)
                time.sleep(1)
            if not arxiv_id:
                print(f"  !! 无法解析 ID: {title}")
                fail += 1
                continue
            if not download_pdf(arxiv_id, dest):
                fail += 1
            time.sleep(1.2)  # arxiv 限流

    print(f"\n完成：{total - fail}/{total} 篇论文下载成功" + (f"，{fail} 篇失败" if fail else ""))


if __name__ == "__main__":
    main()
