import { useEffect, useState } from 'react'
import {
  BookOpen, ArrowRight, Check, Layers, Cpu, Star,
  Monitor, Languages, ChevronRight, CodeXml, Rocket, Sparkles, Menu,
  Mail, X, ExternalLink,
} from 'lucide-react'
import { GITHUB_OWNER, GITHUB_REPO } from '../config.js'
import { PATH_STEPS, RUNNABLE_NOTEBOOKS } from '../data/sidebar.js'

const GITHUB_STARS_CACHE_KEY = `github-stars:${GITHUB_OWNER}/${GITHUB_REPO}`

function formatStarCount(count) {
  if (typeof count !== 'number') return '--'
  return count.toLocaleString('en-US')
}

const SECTION_STYLES = {
  foundation: { bg: 'from-[#f5f3ff] to-[#ede9fe]', tag: 'bg-violet-50 text-violet-600 border-violet-200/50', nameZh: '基础与方法', nameEn: 'Foundation', accent: 'violet', iconBg: 'bg-violet-100 text-violet-600 border-violet-200/50', pathBorder: 'border-l-violet-400' },
  training: { bg: 'from-[#eef2ff] to-[#e0e7ff]', tag: 'bg-indigo-50 text-indigo-600 border-indigo-200/50', nameZh: '训练与进化', nameEn: 'Training & Evolution', accent: 'indigo', iconBg: 'bg-indigo-100 text-indigo-600 border-indigo-200/50', pathBorder: 'border-l-indigo-400' },
  engineering: { bg: 'from-[#faf5ff] to-[#f3e8ff]', tag: 'bg-purple-50 text-purple-600 border-purple-200/50', nameZh: '智能体工程', nameEn: 'Engineering', accent: 'purple', iconBg: 'bg-purple-100 text-purple-600 border-purple-200/50', pathBorder: 'border-l-purple-400' },
  frontiers: { bg: 'from-[#fdf4ff] to-[#fae8ff]', tag: 'bg-fuchsia-50 text-fuchsia-600 border-fuchsia-200/50', nameZh: '前沿', nameEn: 'Frontiers', accent: 'fuchsia', iconBg: 'bg-fuchsia-100 text-fuchsia-600 border-fuchsia-200/50', pathBorder: 'border-l-fuchsia-400' },
}

const PATH_STEP_STYLES = [
  { numBg: 'bg-[var(--bg-input)] text-[var(--text-secondary)]' },
  { numBg: 'bg-[var(--bg-input)] text-[var(--text-secondary)]' },
  { numBg: 'bg-[var(--bg-input)] text-[var(--text-secondary)]' },
  { numBg: 'bg-[var(--bg-input)] text-[var(--text-secondary)]' },
]

const NOTEBOOK_BG = {
  'nb-1': 'from-[#f5f3ff] to-[#ede9fe]',     // 01 course-overview — lavender
  'nb-2': 'from-[#ede9fe] to-[#ddd6fe]',     // 02 test-time — violet
  'nb-3': 'from-[#eef2ff] to-[#c7d2fe]',     // 04 ReAct — indigo
  'nb-4': 'from-[#f5f3ff] to-[#c4b5fd]',     // 05 planning — light violet
  'nb-5': 'from-[#e0e7ff] to-[#a5b4fc]',     // 06 RL — indigo
  'nb-6': 'from-[#faf5ff] to-[#e9d5ff]',     // 07 evolution — purple
  'nb-7': 'from-[#ede9fe] to-[#c4b5fd]',     // 14 memory — lavender
  'nb-8': 'from-[#f5f3ff] to-[#ddd6fe]',     // 17 eval — soft violet
  'nb-9': 'from-[#fdf4ff] to-[#f0abfc]',     // 15 reasoning — fuchsia
  'nb-10': 'from-[#faf5ff] to-[#d8b4fe]',    // 16 proof — bright violet
}

const NOTEBOOK_SVGS = {
  'nb-1': ( // 01 course-overview — agent loop
    <svg width="60" height="60" viewBox="0 0 60 60" className="opacity-80">
      <circle cx="30" cy="30" r="7" fill="#7c3aed" />
      <text x="26" y="33" fontSize="6" fill="white" fontWeight="bold">LLM</text>
      <circle cx="10" cy="10" r="4" fill="#a78bfa" opacity="0.85" />
      <text x="7" y="13" fontSize="5" fill="white">S</text>
      <circle cx="50" cy="10" r="4" fill="#c4b5fd" opacity="0.85" />
      <text x="47" y="13" fontSize="5" fill="white">A</text>
      <circle cx="50" cy="50" r="4" fill="#c4b5fd" opacity="0.85" />
      <text x="47" y="53" fontSize="5" fill="white">O</text>
      <circle cx="10" cy="50" r="4" fill="#a78bfa" opacity="0.85" />
      <text x="7" y="53" fontSize="5" fill="white">T</text>
      <line x1="14" y1="10" x2="43" y2="10" stroke="#7c3aed" strokeWidth="1" opacity="0.5" />
      <line x1="50" y1="14" x2="50" y2="46" stroke="#7c3aed" strokeWidth="1" opacity="0.5" />
      <line x1="43" y1="50" x2="14" y2="50" stroke="#7c3aed" strokeWidth="1" opacity="0.5" />
      <line x1="10" y1="46" x2="10" y2="14" stroke="#7c3aed" strokeWidth="1" opacity="0.5" />
    </svg>
  ),
  'nb-2': ( // 02 test-time compute — repeated sampling
    <svg width="60" height="60" viewBox="0 0 60 60" className="opacity-80">
      <rect x="4" y="14" width="16" height="16" rx="3" fill="#a78bfa" opacity="0.55" />
      <text x="8" y="25" fontSize="6" fill="white">×1</text>
      <rect x="22" y="14" width="16" height="16" rx="3" fill="#7c3aed" opacity="0.75" />
      <text x="26" y="25" fontSize="6" fill="white">×N</text>
      <rect x="40" y="14" width="16" height="16" rx="3" fill="#7c3aed" opacity="0.95" />
      <text x="45" y="25" fontSize="6" fill="white">✓</text>
      <line x1="30" y1="30" x2="30" y2="38" stroke="#a78bfa" strokeWidth="1.5" />
      <rect x="14" y="38" width="32" height="12" rx="3" fill="#ede9fe" />
      <text x="20" y="47" fontSize="6" fill="#6d28d9">投票 / best-of-n</text>
    </svg>
  ),
  'nb-3': ( // 04 ReAct — thought/action/observation loop
    <svg width="60" height="60" viewBox="0 0 60 60" className="opacity-80">
      <rect x="4" y="6" width="24" height="12" rx="3" fill="#ede9fe" />
      <text x="8" y="15" fontSize="6" fill="#6d28d9">Thought</text>
      <line x1="28" y1="12" x2="32" y2="12" stroke="#7c3aed" strokeWidth="1.5" />
      <rect x="32" y="6" width="24" height="12" rx="3" fill="#7c3aed" opacity="0.85" />
      <text x="37" y="15" fontSize="6" fill="white">Action</text>
      <line x1="44" y1="18" x2="44" y2="22" stroke="#7c3aed" strokeWidth="1.5" />
      <rect x="32" y="22" width="24" height="12" rx="3" fill="#a78bfa" opacity="0.75" />
      <text x="38" y="31" fontSize="6" fill="white">Env</text>
      <line x1="32" y1="28" x2="4" y2="28" stroke="#7c3aed" strokeWidth="1.5" />
      <line x1="4" y1="28" x2="4" y2="18" stroke="#7c3aed" strokeWidth="1.5" />
      <rect x="4" y="40" width="28" height="12" rx="3" fill="#c4b5fd" opacity="0.6" />
      <text x="8" y="49" fontSize="6" fill="white">Observation</text>
    </svg>
  ),
  'nb-4': ( // 05 planning — tree search
    <svg width="60" height="60" viewBox="0 0 60 60" className="opacity-80">
      <circle cx="30" cy="8" r="4" fill="#7c3aed" />
      <line x1="28" y1="12" x2="14" y2="24" stroke="#a78bfa" strokeWidth="1.5" />
      <line x1="32" y1="12" x2="30" y2="24" stroke="#7c3aed" strokeWidth="1.5" />
      <line x1="32" y1="12" x2="46" y2="24" stroke="#c4b5fd" strokeWidth="1.5" />
      <circle cx="12" cy="28" r="3" fill="#a78bfa" opacity="0.85" />
      <circle cx="30" cy="28" r="3.5" fill="#7c3aed" opacity="0.95" />
      <circle cx="48" cy="28" r="3" fill="#c4b5fd" opacity="0.85" />
      <line x1="12" y1="31" x2="18" y2="42" stroke="#a78bfa" strokeWidth="1" />
      <line x1="30" y1="31" x2="30" y2="42" stroke="#7c3aed" strokeWidth="1.5" />
      <line x1="48" y1="31" x2="42" y2="42" stroke="#c4b5fd" strokeWidth="1" />
      <circle cx="20" cy="46" r="3" fill="#a78bfa" opacity="0.6" />
      <circle cx="30" cy="46" r="3" fill="#7c3aed" opacity="0.95" />
      <circle cx="40" cy="46" r="3" fill="#c4b5fd" opacity="0.6" />
    </svg>
  ),
  'nb-5': ( // 06 RL — agent/env reward loop
    <svg width="60" height="60" viewBox="0 0 60 60" className="opacity-80">
      <rect x="6" y="8" width="20" height="14" rx="3" fill="#7c3aed" opacity="0.85" />
      <text x="9" y="18" fontSize="6" fill="white">Agent</text>
      <rect x="34" y="8" width="20" height="14" rx="3" fill="#a78bfa" opacity="0.75" />
      <text x="37" y="18" fontSize="6" fill="white">Env</text>
      <line x1="26" y1="12" x2="34" y2="12" stroke="#7c3aed" strokeWidth="1.5" />
      <text x="27" y="11" fontSize="5" fill="#7c3aed">a</text>
      <line x1="34" y1="18" x2="26" y2="18" stroke="#a78bfa" strokeWidth="1.5" strokeDasharray="2 2" />
      <text x="27" y="22" fontSize="5" fill="#a78bfa">s,r</text>
      <rect x="12" y="32" width="36" height="14" rx="3" fill="#ede9fe" />
      <text x="16" y="42" fontSize="6" fill="#6d28d9">reward → 更新策略</text>
    </svg>
  ),
  'nb-6': ( // 07 evolution — fitness ascending
    <svg width="60" height="60" viewBox="0 0 60 60" className="opacity-80">
      <rect x="6" y="40" width="10" height="14" rx="2" fill="#c4b5fd" opacity="0.7" />
      <rect x="18" y="32" width="10" height="22" rx="2" fill="#a78bfa" opacity="0.8" />
      <rect x="30" y="22" width="10" height="32" rx="2" fill="#7c3aed" opacity="0.85" />
      <rect x="42" y="12" width="10" height="42" rx="2" fill="#7c3aed" />
      <path d="M8 6 L50 6" stroke="#a78bfa" strokeWidth="1" strokeDasharray="2 2" />
      <text x="6" y="56" fontSize="6" fill="#7c3aed" opacity="0.7">适应度 ↑</text>
    </svg>
  ),
  'nb-7': ( // 14 memory — hierarchical layers
    <svg width="60" height="60" viewBox="0 0 60 60" className="opacity-80">
      <rect x="8" y="4" width="44" height="12" rx="2" fill="#7c3aed" opacity="0.9" />
      <text x="16" y="13" fontSize="6" fill="white">Main</text>
      <rect x="8" y="18" width="44" height="12" rx="2" fill="#a78bfa" opacity="0.8" />
      <text x="16" y="27" fontSize="6" fill="white">Ext</text>
      <rect x="8" y="32" width="44" height="12" rx="2" fill="#c4b5fd" opacity="0.7" />
      <rect x="8" y="46" width="44" height="12" rx="2" fill="#ede9fe" />
      <text x="12" y="55" fontSize="6" fill="#6d28d9">换出 / 召回</text>
    </svg>
  ),
  'nb-8': ( // 17 evaluation — gauge
    <svg width="60" height="60" viewBox="0 0 60 60" className="opacity-80">
      <path d="M8 44 A26 26 0 1 1 52 44" stroke="#ede9fe" strokeWidth="5" fill="none" />
      <path d="M8 44 A26 26 0 1 1 30 18" stroke="#7c3aed" strokeWidth="5" fill="none" />
      <line x1="30" y1="30" x2="42" y2="18" stroke="#7c3aed" strokeWidth="2" strokeLinecap="round" />
      <circle cx="30" cy="30" r="3" fill="#7c3aed" />
      <text x="22" y="54" fontSize="6" fill="#7c3aed">pass@1</text>
    </svg>
  ),
  'nb-9': ( // 15 reasoning — CoT bubbles
    <svg width="60" height="60" viewBox="0 0 60 60" className="opacity-80">
      <rect x="4" y="6" width="24" height="10" rx="5" fill="#7c3aed" opacity="0.35" />
      <text x="7" y="14" fontSize="7" fill="#7c3aed" opacity="0.8">步骤1</text>
      <rect x="32" y="6" width="24" height="10" rx="5" fill="#7c3aed" opacity="0.55" />
      <text x="35" y="14" fontSize="7" fill="white" opacity="0.9">步骤2</text>
      <rect x="4" y="22" width="24" height="10" rx="5" fill="#7c3aed" opacity="0.75" />
      <text x="7" y="30" fontSize="7" fill="white">步骤3</text>
      <rect x="32" y="22" width="24" height="10" rx="5" fill="#7c3aed" opacity="0.9" />
      <text x="35" y="30" fontSize="7" fill="white">步骤4</text>
      <line x1="28" y1="11" x2="32" y2="11" stroke="#a78bfa" strokeWidth="1" />
      <line x1="16" y1="16" x2="16" y2="22" stroke="#a78bfa" strokeWidth="1" />
      <line x1="44" y1="16" x2="44" y2="22" stroke="#a78bfa" strokeWidth="1" />
      <rect x="14" y="38" width="32" height="14" rx="4" fill="#a78bfa" />
      <text x="24" y="48" fontSize="7" fill="white" fontWeight="bold">答案</text>
    </svg>
  ),
  'nb-10': ( // 16 math proof — theorem + checkmark
    <svg width="60" height="60" viewBox="0 0 60 60" className="opacity-80">
      <rect x="6" y="6" width="48" height="20" rx="3" fill="#ede9fe" />
      <text x="10" y="15" fontSize="6" fill="#6d28d9">定理：偶+偶=偶</text>
      <rect x="6" y="30" width="48" height="20" rx="3" fill="#7c3aed" opacity="0.85" />
      <text x="10" y="39" fontSize="6" fill="white">证明步骤 ...</text>
      <path d="M44 37 l4 4 l8 -8" stroke="#fff" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
}

function ReaderLetterModal({ isOpen, onClose, lang }) {
  useEffect(() => {
    if (!isOpen) return
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  const isZh = lang === 'zh'
  const title = isZh ? '给读者的一封信' : 'A Letter to Readers'
  const eyebrow = isZh ? '开始之前，先看清这条路会带你去哪' : 'Before you start, see where this path leads'
  const intro = isZh
    ? '亲爱的读者：你好。在正式开始之前，我想简单聊聊，做这套教程的初衷。'
    : 'This tutorial answers one concrete question: how does an agent grow from loops, verifiers, planning, RL, and memory instead of framework magic?'
  const sections = isZh ? [
    {
      heading: null,
      items: [
        '现在想学 Agent 的人越来越多，大家每天都会接触到各式各样的名词：ReAct、验证器、GRPO、树搜索、记忆分页。看得多、搜得多，但大多只是零散的碎片，心里始终差一条完整的主线。市面上关于 LLM 的教程很多，真正把 Agent 从底层循环、验证、规划、训练到评测完整串起来的却很少。',
      ],
    },
    {
      heading: null,
      items: [
        '这也让很多人卡在同一个尴尬的状态：会调模型 API、会套 Agent 框架，却始终搞不懂一个 Agent 究竟如何从零搭建、如何被验证、如何通过强化学习变强、又如何被可靠地评测。',
      ],
    },
    {
      heading: null,
      items: [
        '我会和你一起走过一条完整的 Agent 知识链路：既吃透底层机制，也看懂当下前沿。从 ReAct 循环、Test-time Compute、验证器开始，到树搜索、GRPO 强化学习、开放进化、记忆与评测，再到数学证明、机器人等前沿方向。不求速成，只求通透。',
      ],
    },
    {
      heading: null,
      items: [
        '学完这套内容，你不只会调用框架，而是能亲手实现 Agent 的每一个核心算法：自己写 ReAct 循环、自己训练验证器、自己跑 GRPO 更新、自己搭评测 Harness。',
      ],
    },
    {
      heading: null,
      items: [
        '非常感谢你愿意翻开这篇前言，愿意沉下心深耕底层、吃透原理。也许，未来设计出下一代 Agent 架构的人会是你，让机器真正拥有自主思考能力的突破，也终将由你带来。',
      ],
    },
    {
      heading: null,
      items: [
        'Agent 终将重塑人类与技术交互的方式，站在这场变革的浪潮里，在历史的面前，比起害怕，不如鼓起勇气尽情享受，祝你与我在探索的路途上好运！',
      ],
    },
  ] : [
    {
      heading: 'What you will walk through',
      items: [
        'Start from the agent loop: thought, action, observation, and tool use.',
        'Hand-build test-time compute scaling, verifiers, and tree-search planning.',
        'Train for reasoning with STaR bootstrapping and GRPO.',
        'Open-ended evolution, search agents, and memory systems.',
        'Evaluate long-horizon agents and turn frontier papers into runnable examples.',
      ],
    },
    {
      heading: 'What you will have at the end',
      items: [
        'A complete from-zero agent path: loops, verification, planning, RL, memory, evaluation.',
        'Working implementations of every core algorithm, not framework calls.',
        'A mental map for new agent papers: whether they change loops, verification, training, or evaluation.',
      ],
    },
    {
      heading: 'How to study it',
      items: [
        'Read the intuition first, verify with small numbers, then run the code and inspect the output.',
        'When stuck, ask what problem the component solves and what shapes flow in and out.',
        'Use AI for hints and direction checks, but still edit code, run experiments, and observe behavior yourself.',
      ],
    },
  ]

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-label={title}>
      <div className="modal-backdrop" onClick={onClose} />
      <div className="modal-card" style={{ maxWidth: 760, maxHeight: '86vh' }}>
        <div className="modal-header">
          <div className="min-w-0">
            <div className="text-[10px] sm:text-[11px] font-extrabold tracking-[0.12em] uppercase text-violet-600 mb-1">
              {eyebrow}
            </div>
            <h2>{title}</h2>
          </div>
          <button className="modal-close" onClick={onClose} aria-label={isZh ? '关闭' : 'Close'}>
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="modal-body">
          <div className="space-y-6 text-[13px] sm:text-sm leading-7 text-[var(--text-secondary)]">
            <p className="text-base sm:text-lg leading-8 font-bold text-[var(--text-primary)]">
              {intro}
            </p>

            {sections.map((section, idx) => (
              <section key={section.heading || idx} className="space-y-3">
                {section.heading && (
                  <h3 className="text-[15px] sm:text-base font-extrabold text-[var(--text-primary)]">
                    {section.heading}
                  </h3>
                )}
                <div className="space-y-2.5">
                  {section.items.map((item) => (
                    <div key={item} className="flex gap-2.5">
                      {section.heading && <Check className="w-4 h-4 mt-1 text-violet-600 shrink-0" />}
                      <p>{item}</p>
                    </div>
                  ))}
                </div>
              </section>
            ))}

            {!isZh && (
              <div className="rounded-2xl border border-violet-200/70 bg-violet-50/80 p-4 text-violet-900">
                <p className="font-bold">
                  In one sentence: this is not an API tour, but a path to open up, modify, train, and evaluate an LLM system from the inside.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Welcome({ catalog, lang, onLanguageChange, onSelect, onStartTour }) {
  const [starCount, setStarCount] = useState(null)
  const [isLetterOpen, setIsLetterOpen] = useState(false)
  const catalogById = new Map(catalog.map(item => [item.id, item]))
  const notebookCount = catalog.length

  useEffect(() => {
    let cancelled = false

    const cachedCount = window.localStorage.getItem(GITHUB_STARS_CACHE_KEY)
    if (cachedCount !== null) {
      const parsedCount = Number(cachedCount)
      if (Number.isFinite(parsedCount)) {
        setStarCount(parsedCount)
      }
    }

    fetch(`https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`GitHub API status: ${response.status}`)
        }
        return response.json()
      })
      .then((repo) => {
        if (cancelled || typeof repo.stargazers_count !== 'number') return
        setStarCount(repo.stargazers_count)
        window.localStorage.setItem(GITHUB_STARS_CACHE_KEY, String(repo.stargazers_count))
      })
      .catch(() => {
        // GitHub API 偶尔会被限流；页面保留缓存或占位即可。
      })

    return () => {
      cancelled = true
    }
  }, [])

  const t = lang === 'zh' ? {
    bannerBadge: '从零构建自进化 Agent',
    bannerTitleLine1: '亲手构建自进化 Agent，',
    bannerTitleLine2: '从原理到自我进化。',
    bannerDesc: 'Self-Improving Agent Notebook 通过交互式 Notebook，带你从零实现会自我进化的 Agent：循环、验证、规划、强化学习与记忆。',
    startBtn: '开始学习',
    browsePath: '浏览学习路径',
    readerLetter: '给读者的一封信',
    readerLetterDesc: '给读者们的一封信',
    runHintTitle: '每篇 Notebook 都可以直接运行',
    runHintDesc: '进入章节后，点击顶部按钮即可在 ModelScope 或 Colab 中打开，无需本地配置。',
    check1: '交互式 Notebook', check2: '逐步构建知识', check3: '代码即文档', check4: '实验即理解',
    learningPathTitle: '学习路径',
    learningPathSub: '科学规划，逐步深入',
    viewAllPaths: '查看全部路径',
    runnableNotebooksTitle: '可在线运行的 Notebook',
    runnableNotebooksSub: '精选推荐，点击即可开始学习',
    allNotebooksLink: '全部 Notebook',
    footerQuote: '"构建是最好的学习方式。" — Self-Improving Agent Notebook',
    feature1: '可运行 Notebook', feature1d: '浏览器中渲染，无需配置',
    feature2: '从原理到实践', feature2d: '由浅入深，循序渐进',
    feature3: 'Agent 流程可视化', feature3d: '循环、搜索、反馈全程图解',
    feature4: '论文驱动学习', feature4d: '每讲精读论文，产出研读笔记',
    feature5: '面向未来', feature5d: '紧跟前沿，持续更新',
    sponsorsTitle: '合作方',
    sponsorsSub: '本项目由以下合作方提供计算资源与技术支持',
    amdDesc: 'GPU 计算资源支持',
    modelscopeDesc: '模型托管与开源支持',
  } : {
    bannerBadge: 'Future-Ready LLM Learning Method',
    bannerTitleLine1: 'Build Self-Improving Agents,',
    bannerTitleLine2: 'From Theory to Self-Evolution.',
    bannerDesc: 'Self-Improving Agent Notebook guides you into self-improving agents — loops, verifiers, planning, RL, and memory — via interactive Notebooks.',
    startBtn: 'Start Learning',
    browsePath: 'Browse Pathways',
    readerLetter: 'A Letter to Readers',
    readerLetterDesc: 'See the full route, final builds, and skills you will gain',
    runHintTitle: 'Run every Notebook online',
    runHintDesc: 'Open a chapter, then use the top buttons to launch it in ModelScope or Colab. No local setup needed.',
    check1: 'Interactive Notebook', check2: 'Step-by-step Knowledge', check3: 'Code as Document', check4: 'Understand via Experiments',
    learningPathTitle: 'Learning Paths',
    learningPathSub: 'Structured curriculum, progress step-by-step',
    viewAllPaths: 'View All Paths',
    runnableNotebooksTitle: 'Runnable Online Notebooks',
    runnableNotebooksSub: 'Selected recommendations, click to start learning',
    allNotebooksLink: 'All Notebooks',
    footerQuote: '"The best way to learn is to build." — Self-Improving Agent Notebook',
    feature1: 'Runnable Notebooks', feature1d: 'Rendered in browser, no setup needed',
    feature2: 'Theory to Practice', feature2d: 'Step-by-step progress',
    feature3: 'Agent Flow Visuals', feature3d: 'Loops, search, and feedback visualized',
    feature4: 'Paper-Driven', feature4d: 'Read each paper, write study notes',
    feature5: 'Future Oriented', feature5d: 'Up-to-date documentation',
    sponsorsTitle: 'Partners',
    sponsorsSub: 'Compute resources and technical support provided by our partners',
    amdDesc: 'GPU Compute Resources',
    modelscopeDesc: 'Model Hub & Open Source',
  }

  const scrollToPath = () => {
    const el = document.getElementById('learning-path-section')
    if (el) el.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 sm:space-y-8 max-w-7xl w-full mx-auto">

        {/* HERO BANNER */}
        <section className="hero rounded-3xl p-6 md:p-10 relative overflow-hidden shadow-sm border bg-gradient-to-br from-[#ede9fe]/90 via-[#f5f3ff] to-[#faf8ff] border-violet-100/50">
          <div className="absolute top-[-20%] right-[-10%] w-[350px] h-[350px] rounded-full bg-violet-400/10 blur-[80px] pointer-events-none"></div>
          <div className="absolute bottom-[-10%] left-[20%] w-[250px] h-[250px] rounded-full bg-indigo-300/10 blur-[60px] pointer-events-none"></div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center relative z-10">
            <div className="lg:col-span-7 space-y-6">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-bold bg-[#f1ecfb] text-violet-600 border border-violet-200/50 shadow-sm">
                <span className="w-2.5 h-2.5 rounded-full bg-violet-600 animate-pulse"></span>
                <span>{t.bannerBadge}</span>
              </div>

              <h1 className="text-3xl sm:text-4xl md:text-[46px] font-extrabold tracking-tight text-slate-900 leading-[1.2]">
                {t.bannerTitleLine1}
                <br className="hidden md:inline" />
                <span className="bg-gradient-to-r from-violet-600 to-indigo-600 bg-clip-text text-transparent">
                  {t.bannerTitleLine2}
                </span>
              </h1>

              <p className="text-xs sm:text-sm md:text-base leading-relaxed text-[var(--text-muted)] max-w-xl">
                {t.bannerDesc}
              </p>

              <div className="flex flex-wrap items-center gap-3 sm:gap-4 pt-2">
                <button
                  onClick={() => onSelect('01-course-overview')}
                  className="h-10 sm:h-12 px-5 sm:px-6 rounded-full bg-violet-600 hover:bg-violet-700 text-white font-bold text-xs sm:text-sm shadow-lg shadow-violet-500/10 hover:shadow-violet-500/20 active:scale-[0.98] transition-all flex items-center gap-2"
                >
                  <span>{t.startBtn}</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
                <button
                  onClick={scrollToPath}
                  className="h-10 sm:h-12 px-5 sm:px-6 rounded-full bg-white hover:bg-[var(--bg-input)] text-slate-700 border border-[var(--border-light)]/90 font-bold text-xs sm:text-sm active:scale-[0.98] transition-all"
                >
                  {t.browsePath}
                </button>
                <button
                  onClick={() => setIsLetterOpen(true)}
                  className="group h-10 sm:h-12 px-3 sm:px-4 rounded-xl border border-violet-200/80 bg-white/85 hover:bg-white shadow-sm hover:shadow-md transition-all active:scale-[0.99] flex items-center gap-2"
                >
                  <Mail className="w-4 h-4 text-violet-600 shrink-0" />
                  <span className="text-xs sm:text-sm font-extrabold text-slate-900 whitespace-nowrap">
                    {t.readerLetter}
                  </span>
                  <ChevronRight className="w-4 h-4 text-violet-600 group-hover:translate-x-0.5 transition-transform shrink-0" />
                </button>
              </div>

              <div className="flex items-start gap-3 max-w-xl rounded-2xl border border-violet-200/80 bg-white/75 px-3.5 py-3 shadow-sm">
                <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-violet-50 text-violet-600">
                  <ExternalLink className="h-4 w-4" strokeWidth={2.2} />
                </div>
                <div className="min-w-0">
                  <p className="text-xs sm:text-sm font-extrabold text-slate-900 leading-snug">
                    {t.runHintTitle}
                  </p>
                  <p className="mt-1 text-[10px] sm:text-xs leading-relaxed text-[var(--text-muted)]">
                    {t.runHintDesc}
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-x-4 sm:gap-x-6 gap-y-2 sm:gap-y-3 border-t border-[var(--border-light)]/50 pt-4 sm:pt-5 max-w-lg select-none">
                {[t.check1, t.check2, t.check3, t.check4].map((feature, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-[10px] sm:text-xs font-bold text-[var(--text-muted)]">
                    <div className="w-4 h-4 sm:w-4.5 sm:h-4.5 rounded-full bg-violet-50 border border-violet-200 flex items-center justify-center text-violet-600 shrink-0">
                      <Check className="w-2.5 h-2.5 sm:w-3 sm:h-3 stroke-[2.5]" />
                    </div>
                    <span className="truncate">{feature}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Right column: floating glass windows — hidden on small screens to prevent overflow */}
            <div className="hidden lg:flex lg:col-span-5 relative min-h-[300px] items-center justify-center select-none">
              <div className="absolute w-[280px] h-[190px] rounded-2xl glass-effect shadow-xl p-4 border border-white/60 left-[5%] top-[10%] animate-float-1 z-10 overflow-hidden">
                <div className="flex items-center justify-between mb-3 border-b border-[var(--border-light)]/40 pb-1.5">
                  <div className="flex gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-red-400"></span>
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
                    <span className="w-2.5 h-2.5 rounded-full bg-green-400"></span>
                  </div>
                  <span className="text-[9px] font-mono font-bold text-violet-600 bg-violet-50 px-2 py-0.5 rounded border border-violet-100">agent.py</span>
                </div>
                <pre className="font-mono text-[10px] text-[var(--text-secondary)] space-y-0.5">
                  <div><span className="text-purple-600 font-bold">while</span> not done:</div>
                  <div>    <span className="text-violet-600 font-bold">obs</span> = env.step(action)</div>
                  <div>    <span className="text-violet-600 font-bold">thought</span> = llm.reason(history)</div>
                  <div>    action = parse(thought)</div>
                  <div>    history.append(obs)</div>
                  <div className="text-slate-300"># ReAct 循环：思考 → 行动 → 观察</div>
                </pre>
              </div>

              <div className="absolute w-[260px] h-[170px] rounded-2xl glass-effect shadow-lg p-3.5 border border-white/60 right-0 bottom-[5%] animate-float-2 z-0">
                <div className="flex justify-between items-center text-[9px] font-semibold text-[var(--text-muted)] mb-2">
                  <span className="font-bold flex items-center gap-1">
                    <Sparkles className="w-3 h-3 text-purple-500" />
                    Attention Map
                  </span>
                  <span>Head 1</span>
                </div>
                <div className="grid grid-cols-4 gap-1.5 pt-0.5">
                  {[
                    "bg-violet-600/30 border border-violet-600/10", "bg-purple-600/70 border border-purple-600/10", "bg-indigo-600/10 border border-indigo-600/10", "bg-violet-600/20 border border-violet-600/10",
                    "bg-violet-600/10 border border-violet-600/10", "bg-indigo-600/40 border border-indigo-600/10", "bg-purple-600/90 border border-purple-600/10", "bg-purple-600/15 border border-purple-600/10",
                    "bg-indigo-600/60 border border-indigo-600/10", "bg-purple-600/20 border border-purple-600/10", "bg-violet-600/10 border border-violet-600/10", "bg-indigo-600/80 border border-indigo-600/10",
                    "bg-purple-600/15", "bg-indigo-600/10", "bg-violet-600/50", "bg-purple-600/40"
                  ].map((cls, j) => (
                    <div key={j} className={`h-5 rounded-md ${cls}`}></div>
                  ))}
                </div>
              </div>

              <div className="absolute top-[5%] right-[25%] bg-white/80 p-2.5 rounded-full shadow-md animate-float-3 border border-white/50 z-10">
                <CodeXml className="w-4.5 h-4.5 text-violet-600" />
              </div>
              <div className="absolute bottom-[20%] left-[20%] bg-white/85 p-2 rounded-xl shadow-md animate-float-1 border border-white/50 z-20">
                <Rocket className="w-4 h-4 text-purple-600" />
              </div>
            </div>
          </div>
        </section>

        {/* STATS BAR */}
        <section className="stats grid grid-cols-[repeat(auto-fit,minmax(min(100%,190px),1fr))] bg-[var(--bg-sidebar)] rounded-2xl border border-black/10 shadow-sm overflow-hidden">
          <div className="p-4 sm:p-5 md:p-6 flex items-center gap-3 sm:gap-4 hover:bg-[var(--bg-input)]/45 transition-colors">
            <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-[var(--bg-input)] text-[var(--text-muted)] flex items-center justify-center border border-[var(--border-light)]/50 shrink-0">
              <BookOpen className="w-5 h-5 sm:w-6 sm:h-6 stroke-[1.5]" />
            </div>
            <div className="space-y-0.5 min-w-0">
              <div className="text-lg sm:text-xl md:text-2xl font-bold text-[var(--text-primary)] tracking-tight leading-none">{notebookCount}</div>
              <div className="text-[10px] sm:text-[11px] font-medium text-[var(--text-label)] leading-snug break-words">{lang === 'zh' ? '可在线运行 Notebook' : 'Runnable Online Notebooks'}</div>
            </div>
          </div>
          <div className="p-4 sm:p-5 md:p-6 flex items-center gap-3 sm:gap-4 hover:bg-[var(--bg-input)]/45 transition-colors">
            <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-[var(--bg-input)] text-[var(--text-muted)] flex items-center justify-center border border-[var(--border-light)]/50 shrink-0">
              <Layers className="w-5 h-5 sm:w-6 sm:h-6 stroke-[1.5]" />
            </div>
            <div className="space-y-0.5 min-w-0">
              <div className="text-lg sm:text-xl md:text-2xl font-bold text-[var(--text-primary)] tracking-tight leading-none">4</div>
              <div className="text-[10px] sm:text-[11px] font-medium text-[var(--text-label)] leading-snug break-words">{lang === 'zh' ? '学习路径' : 'Learning Paths'}</div>
            </div>
          </div>
          <div className="p-4 sm:p-5 md:p-6 flex items-center gap-3 sm:gap-4 hover:bg-[var(--bg-input)]/45 transition-colors">
            <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-[var(--bg-input)] text-[var(--text-muted)] flex items-center justify-center border border-[var(--border-light)]/50 shrink-0">
              <Cpu className="w-5 h-5 sm:w-6 sm:h-6 stroke-[1.5]" />
            </div>
            <div className="space-y-0.5 min-w-0">
              <div className="text-lg sm:text-xl md:text-2xl font-bold text-[var(--text-primary)] tracking-tight leading-none">20+</div>
              <div className="text-[10px] sm:text-[11px] font-medium text-[var(--text-label)] leading-snug break-words">{lang === 'zh' ? '核心模块' : 'Core Modules'}</div>
            </div>
          </div>
          <div className="p-4 sm:p-5 md:p-6 flex items-center gap-3 sm:gap-4 hover:bg-[var(--bg-input)]/45 transition-colors">
            <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-[var(--bg-input)] text-[var(--text-muted)] flex items-center justify-center border border-[var(--border-light)]/50 shrink-0">
              <Star className="w-5 h-5 sm:w-6 sm:h-6 stroke-[1.5]" />
            </div>
            <div className="space-y-0.5 min-w-0">
              <div className="text-lg sm:text-xl md:text-2xl font-bold text-[var(--text-primary)] tracking-tight leading-none">{formatStarCount(starCount)}</div>
              <div className="text-[10px] sm:text-[11px] font-medium text-[var(--text-label)] leading-snug break-words">{lang === 'zh' ? '开源社区支持' : 'Open Source Community'}</div>
            </div>
          </div>
        </section>

        {/* FEATURES STRIP */}
        <section data-tour="features" className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,180px),1fr))] bg-[var(--bg-sidebar)] rounded-2xl border border-black/10 shadow-sm p-3 sm:p-4 gap-3 sm:gap-4">
          {[
            { icon: <Monitor className="w-4 h-4 sm:w-4.5 sm:h-4.5 stroke-[2]" />, title: t.feature1, desc: t.feature1d },
            { icon: <ArrowRight className="w-4 h-4 sm:w-4.5 sm:h-4.5 stroke-[2.5]" />, title: t.feature2, desc: t.feature2d },
            { icon: <Layers className="w-4 h-4 sm:w-4.5 sm:h-4.5 stroke-[2]" />, title: t.feature3, desc: t.feature3d },
            { icon: <Languages className="w-4 h-4 sm:w-4.5 sm:h-4.5 stroke-[2]" />, title: t.feature4, desc: t.feature4d },
            { icon: <Rocket className="w-4 h-4 sm:w-4.5 sm:h-4.5 stroke-[2]" />, title: t.feature5, desc: t.feature5d },
          ].map((f, i) => (
            <div key={i} className="p-1.5 sm:p-2 flex items-start gap-2.5 sm:gap-3.5">
              <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center border shrink-0 bg-[var(--bg-input)] text-[var(--text-muted)] border-[var(--border-light)]/50">{f.icon}</div>
              <div className="space-y-0.5 min-w-0">
                <h3 className="text-[11px] sm:text-[13px] font-bold text-[var(--text-primary)] leading-snug break-words">{f.title}</h3>
                <p className="text-[10px] sm:text-[11px] text-[var(--text-label)] font-medium leading-normal break-words">{f.desc}</p>
              </div>
            </div>
          ))}
        </section>

        {/* LEARNING PATH */}
        <section id="learning-path-section" className="parts bg-[var(--bg-sidebar)] rounded-2xl border border-black/10 shadow-sm p-5 md:p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <h2 className="text-[18px] md:text-[20px] font-bold text-[var(--text-primary)]">{t.learningPathTitle}</h2>
              <p className="text-xs text-[var(--text-muted)] font-medium">{t.learningPathSub}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5 relative">
            {PATH_STEPS.map((step, idx) => {
              const ps = PATH_STEP_STYLES[idx]
              return (
              <div key={idx} className="relative flex items-center w-full">
                <div
                  onClick={() => {
                    window.dispatchEvent(new CustomEvent('sidebar-scroll-to', { detail: { section: step.section } }))
                  }}
                  className="w-full bg-white rounded-[10px] p-3 sm:p-4 border border-black/10 flex flex-col justify-between shadow-sm relative hover:bg-slate-50 transition-colors duration-200 cursor-pointer group overflow-hidden"
                >
                  <span className={`text-[11px] font-mono font-bold tracking-wider px-1.5 py-0.5 rounded ${ps.numBg} w-fit`}>{step.num}</span>
                  <div className="space-y-1 mt-2 sm:mt-3">
                    <h3 className="text-[12px] sm:text-[13px] font-semibold text-slate-700 group-hover:text-[#7c3aed] transition-colors">{lang === 'zh' ? step.title : step.titleEn}</h3>
                    <p className="text-[10px] sm:text-[11px] text-[var(--text-muted)] leading-normal line-clamp-2">{lang === 'zh' ? step.desc : step.descEn}</p>
                  </div>
                </div>

                {idx < PATH_STEPS.length - 1 && (
                  <div className="hidden lg:flex absolute right-[-14px] top-1/2 -translate-y-1/2 z-10 pointer-events-none items-center justify-center text-slate-200 w-6">
                    <div className="w-full border-t-2 border-dashed border-[var(--border-light)]/80"></div>
                    <div className="w-2.5 h-2.5 rounded-full border-2 border-[var(--border-light)] bg-white absolute"></div>
                  </div>
                )}
              </div>
              )
            })}
          </div>
        </section>

        {/* RUNNABLE NOTEBOOKS */}
        <section data-tour="notebooks" className="bg-[var(--bg-sidebar)] rounded-2xl border border-black/10 shadow-sm p-5 md:p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <h2 className="text-[18px] md:text-[20px] font-bold text-[var(--text-primary)]">{t.runnableNotebooksTitle}</h2>
              <p className="text-xs text-[var(--text-muted)] font-medium">{t.runnableNotebooksSub}</p>
            </div>
          </div>

          <div className="flex gap-3.5 overflow-x-auto pb-2 snap-x snap-mandatory scrollbar-thin">
            {RUNNABLE_NOTEBOOKS.map((nb) => {
              const style = SECTION_STYLES[nb.section] || SECTION_STYLES.foundation
              const notebookMeta = catalogById.get(nb.lessonId)
              const notebookTitle = notebookMeta?.title || (lang === 'zh' ? nb.title : nb.titleEn)
              return (
                <div
                  key={nb.id}
                  onClick={() => onSelect(nb.lessonId)}
                  className="shrink-0 w-[180px] sm:w-[200px] border border-black/10 rounded-[10px] overflow-hidden cursor-pointer bg-white relative hover:bg-slate-50 transition-colors duration-200 group flex flex-col snap-start"
                >
                  <div className={`h-[90px] flex items-center justify-center relative overflow-hidden bg-gradient-to-br ${NOTEBOOK_BG[nb.id] || style.bg}`}>
                    {NOTEBOOK_SVGS[nb.id] || NOTEBOOK_SVGS['nb-1']}
                  </div>
                  <div className="p-3 bg-white flex-1 flex flex-col justify-between">
                    <div>
                      <h4 className="text-[12px] font-semibold text-[var(--text-primary)] group-hover:text-[#7c3aed] transition-colors mb-1 line-clamp-1">{notebookTitle}</h4>
                      <p className="text-[11px] text-[var(--text-label)] line-clamp-1 leading-relaxed">{lang === 'zh' ? nb.desc : nb.descEn}</p>
                    </div>
                    <div className="flex items-center justify-between text-[10px] mt-2">
                      <span className={`inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-semibold border ${style.tag}`}>
                        {lang === 'zh' ? style.nameZh : style.nameEn}
                      </span>
                      <span className="text-[var(--text-label)] font-medium">{nb.duration}{lang === 'zh' ? '分钟' : 'm'}</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </section>

        {/* SPONSORS */}
        <section className="bg-[var(--bg-sidebar)] rounded-2xl border border-black/10 shadow-sm p-5 md:p-6 space-y-4">
          <div className="space-y-0.5">
            <h2 className="text-[18px] md:text-[20px] font-bold text-[var(--text-primary)]">{t.sponsorsTitle}</h2>
            <p className="text-xs text-[var(--text-muted)] font-medium">{t.sponsorsSub}</p>
          </div>

          <div className="flex flex-wrap items-stretch gap-3">
            <a
              href="https://www.amd.com/en/products/accelerators/instinct.html"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-4 px-4 sm:px-5 py-3 sm:py-3.5 bg-white rounded-xl border border-black/10 hover:border-black/20 hover:shadow-md transition-all"
            >
              <svg width="84" height="20" viewBox="0 0 512 123" className="shrink-0 text-[var(--text-primary)]" aria-hidden="true" fill="currentColor">
                <path d="M120.415 114.002H91.868L83.184 93.04H35.839L27.923 114H0L42.654 8.172h30.562zM58.522 33.383L42.838 74.61h32.577zM223.386 8.172h22.976v105.83h-26.384v-65.96l-28.546 33.2h-4.03l-28.547-33.347v65.96H132.47V8.172h22.976l33.97 39.356zm89.816 0c38.624 0 58.632 24.039 58.632 53.061c0 30.415-19.239 52.769-61.453 52.769h-43.9V8.172zm-20.337 86.445h17.223c26.53 0 34.446-17.993 34.446-33.53c0-18.323-9.785-33.53-34.74-33.53h-16.93zm131.261-54.674v47.931h47.931l-34.226 34.263H389.9V74.169zM512 0v121.11l-33.273-33.273V33.273h-54.564L390.926 0z" />
              </svg>
              <div className="border-l border-[var(--border-light)] pl-3 sm:pl-4 hidden sm:block">
                <div className="text-[11px] font-bold text-[var(--text-primary)] leading-tight">AMD Instinct™</div>
                <div className="text-[10px] text-[var(--text-muted)] mt-0.5 leading-tight">{t.amdDesc}</div>
              </div>
            </a>

            <a
              href="https://modelscope.cn"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 px-4 sm:px-5 py-3 sm:py-3.5 bg-white rounded-xl border border-black/10 hover:border-black/20 hover:shadow-md transition-all"
            >
              <div className="flex items-center gap-2 text-[var(--text-primary)]">
                <svg width="22" height="22" viewBox="0 0 24 24" className="shrink-0" aria-hidden="true" fill="currentColor">
                  <path d="M2.667 5.3H8v2.667H5.333v2.666H2.667V8.467H.5v2.166h2.167V13.3H0V7.967h2.667V5.3zM2.667 13.3h2.666v2.667H8v2.666H2.667V13.3zM8 10.633h2.667V13.3H8v-2.667zM13.333 13.3v2.667h-2.666V13.3h2.666zM13.333 13.3v-2.667H16V13.3h-2.667z" />
                  <path fillRule="evenodd" clipRule="evenodd" d="M21.333 13.3v-2.667h-2.666V7.967H16V5.3h5.333v2.667H24V13.3h-2.667zm0-2.667H23.5V8.467h-2.167v2.166z" />
                  <path d="M21.333 13.3v5.333H16v-2.666h2.667V13.3h2.666z" />
                </svg>
                <span className="text-base font-bold tracking-tight leading-none">ModelScope</span>
              </div>
              <div className="border-l border-[var(--border-light)] pl-3 sm:pl-4 hidden sm:block">
                <div className="text-[11px] font-bold text-[var(--text-primary)] leading-tight">{lang === 'zh' ? '魔搭社区' : 'Community'}</div>
                <div className="text-[10px] text-[var(--text-muted)] mt-0.5 leading-tight">{t.modelscopeDesc}</div>
              </div>
            </a>
          </div>
        </section>

        {/* FOOTER */}
        <div className="w-full flex items-center justify-center pt-2 pb-6 select-none">
          <span className="text-xs text-[var(--text-label)] tracking-wide">{t.footerQuote}</span>
        </div>
        <ReaderLetterModal
          isOpen={isLetterOpen}
          onClose={() => setIsLetterOpen(false)}
          lang={lang}
        />
    </div>
  )
}
