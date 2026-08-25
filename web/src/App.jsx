import { useState, useEffect, useCallback, useRef } from 'react'
import { flushSync } from 'react-dom'
import { PanelLeftOpen } from 'lucide-react'
import Sidebar from './components/Sidebar.jsx'
import NotebookViewer from './components/NotebookViewer.jsx'
import NotesPanel from './components/NotesPanel.jsx'
import Welcome from './components/Welcome.jsx'
import GuidedTour from './components/GuidedTour.jsx'
import SettingsPanel from './components/SettingsPanel.jsx'
import ChangelogModal from './components/ChangelogModal.jsx'
import WalkingLabsModal from './components/WalkingLabsModal.jsx'
import { SettingsProvider } from './context/SettingsContext.jsx'
import useSettings from './hooks/useSettings.js'
import useTheme from './hooks/useTheme.js'
import useNotesAndBookmarks from './hooks/useNotesAndBookmarks.js'
import { getCatalog, getNotebook, getCachedNotebook, prefetchNotebook } from './data/notebooks.js'

const NOTES_SENTINEL = '__notes__'

const DEFAULT_LANG = 'zh'

const LEGACY_NOTEBOOK_IDS = {}

// 从构建时注入的 git log 数据中读取
const CHANGELOG_COMMITS = typeof __CHANGELOG_COMMITS__ !== 'undefined' ? __CHANGELOG_COMMITS__ : []

function normalizeNotebookId(id) {
  return LEGACY_NOTEBOOK_IDS[id] || id
}

function normalizeLang(lang) {
  return lang === 'en' ? 'en' : DEFAULT_LANG
}

function getInitialLang() {
  const params = new URLSearchParams(window.location.search)
  return normalizeLang(params.get('lang') || window.localStorage.getItem('language'))
}

function writeLangToUrl(lang) {
  const params = new URLSearchParams(window.location.search)
  if (lang === DEFAULT_LANG) {
    params.delete('lang')
  } else {
    params.set('lang', lang)
  }
  const query = params.toString()
  const nextUrl = `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`
  window.history.replaceState(null, '', nextUrl)
}

function replaceUrlWithHash(id, lang = DEFAULT_LANG) {
  const params = new URLSearchParams(window.location.search)
  if (lang === DEFAULT_LANG) {
    params.delete('lang')
  } else {
    params.set('lang', lang)
  }
  const query = params.toString()
  const hash = id ? `#${id}` : ''
  window.history.replaceState(null, '', `${window.location.pathname}${query ? `?${query}` : ''}${hash}`)
}

function getInitialNotebookId() {
  const hash = window.location.hash.replace(/^#\/?/, '')
  return hash ? normalizeNotebookId(hash) : null
}

function getInitialSidebarOpen() {
  return window.innerWidth >= 768
}

function AppContent() {
  const [lang, setLang] = useState(() => getInitialLang())
  const [catalog, setCatalog] = useState(() => getCatalog(lang))
  const [currentId, setCurrentId] = useState(() => getInitialNotebookId())
  const [notebook, setNotebook] = useState(null)
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(() => getInitialSidebarOpen())
  const [tourActive, setTourActive] = useState(false)
  const [tourStepIndex, setTourStepIndex] = useState(0)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [changelogOpen, setChangelogOpen] = useState(false)
  const [walkingLabsOpen, setWalkingLabsOpen] = useState(false)

  const { settings, updateSettings } = useSettings()
  const { resolvedTheme, toggleTheme } = useTheme(settings.theme)
  const nbm = useNotesAndBookmarks()

  // catalog 走 ref,schedulePrefetch 的回调身份就能保持稳定,不会触发下游 effect 重跑
  const catalogRef = useRef(catalog)
  useEffect(() => {
    catalogRef.current = catalog
  }, [catalog])

  // 浏览器空闲时预取下一篇 notebook,点击下一篇时基本秒开
  const schedulePrefetch = useCallback((id, lng) => {
    const list = catalogRef.current
    const idx = list.findIndex(n => n.id === id)
    if (idx < 0) return
    const next = list[idx + 1]
    if (!next) return
    const run = () => prefetchNotebook(next.id, lng)
    if ('requestIdleCallback' in window) {
      window.requestIdleCallback(run, { timeout: 2000 })
    } else {
      setTimeout(run, 200)
    }
  }, [])

  // 同步字号到 CSS 变量
  useEffect(() => {
    const sizeMap = { small: '14.5px', default: '16.5px', large: '18.5px' }
    document.documentElement.style.setProperty('--font-size-notebook', sizeMap[settings.fontSize] || '16.5px')
  }, [settings.fontSize])

  useEffect(() => {
    const mobileQuery = window.matchMedia('(max-width: 767px)')
    const syncSidebarForMobile = () => {
      if (mobileQuery.matches) setSidebarOpen(false)
    }

    syncSidebarForMobile()
    mobileQuery.addEventListener?.('change', syncSidebarForMobile)
    return () => mobileQuery.removeEventListener?.('change', syncSidebarForMobile)
  }, [])

  useEffect(() => {
    setCatalog(getCatalog(lang))
    document.documentElement.lang = lang === 'en' ? 'en' : 'zh-CN'
    window.localStorage.setItem('language', lang)
    writeLangToUrl(lang)
  }, [lang])

  useEffect(() => {
    let cancelled = false

    if (!currentId || currentId === NOTES_SENTINEL) {
      setNotebook(null)
      setLoading(false)
      return () => {
        cancelled = true
      }
    }

    // 缓存命中:同步 set,跳过 spinner,整次切换无 loading 态
    const cached = getCachedNotebook(currentId, lang)
    if (cached) {
      setNotebook(cached)
      setLoading(false)
      schedulePrefetch(currentId, lang)
      return () => {
        cancelled = true
      }
    }

    // 缓存未命中:第一次访问,只能走异步
    // 不清空旧 notebook —— 保留它直到新 notebook 就绪,避免全屏 spinner 闪一下
    setLoading(true)

    getNotebook(currentId, lang)
      .then((nextNotebook) => {
        if (!cancelled) {
          setNotebook(nextNotebook)
          schedulePrefetch(currentId, lang)
        }
      })
      .catch((error) => {
        console.error(`Failed to load notebook ${currentId}`, error)
        if (!cancelled) {
          setNotebook(null)
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [currentId, lang])

  useEffect(() => {
    const syncFromHash = () => {
      const nextId = getInitialNotebookId()
      if (nextId === NOTES_SENTINEL) return
      setCurrentId((prev) => {
        if (prev === nextId) return prev
        return nextId
      })
      if (nextId && window.location.hash !== `#${nextId}`) {
        replaceUrlWithHash(nextId, lang)
      }
    }

    window.addEventListener('hashchange', syncFromHash)
    window.addEventListener('popstate', syncFromHash)
    syncFromHash()
    return () => {
      window.removeEventListener('hashchange', syncFromHash)
      window.removeEventListener('popstate', syncFromHash)
    }
  }, [lang])

  const handleSelect = useCallback((id) => {
    flushSync(() => setCurrentId(id))
    replaceUrlWithHash(id, lang)
    if (window.innerWidth < 768) {
      setSidebarOpen(false)
    }
  }, [lang])

  const handleLanguageChange = useCallback((nextLang) => {
    setLang(normalizeLang(nextLang))
  }, [])

  const handleHome = useCallback(() => {
    flushSync(() => setCurrentId(null))
    replaceUrlWithHash(null, lang)
    if (window.innerWidth < 768) {
      setSidebarOpen(false)
    }
  }, [lang])

  const handleOpenNotes = useCallback(() => {
    flushSync(() => setCurrentId(NOTES_SENTINEL))
    replaceUrlWithHash(null, lang)
    if (window.innerWidth < 768) {
      setSidebarOpen(false)
    }
  }, [lang])

  const currentMeta = catalog.find(n => n.id === currentId)
  const tourNotebookId = catalog.find(n => n.id === '01-course-overview')?.id || catalog[0]?.id
  const tourCopy = {
    zh: [
      {
        'target': '.hero',
        'title': '欢迎来到 Self-Improving Agent Notebook',
        'body': '这不是框架调用教程，而是从零重建 Agent 系统的交互式课程。你会沿着一个 Agent 真正运转的顺序，理解循环、验证、规划、强化学习与记忆。',
      },
      {
        'target': '.stats',
        'title': '课程规模',
        'body': '17 篇可运行 Notebook，覆盖 4 大学习路径、20+ 核心模块。每篇都是可执行代码，读完马上能改代码观察结果。',
      },
      {
        'target': '[data-tour="features"]',
        'title': '课程特色',
        'body': '浏览器内直接渲染 Notebook，无需配置环境。支持 脚本化 离线运行，配置 API key 后即切换真实模型行为。',
      },
      {
        'target': '.parts',
        'title': '4 大学习路径',
        'body': '这里是整套课程的地图：基础与方法、训练与进化、智能体工程、前沿。每张卡片概括一个阶段要解决的问题，点击后定位到对应学习路径。',
      },
      {
        'target': '[data-tour="notes-saved"]',
        'title': '笔记与收藏',
        'body': '点击这里打开笔记与收藏面板。阅读时可以收藏 Notebook、选中文字添加笔记或高亮标记，统一管理并支持导出备份。',
        'nextLabel': '打开看看',
        'action': 'open-notes',
      },
      {
        'target': '.notes-panel',
        'title': '笔记与收藏面板',
        'body': '这里展示了你所有的收藏和笔记。上方可以按收藏或笔记筛选，每条笔记都会标注出处章节。底部有导出和导入按钮。',
      },
      {
        'target': '[data-tour="settings"]',
        'title': '个性化设置',
        'body': '点击齿轮图标打开设置面板，可以切换浅色/深色/跟随系统主题，还能调整正文字号大小。设置会自动保存。',
        'nextLabel': '打开看看',
        'action': 'open-settings',
      },
      {
        'target': '.modal-card',
        'title': '设置面板',
        'body': '这里可以切换主题：浅色、深色或跟随系统自动切换。下方还能调整正文字号（小/中/大），选择后立即生效。',
      },
      {
        'target': '[data-tour="walkinglabs"]',
        'title': '扫码加入社群',
        'body': '点击左上角的「...」按钮，可以打开 WalkingLabs 页面，一个专注 Agent 技术的开源实验室。',
        'nextLabel': '打开看看',
        'action': 'open-walkinglabs',
      },
      {
        'target': '.modal-card',
        'title': '社群与交流',
        'body': 'WalkingLabs 专注于 Agent 相关基础建设与教程。扫码加入微信群，和其他开发者一起探讨 Agent 技术。',
      },
      {
        'target': '[data-tour="notebooks"]',
        'title': '精选可运行 Notebook',
        'body': '这里展示了精选 Notebook，每篇都有可视化封面。点击即可进入阅读，支持一键打开到 ModelScope 或 Colab 运行。',
        'nextLabel': '进入 01',
        'action': 'open-notebook',
      },
      {
        'target': '.viewer-launches',
        'title': '一键运行',
        'body': '顶部按钮可以把当前 Notebook 打开到 ModelScope 或 Colab，在线运行代码，无需本地配置。',
      },
      {
        'target': '.bookmark-star',
        'title': '收藏与笔记',
        'body': '点击运行按钮左侧的星标可收藏当前 Notebook。所有收藏和笔记在左侧边栏「笔记与收藏」中统一管理。',
      },
      {
        'target': '.viewer-body',
        'title': '选中文字即可操作',
        'body': '阅读时选中任意文字，会弹出工具栏：复制内容、添加笔记、黄色高亮标记重点。',
      },
      {
        'target': '.toc',
        'title': '右侧大纲导航',
        'body': '右侧大纲对应每个学习环节，点击可快速跳转。大纲上的圆点表示该章节有笔记。推荐顺序：先看直觉，再手算，然后运行代码。',
      },
      {
        'target': '.code_cell',
        'title': '代码和输出可展开',
        'body': '每个核心算法都会落到代码。长代码和长输出默认折叠，点击可展开。右上角按钮可以复制代码。',
      },
      {
        'target': '.viewer-header',
        'title': '开始你的学习旅程',
        'body': '从 01 什么是 AI Agent 开始，一步步搭建你的 Agent 知识体系。每篇 Notebook 都是自包含的，可以按任意顺序阅读。',
      },
    ],
    en: [
      {
        'target': '.hero',
        'title': 'Welcome to Self-Improving Agent Notebook',
        'body': 'Not an API tutorial. Rebuild agent systems from scratch: loops, verification, planning, RL and memory.',
      },
      {
        'target': '.stats',
        'title': 'Course Overview',
        'body': '17 runnable notebooks across 4 learning paths and 20+ core modules. Every notebook is executable.',
      },
      {
        'target': '[data-tour="features"]',
        'title': 'What Makes It Different',
        'body': 'Notebooks render in your browser with zero setup. Scripted mode runs offline; add an API key for real models.',
      },
      {
        'target': '.parts',
        'title': '4 Learning Paths',
        'body': 'The course map: Foundation, Training & Evolution, Engineering, Frontiers.',
      },
      {
        'target': '[data-tour="notes-saved"]',
        'title': 'Notes & Bookmarks',
        'body': 'Bookmark notebooks, highlight text, or add notes while reading.',
        'nextLabel': 'Take a look',
        'action': 'open-notes',
      },
      {
        'target': '.notes-panel',
        'title': 'Notes Panel',
        'body': 'All bookmarks and notes are listed here, with export and import support.',
      },
      {
        'target': '[data-tour="settings"]',
        'title': 'Personalization',
        'body': 'Switch light/dark/system themes or adjust font size. Saved automatically.',
        'nextLabel': 'Take a look',
        'action': 'open-settings',
      },
      {
        'target': '.modal-card',
        'title': 'Settings Panel',
        'body': 'Switch themes and adjust the reading font size here.',
      },
      {
        'target': '[data-tour="walkinglabs"]',
        'title': 'Join the Community',
        'body': 'Open the WalkingLabs page from the "..." button. An open-source lab for Agent technology.',
        'nextLabel': 'Take a look',
        'action': 'open-walkinglabs',
      },
      {
        'target': '.modal-card',
        'title': 'Community & Discussion',
        'body': 'Scan the QR code to join the community and discuss Agent tech.',
      },
      {
        'target': '[data-tour="notebooks"]',
        'title': 'Runnable Notebooks',
        'body': 'Selected notebooks with visual covers. Open in ModelScope or Colab to run online.',
        'nextLabel': 'Open 01',
        'action': 'open-notebook',
      },
      {
        'target': '.viewer-launches',
        'title': 'One-Click Run',
        'body': 'Open the current notebook in ModelScope or Colab to run code online.',
      },
      {
        'target': '.bookmark-star',
        'title': 'Bookmarks & Notes',
        'body': 'Click the star to bookmark. Manage everything under "Notes & Saved".',
      },
      {
        'target': '.viewer-body',
        'title': 'Select Text to Annotate',
        'body': 'Select any text for a toolbar: copy, add a note, or highlight key points.',
      },
      {
        'target': '.toc',
        'title': 'Table of Contents',
        'body': 'Outline each section. Dots mark sections with notes. Recommended order: intuition, then hand calculation, then code.',
      },
      {
        'target': '.code_cell',
        'title': 'Expandable Code & Output',
        'body': 'Long code and output are collapsed by default — click to expand. Copy button on the right.',
      },
      {
        'target': '.viewer-header',
        'title': 'Start Your Journey',
        'body': 'Begin with 01 Course Overview and build your agent knowledge step by step.',
      },
    ],
  }

  const tourSteps = tourCopy[lang]

  const startTour = useCallback(() => {
    flushSync(() => {
      setCurrentId(null)
      setSidebarOpen(true)
      setTourStepIndex(0)
      setTourActive(true)
    })
    replaceUrlWithHash(null, lang)
  }, [lang])

  const stopTour = useCallback(() => {
    setTourActive(false)
    setSettingsOpen(false)
    setWalkingLabsOpen(false)
  }, [])

  const handleTourNext = useCallback(() => {
    const step = tourSteps[tourStepIndex]
    if (step?.action === 'open-notebook' && tourNotebookId) {
      flushSync(() => {
        setCurrentId(tourNotebookId)
        setSidebarOpen(true)
      })
      replaceUrlWithHash(tourNotebookId, lang)
    }
    if (step?.action === 'open-notes') {
      flushSync(() => {
        setSettingsOpen(false)
        setWalkingLabsOpen(false)
        setCurrentId(NOTES_SENTINEL)
      })
    }
    if (step?.action === 'open-settings') {
      flushSync(() => {
        setCurrentId(null)
        setWalkingLabsOpen(false)
        setSettingsOpen(true)
      })
    }
    if (step?.action === 'open-walkinglabs') {
      flushSync(() => {
        setCurrentId(null)
        setSettingsOpen(false)
        setWalkingLabsOpen(true)
      })
    }

    if (tourStepIndex >= tourSteps.length - 1) {
      setSettingsOpen(false)
      setWalkingLabsOpen(false)
      setTourActive(false)
      return
    }
    const nextStep = tourSteps[tourStepIndex + 1]
    if (!nextStep?.target?.startsWith('.notes-panel') && !nextStep?.target?.startsWith('.modal-card')) {
      setSettingsOpen(false)
      setWalkingLabsOpen(false)
    }
    setTourStepIndex(i => i + 1)
  }, [lang, tourNotebookId, tourStepIndex, tourSteps])

  const handleTourPrev = useCallback(() => {
    setSettingsOpen(false)
    setWalkingLabsOpen(false)
    setTourStepIndex(i => Math.max(0, i - 1))
  }, [])

  return (
    <div className="h-screen flex overflow-hidden bg-[var(--bg-app)] text-[var(--text-body)] font-sans antialiased">
      {!sidebarOpen && (
        <button
          onClick={() => setSidebarOpen(true)}
          className="sidebar-toggle-btn"
          aria-label={lang === 'zh' ? '展开左侧栏' : 'Expand sidebar'}
          title={lang === 'zh' ? '展开左侧栏' : 'Expand sidebar'}
        >
          <PanelLeftOpen className="w-5 h-5" />
        </button>
      )}

      {/* Sidebar overlay for mobile */}
      {sidebarOpen && (
        <div
          className="md:hidden fixed inset-0 z-20 bg-black/20 backdrop-blur-sm"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <Sidebar
        catalog={catalog}
        currentId={currentId}
        lang={lang}
        onLanguageChange={handleLanguageChange}
        onSelect={handleSelect}
        onHome={handleHome}
        onStartTour={startTour}
        onOpenNotes={handleOpenNotes}
        onOpenSettings={() => setSettingsOpen(true)}
        onOpenChangelog={() => setChangelogOpen(true)}
        onOpenWalkingLabs={() => setWalkingLabsOpen(true)}
        bookmarks={nbm.bookmarks}
        notes={nbm.notes}
        notebooksWithNotes={nbm.notebooksWithNotes}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="flex-1 flex flex-col h-screen min-w-0 overflow-y-auto">
        {currentId === NOTES_SENTINEL ? (
          <NotesPanel
            catalog={catalog}
            bookmarks={nbm.bookmarks}
            notes={nbm.notes}
            notebooksWithNotes={nbm.notebooksWithNotes}
            getSectionNotes={nbm.getSectionNotes}
            exportData={nbm.exportData}
            importFile={nbm.importFile}
            onClearAll={nbm.clearAll}
            onSelect={handleSelect}
            lang={lang}
          />
        ) : currentId ? (
          <NotebookViewer
            notebook={notebook}
            meta={currentMeta}
            loading={loading}
            isBookmarked={nbm.isBookmarked}
            toggleBookmark={nbm.toggleBookmark}
            notes={nbm.notes}
            saveNote={nbm.saveNote}
            deleteNote={nbm.deleteNote}
            updateNoteSection={nbm.updateNoteSection}
          />
        ) : (
          <Welcome
            catalog={catalog}
            lang={lang}
            onLanguageChange={handleLanguageChange}
            onSelect={handleSelect}
            onStartTour={startTour}
          />
        )}
      </main>

      <GuidedTour
        active={tourActive}
        step={tourSteps[tourStepIndex]}
        stepIndex={tourStepIndex}
        totalSteps={tourSteps.length}
        lang={lang}
        onNext={handleTourNext}
        onPrev={handleTourPrev}
        onClose={stopTour}
      />

      <SettingsPanel
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        lang={lang}
      />

      <ChangelogModal
        isOpen={changelogOpen}
        onClose={() => setChangelogOpen(false)}
        lang={lang}
        commits={CHANGELOG_COMMITS}
      />

      <WalkingLabsModal
        isOpen={walkingLabsOpen}
        onClose={() => setWalkingLabsOpen(false)}
        lang={lang}
      />
    </div>
  )
}

export default function App() {
  const { settings, updateSettings } = useSettings()
  const { resolvedTheme, toggleTheme } = useTheme(settings.theme)

  return (
    <SettingsProvider settings={settings} updateSettings={updateSettings} resolvedTheme={resolvedTheme} toggleTheme={toggleTheme}>
      <AppContent />
    </SettingsProvider>
  )
}
