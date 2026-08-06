import { X, Sun, Moon, Monitor } from 'lucide-react'
import { useSettingsContext } from '../context/SettingsContext.jsx'

const THEME_OPTIONS = [
  { value: 'light', icon: Sun, labelZh: '浅色', labelEn: 'Light' },
  { value: 'system', icon: Monitor, labelZh: '跟随系统', labelEn: 'System' },
  { value: 'dark', icon: Moon, labelZh: '深色', labelEn: 'Dark' },
]

const FONT_SIZE_OPTIONS = [
  { value: 'small', label: 'S', sizeZh: '小', sizeEn: 'Small' },
  { value: 'default', label: 'M', sizeZh: '中', sizeEn: 'Default' },
  { value: 'large', label: 'L', sizeZh: '大', sizeEn: 'Large' },
]

export default function SettingsPanel({ isOpen, onClose, lang }) {
  const { settings, updateSettings, toggleTheme, resolvedTheme } = useSettingsContext()

  if (!isOpen) return null

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className="modal-backdrop" onClick={onClose} />
      <div className="modal-card" style={{ maxWidth: 420 }}>
        <div className="modal-header">
          <h2>{lang === 'zh' ? '设置' : 'Settings'}</h2>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="modal-body space-y-6">
          {/* Theme */}
          <div>
            <div className="text-xs font-bold text-[var(--text-secondary)] mb-3 tracking-wide uppercase">
              {lang === 'zh' ? '主题' : 'Theme'}
            </div>
            <div className="flex gap-2">
              {THEME_OPTIONS.map(opt => {
                const Icon = opt.icon
                const active = settings.theme === opt.value
                return (
                  <button
                    key={opt.value}
                    onClick={() => updateSettings({ theme: opt.value })}
                    className={`flex-1 flex flex-col items-center gap-1.5 py-3 rounded-xl border transition-all text-xs font-semibold ${
                      active
                        ? 'border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]'
                        : 'border-[var(--border-light)] text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-secondary)]'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{lang === 'zh' ? opt.labelZh : opt.labelEn}</span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Font Size */}
          <div>
            <div className="text-xs font-bold text-[var(--text-secondary)] mb-3 tracking-wide uppercase">
              {lang === 'zh' ? '正文字号' : 'Font Size'}
            </div>
            <div className="flex gap-2">
              {FONT_SIZE_OPTIONS.map(opt => {
                const active = settings.fontSize === opt.value
                return (
                  <button
                    key={opt.value}
                    onClick={() => updateSettings({ fontSize: opt.value })}
                    className={`flex-1 py-2.5 rounded-xl border transition-all font-bold ${
                      active
                        ? 'border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]'
                        : 'border-[var(--border-light)] text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-secondary)]'
                    }`}
                    title={lang === 'zh' ? opt.sizeZh : opt.sizeEn}
                  >
                    {opt.label}
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
