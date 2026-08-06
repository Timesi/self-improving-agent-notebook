import { X, GitCommitHorizontal } from 'lucide-react'

export default function ChangelogModal({ isOpen, onClose, lang, commits }) {
  if (!isOpen) return null

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className="modal-backdrop" onClick={onClose} />
      <div className="modal-card" style={{ maxWidth: 540 }}>
        <div className="modal-header">
          <div className="flex items-center gap-3">
            <h2>{lang === 'zh' ? '更新日志' : 'Changelog'}</h2>
            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[var(--accent-soft)] text-[var(--accent)] tracking-wide">
              v1.0.5
            </span>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="modal-body">
          {commits.length === 0 ? (
            <div className="text-center py-8 text-[var(--text-muted)] text-sm">
              {lang === 'zh' ? '暂无更新记录' : 'No commits found'}
            </div>
          ) : (
            <div className="space-y-1">
              {commits.map((commit, i) => (
                <div
                  key={commit.hash}
                  className="flex items-start gap-3 py-2.5 px-3 rounded-lg hover:bg-[var(--bg-hover)] transition-colors"
                >
                  <GitCommitHorizontal className="w-4 h-4 mt-0.5 text-[var(--text-muted)] shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="text-sm text-[var(--text-primary)] truncate">
                      {commit.message}
                    </div>
                    <div className="text-xs text-[var(--text-muted)] mt-0.5 flex items-center gap-2">
                      <span className="font-mono">{commit.hash}</span>
                      <span>{commit.date}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
