import { useEffect, useLayoutEffect, useRef, useState } from 'react'

function getPlacement(rect) {
  const roomRight = window.innerWidth - rect.right
  const roomLeft = rect.left
  const roomBelow = window.innerHeight - rect.bottom

  if (roomRight > 360) return 'right'
  if (roomLeft > 360) return 'left'
  if (roomBelow > 220) return 'bottom'
  return 'top'
}

function getCardStyle(rect, placement) {
  const gap = 18
  const width = Math.min(400, window.innerWidth - 32)

  if (!rect) {
    return {
      left: 16,
      bottom: 82,
      width,
    }
  }

  if (placement === 'right') {
    return {
      left: Math.min(rect.right + gap, window.innerWidth - width - 16),
      top: Math.max(16, Math.min(rect.top, window.innerHeight - 260)),
      width,
    }
  }

  if (placement === 'left') {
    return {
      left: Math.max(16, rect.left - width - gap),
      top: Math.max(16, Math.min(rect.top, window.innerHeight - 260)),
      width,
    }
  }

  if (placement === 'bottom') {
    return {
      left: Math.max(16, Math.min(rect.left, window.innerWidth - width - 16)),
      top: rect.bottom + gap,
      width,
    }
  }

  return {
    left: Math.max(16, Math.min(rect.left, window.innerWidth - width - 16)),
    top: Math.max(16, rect.top - 236),
    width,
  }
}

function GuidedTour({ active, step, stepIndex, totalSteps, lang, onNext, onPrev, onClose }) {
  const [targetRect, setTargetRect] = useState(null)
  const rafRef = useRef(null)
  const mascotExpressions = [
    'normal',
    'smile',
    'normal',
    'cry',
    'normal',
    'think',
    'normal',
    'smile',
    'normal',
    'normal',
    'normal',
    'normal',
  ]

  useLayoutEffect(() => {
    if (!active || !step) return

    const update = () => {
      const target = step.target ? document.querySelector(step.target) : null
      if (!target) {
        setTargetRect(null)
        return
      }

      target.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' })
      rafRef.current = requestAnimationFrame(() => {
        const rect = target.getBoundingClientRect()
        setTargetRect({
          top: rect.top,
          right: rect.right,
          bottom: rect.bottom,
          left: rect.left,
          width: rect.width,
          height: rect.height,
        })
      })
    }

    update()
    const timer = window.setTimeout(update, 460)

    return () => {
      window.clearTimeout(timer)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [active, step])

  useEffect(() => {
    if (!active) return

    const update = () => {
      const target = step?.target ? document.querySelector(step.target) : null
      if (!target) return
      const rect = target.getBoundingClientRect()
      setTargetRect({
        top: rect.top,
        right: rect.right,
        bottom: rect.bottom,
        left: rect.left,
        width: rect.width,
        height: rect.height,
      })
    }

    window.addEventListener('resize', update)
    window.addEventListener('scroll', update, true)
    return () => {
      window.removeEventListener('resize', update)
      window.removeEventListener('scroll', update, true)
    }
  }, [active, step])

  useEffect(() => {
    if (!active) return

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose()
        return
      }

      if (event.key === 'ArrowLeft') {
        event.preventDefault()
        onPrev()
        return
      }

      if (
        event.key === 'Enter'
        || event.key === ' '
        || event.key === 'Spacebar'
        || event.key === 'ArrowRight'
      ) {
        event.preventDefault()
        onNext()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [active, onClose, onNext, onPrev])

  const handleLayerClick = (event) => {
    if (event.target.closest('button')) return
    onNext()
  }

  if (!active || !step) return null

  const placement = targetRect ? getPlacement(targetRect) : 'bottom'
  const cardStyle = getCardStyle(targetRect, placement)
  const isLast = stepIndex === totalSteps - 1
  const lookClass = {
    right: 'look-left',
    left: 'look-right',
    bottom: 'look-up',
    top: 'look-down',
  }[placement]

  return (
    <div
      className="tour-layer"
      role="dialog"
      aria-modal="true"
      aria-label={lang === 'en' ? 'Guided tour' : '新手引导'}
      onClick={handleLayerClick}
    >
      <div className="tour-backdrop" />
      {targetRect && (
        <div
          className="tour-spotlight"
          style={{
            top: targetRect.top - 8,
            left: targetRect.left - 8,
            width: targetRect.width + 16,
            height: targetRect.height + 16,
          }}
        />
      )}
      <div className={`tour-card tour-card-${placement}`} style={cardStyle}>
        <div
          className={`tour-mascot ${lookClass} tour-mascot-${mascotExpressions[stepIndex % mascotExpressions.length]}`}
          aria-hidden="true"
        >
          <span className="tour-mascot-eye tour-mascot-eye-left" />
          <span className="tour-mascot-eye tour-mascot-eye-right" />
          <span className="tour-mascot-mouth" />
          <span className="tour-mascot-spark tour-mascot-spark-one" />
          <span className="tour-mascot-spark tour-mascot-spark-two" />
        </div>
        <div className="tour-step-count">{stepIndex + 1} / {totalSteps}</div>
        <h2>{step.title}</h2>
        <p>{step.body}</p>
        <div className="tour-actions">
          <button className="tour-ghost" onClick={onClose}>
            {lang === 'en' ? 'Skip' : '跳过'}
          </button>
          <div className="tour-action-group">
            {stepIndex > 0 && (
              <button className="tour-secondary" onClick={onPrev}>
                {lang === 'en' ? 'Back' : '上一步'}
              </button>
            )}
            <button className="tour-primary" onClick={onNext}>
              {isLast
                ? (lang === 'en' ? 'Start learning' : '开始学习')
                : step.nextLabel || (lang === 'en' ? 'Next' : '下一步')}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default GuidedTour
