import { useEffect, useState } from 'react'
import s from './RunningWave.module.css'

/**
 * Bottom-of-window wave that fades in while a model is processing.
 * Three overlapping SVG sine layers, each periodic in the viewBox so
 * translateX(-50%) loops seamlessly.
 */
export default function RunningWave({ active }) {
  const [mounted, setMounted] = useState(active)

  useEffect(() => {
    if (active) { setMounted(true); return }
    const t = setTimeout(() => setMounted(false), 500) // outlast the fade-out
    return () => clearTimeout(t)
  }, [active])

  if (!mounted) return null

  return (
    <div className={`${s.wrap} ${active ? s.wrapActive : ''}`} aria-hidden="true">
      <div className={`${s.layer} ${s.layerA}`}>
        <svg viewBox="0 0 1200 60" preserveAspectRatio="none">
          <path
            d="M0,30 Q150,5 300,30 T600,30 T900,30 T1200,30 L1200,60 L0,60 Z"
            fill="rgba(99,102,241,0.34)"
          />
        </svg>
      </div>
      <div className={`${s.layer} ${s.layerB}`}>
        <svg viewBox="0 0 1200 60" preserveAspectRatio="none">
          <path
            d="M0,38 Q150,56 300,38 T600,38 T900,38 T1200,38 L1200,60 L0,60 Z"
            fill="rgba(139,141,255,0.22)"
          />
        </svg>
      </div>
      <div className={`${s.layer} ${s.layerC}`}>
        <svg viewBox="0 0 1200 60" preserveAspectRatio="none">
          <path
            d="M0,34 Q150,18 300,34 T600,34 T900,34 T1200,34 L1200,60 L0,60 Z"
            fill="rgba(167,139,250,0.16)"
          />
        </svg>
      </div>
    </div>
  )
}
