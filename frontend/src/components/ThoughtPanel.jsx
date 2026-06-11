import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { FileCode2, Search, FilePlus, Pencil, CheckCircle2, XCircle, RefreshCw } from 'lucide-react'
import s from './ThoughtPanel.module.css'

const TOOL_ICONS = {
  read_file:   <FileCode2 size={11} strokeWidth={1.8} />,
  write_file:  <FilePlus  size={11} strokeWidth={1.8} />,
  patch_file:  <Pencil    size={11} strokeWidth={1.8} />,
  search_grep: <Search    size={11} strokeWidth={1.8} />,
}

/* ── Merge plan_chunk events into a single accumulated plan text ─────── */
function processEvents(events) {
  const items = []
  let planAcc = null

  for (const ev of events) {
    if (ev.type === 'plan_chunk') {
      planAcc = (planAcc ?? '') + ev.chunk
    } else if (ev.type === 'plan') {
      // Final confirmed plan — prefer its content over accumulated chunks
      items.push({ type: 'plan', content: ev.content || planAcc || '' })
      planAcc = null
    } else {
      if (planAcc !== null) {
        items.push({ type: 'plan', content: planAcc })
        planAcc = null
      }
      items.push(ev)
    }
  }
  if (planAcc !== null) {
    items.push({ type: 'plan', content: planAcc })
  }
  return items
}

/* ── Entry renderers ─────────────────────────────────────────────────── */

function PlanEntry({ content }) {
  return (
    <div className={s.section}>
      <div className={s.sectionLabel}>Plan of attack</div>
      <div className={s.planMd}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    </div>
  )
}

function ToolEntry({ event }) {
  const { tool, args } = event
  const file = args?.path || args?.pattern || ''
  return (
    <div className={s.toolRow}>
      <span className={s.toolIcon}>{TOOL_ICONS[tool] ?? <FileCode2 size={11} strokeWidth={1.8} />}</span>
      <span className={s.toolName}>{tool}</span>
      {file && <span className={s.toolArg}>{file}</span>}
    </div>
  )
}

function ValidationEntry({ event }) {
  const { phase, passed, output } = event
  const label = phase === 'lint' ? 'Linter' : 'Tests'
  return (
    <div className={`${s.validRow} ${passed ? s.validPass : s.validFail}`}>
      <span className={s.validIcon}>
        {passed ? <CheckCircle2 size={12} strokeWidth={2} /> : <XCircle size={12} strokeWidth={2} />}
      </span>
      <span className={s.validLabel}>{label} — {passed ? 'passed' : 'failed'}</span>
      {!passed && output && (
        <pre className={s.validOutput}>{output.length > 600 ? output.slice(0, 600) + '\n…' : output}</pre>
      )}
    </div>
  )
}

function CorrectionEntry({ event }) {
  return (
    <div className={s.correctionRow}>
      <RefreshCw size={11} strokeWidth={2} />
      Self-correcting — attempt {event.attempt} of 3
    </div>
  )
}

function DoneEntry() {
  return (
    <div className={s.doneRow}>
      <CheckCircle2 size={12} strokeWidth={2} />
      Complete
    </div>
  )
}

/* ── ThoughtPanel ────────────────────────────────────────────────────── */

export default function ThoughtPanel({ isOpen, events = [] }) {
  const bodyRef = useRef(null)

  useEffect(() => {
    const el = bodyRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [events])

  const items = processEvents(events)

  return (
    <div className={`${s.panel} ${isOpen ? '' : s.collapsed}`}>
      <div className={s.drag} />

      <div className={s.header}>
        <span className={s.headerTitle}>Agent Reasoning</span>
      </div>

      <div className={s.body} ref={bodyRef}>
        {items.length === 0 ? (
          <div className={s.empty}>
            <p>The agent's plan, tool calls, and validation results will appear here.</p>
          </div>
        ) : (
          <div className={s.feed}>
            {items.map((ev, i) => {
              if (ev.type === 'plan')              return <PlanEntry       key={i} content={ev.content} />
              if (ev.type === 'tool_call')         return <ToolEntry       key={i} event={ev} />
              if (ev.type === 'validation_result') return <ValidationEntry key={i} event={ev} />
              if (ev.type === 'correction')        return <CorrectionEntry key={i} event={ev} />
              if (ev.type === 'done')              return <DoneEntry       key={i} />
              return null
            })}
          </div>
        )}
      </div>
    </div>
  )
}
