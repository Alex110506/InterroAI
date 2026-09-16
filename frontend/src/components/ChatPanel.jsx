import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Plus, ChevronDown, ArrowUp, Brain, FolderOpen,
  PanelRight, PanelRightClose,
  GitBranch, FileCode2, Layers, AlertTriangle, CheckCircle2,
} from 'lucide-react'
import { api } from '../lib/api'
import s from './ChatPanel.module.css'

/* ─── Per-project message history ──────────────────────────────────────
 * In memory only. The backend stores no transcript: every request re-sends
 * the conversation as `history`, and closing the app ends it. */
const projectMessages = {}

/* Turn the panel's own message list into the OpenAI message shape the
 * backend expects for `history` — user turns plus the agent's final replies.
 * Working notes (errors) are not part of the conversation. */
function toHistory(messages) {
  return messages
    .filter((m) => m.role === 'user' || m.subtype === 'message')
    .map((m) => ({ role: m.role === 'user' ? 'user' : 'assistant', content: m.content }))
}

/* ─── Embed step indicators ─────────────────────────────────────────── */
const STEP_LABELS = { A: 'Scanning files', B: 'Chunking code', C: 'Generating embeddings', D: 'Storing vectors' }

function EmbedProgress({ embedStatus, embedSteps, embedProgress, embedError }) {
  if (!embedStatus || embedStatus === 'idle') return null
  const steps = embedSteps ?? {}
  const prog = embedProgress ?? {}

  if (embedStatus === 'done') {
    return (
      <div className={s.embedDoneChip}>
        <CheckCircle2 size={12} strokeWidth={2} />
        Codebase indexed — ready to search
      </div>
    )
  }
  if (embedStatus === 'error') {
    return (
      <div className={s.embedErrorChip}>
        <AlertTriangle size={12} strokeWidth={2} />
        {embedError || 'Embedding failed — check your API key in Settings'}
      </div>
    )
  }

  return (
    <div className={s.embedSteps}>
      {['A', 'B', 'C', 'D'].map((key) => {
        const state = steps[key] ?? 'idle'
        let detail = ''
        if (key === 'A' && state === 'done' && prog.files != null) detail = `${prog.files} files`
        if (key === 'B' && state === 'done' && prog.chunks != null) detail = `${prog.chunks} chunks`
        if (key === 'C' && state === 'running' && prog.embedded != null) detail = `${prog.embedded}/${prog.total}`
        if (key === 'D' && state === 'done' && prog.stored != null) detail = `${prog.stored} stored`
        const dotCls = state === 'done' ? s.dotDone : state === 'running' ? s.dotRunning : s.dotIdle
        return (
          <div key={key} className={`${s.embedStep} ${state !== 'idle' ? s.embedStepActive : ''}`}>
            <span className={`${s.dot} ${dotCls}`} />
            <span className={s.embedStepLabel}>{STEP_LABELS[key]}</span>
            {detail && <span className={s.embedStepDetail}>{detail}</span>}
          </div>
        )
      })}
    </div>
  )
}

/* ─── Project empty / indexing state ────────────────────────────────── */
function ProjectEmptyState({ project }) {
  const { indexStatus, index, folderName, embedStatus, embedSteps, embedProgress, embedError } = project

  if (indexStatus === 'indexing') {
    return (
      <div className={s.empty}>
        <div className={`${s.emptyIcon} ${s.emptyIconPulse}`}>
          <Layers size={24} strokeWidth={1.4} />
        </div>
        <p className={s.emptyTitle}>Indexing project…</p>
        <p className={s.emptyDesc}>
          Scanning file tree and reading git context for <strong>{folderName}</strong>.
        </p>
      </div>
    )
  }

  if (indexStatus === 'error') {
    return (
      <div className={s.empty}>
        <div className={`${s.emptyIcon} ${s.emptyIconWarn}`}>
          <AlertTriangle size={24} strokeWidth={1.4} />
        </div>
        <p className={s.emptyTitle}>Indexing failed</p>
        <p className={s.emptyDesc}>
          Could not scan <strong>{folderName}</strong>. Make sure the backend is running.
        </p>
      </div>
    )
  }

  const git = index?.git_context
  const langs = index?.languages?.slice(0, 4) ?? []

  return (
    <div className={s.empty}>
      <div className={s.emptyIcon}>
        <Brain size={26} strokeWidth={1.4} />
      </div>
      <p className={s.emptyTitle}>Ask the agent anything</p>
      <p className={s.emptyDesc}>
        Ask a question, or describe what to build, fix, or change in <strong>{folderName}</strong>.
      </p>

      {index && (
        <div className={s.indexMeta}>
          <span className={s.metaChip}>
            <FileCode2 size={12} strokeWidth={1.8} />
            {index.total_files} files
          </span>
          {git?.is_git_repo && git.branch && (
            <span className={s.metaChip}>
              <GitBranch size={12} strokeWidth={1.8} />
              {git.branch}
            </span>
          )}
          {git?.modified_files?.length > 0 && (
            <span className={`${s.metaChip} ${s.metaChipModified}`}>
              {git.modified_files.length} modified
            </span>
          )}
          {langs.map((lang) => (
            <span key={lang} className={`${s.metaChip} ${s.metaChipLang}`}>{lang}</span>
          ))}
        </div>
      )}

      <EmbedProgress embedStatus={embedStatus} embedSteps={embedSteps} embedProgress={embedProgress} embedError={embedError} />
    </div>
  )
}

/* ─── Thinking / loading indicator ──────────────────────────────────── */
function ThinkingBubble() {
  return (
    <div className={s.message}>
      <div className={s.agentWrap}>
        <div className={s.thinking}>
          <span className={s.thinkDot} />
          <span className={s.thinkDot} />
          <span className={s.thinkDot} />
        </div>
      </div>
    </div>
  )
}

const MODELS = [
  { id: 'gpt-5.6-sol',   label: 'GPT-5.6 Sol' },
  { id: 'gpt-5.6-terra', label: 'GPT-5.6 Terra' },
  { id: 'gpt-5.6-luna',  label: 'GPT-5.6 Luna' },
]

// The model the picker starts on. Must be one of MODELS above — the backend
// rejects any ID it cannot resolve rather than quietly substituting.
//
// There is no effort picker: chat completions refuses `reasoning_effort`
// alongside function tools for this family, and the agent's middle phase is
// function tools throughout. Offering the choice would mean moving the gateway
// to the Responses API.
const DEFAULT_MODEL = 'gpt-5.6-sol'

/* ─── Main ChatPanel ─────────────────────────────────────────────────── */
export default function ChatPanel({ activeId, projects, thoughtOpen, onToggleThought, onThoughtEvent, clearThought, onAuthLost }) {
  const [input, setInput] = useState('')
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL)
  const [showModelDropdown, setShowModelDropdown] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [, forceUpdate] = useState(0)
  const textareaRef = useRef(null)
  const bottomRef = useRef(null)
  const wsRef = useRef(null)

  const project = projects.find((p) => p.id === activeId) ?? null



  /* ── Auto-resize textarea ── */
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [input])

  /* ── Reset everything when switching projects ── */
  useEffect(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsLoading(false)
    setInput('')
    bottomRef.current?.scrollIntoView({ behavior: 'instant' })
  }, [activeId])

  const messages = activeId ? (projectMessages[activeId] ?? []) : []

  const addMessage = useCallback((msg) => {
    if (!activeId) return
    if (!projectMessages[activeId]) projectMessages[activeId] = []
    projectMessages[activeId].push({ id: Date.now() + Math.random().toString(), ...msg })
    forceUpdate((n) => n + 1)
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)
  }, [activeId])

  /* ── WebSocket message handler ── */
  const handleWsMessage = useCallback((event) => {
    if (event.type === 'message') {
      setIsLoading(false)
      addMessage({ role: 'agent', subtype: 'message', content: event.content })
      wsRef.current?.close()
      wsRef.current = null

    // 'ready' just signals that routing finished and the agent is starting —
    // nothing new to show, the loading indicator already covers it.

    // ── Coder events → right panel ──────────────────────────────────────
    } else if (event.type === 'plan_chunk' || event.type === 'plan') {
      setIsLoading(false)
      onThoughtEvent?.(event)

    } else if (event.type === 'tool_call' || event.type === 'tool_result') {
      onThoughtEvent?.(event)

    } else if (event.type === 'validation_result') {
      onThoughtEvent?.(event)

    } else if (event.type === 'correction') {
      onThoughtEvent?.(event)

    } else if (event.type === 'impl_done') {
      onThoughtEvent?.(event)

    } else if (event.type === 'done') {
      setIsLoading(false)
      onThoughtEvent?.(event)
      if (event.summary) {
        addMessage({ role: 'agent', subtype: 'message', content: event.summary })
      }
      wsRef.current?.close()
      wsRef.current = null

    } else if (event.type === 'error') {
      setIsLoading(false)
      addMessage({ role: 'agent', subtype: 'error', content: event.message })
      wsRef.current?.close()
      wsRef.current = null
      // The cloud session ended: the app goes back to its sign-in screen.
      if (event.code === 'not_signed_in') onAuthLost?.()
    }
  }, [addMessage, onThoughtEvent, onAuthLost])

  /* ── Send handler ── */
  const send = async () => {
    const text = input.trim()
    if (!text || !activeId || isLoading || project?.indexStatus !== 'done') return

    // Snapshot the conversation so far — in OpenAI message shape — before this
    // turn's own user message is appended below. The backend stores no
    // transcript itself, so a follow-up like "what was my last question"
    // only resolves if every request re-sends the history the server needs.
    const history = toHistory(projectMessages[activeId] ?? [])

    addMessage({ role: 'user', content: text })
    setInput('')
    setIsLoading(true)
    clearThought?.()

    let ws
    try {
      ws = await api.openChatSocket()
    } catch (err) {
      setIsLoading(false)
      addMessage({ role: 'agent', subtype: 'error', content: err.message })
      return
    }
    wsRef.current = ws

    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: 'start',
        project_path: project.folderPath,
        project_index: project.index ?? {},
        message: text,
        model: selectedModel,
        history,
      }))
    }

    ws.onmessage = (e) => handleWsMessage(JSON.parse(e.data))

    ws.onerror = () => {
      setIsLoading(false)
      addMessage({ role: 'agent', subtype: 'error', content: 'Connection to backend failed. Is the server running?' })
    }

    ws.onclose = () => {
      setIsLoading((prev) => (prev ? false : prev))
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const PanelIcon = thoughtOpen ? PanelRightClose : PanelRight
  const canSend = !!input.trim() && !isLoading && project?.indexStatus === 'done'

  const placeholder =
    !project ? 'Select a project to begin…' :
    project.indexStatus !== 'done' ? 'Waiting for indexing to finish…' :
    `Ask about ${project.folderName}…`

  const currentModelLabel = MODELS.find(m => m.id === selectedModel)?.label ?? selectedModel
  // Pulled out of the JSX so the picker's look lives in one place.
  const dropdownStyle = {
    position: 'absolute', bottom: '100%', left: 0,
    backgroundColor: 'var(--bg)', border: '1px solid var(--border)',
    borderRadius: '6px', padding: '4px', zIndex: 10, marginBottom: '4px',
    width: 'max-content', display: 'flex', flexDirection: 'column', gap: '2px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
  }
  const optionStyle = (selected) => ({
    background: selected ? 'var(--bg-2)' : 'transparent',
    border: 'none', color: 'var(--text-1)', padding: '6px 12px',
    textAlign: 'left', borderRadius: '4px', cursor: 'pointer', fontSize: '13px',
  })

  /* ── No project selected ── */
  if (!project) {
    return (
      <div className={s.panel}>
        <header className={s.header}>
          <button
            className={`${s.panelToggle} ${thoughtOpen ? s.panelToggleActive : ''}`}
            onClick={onToggleThought}
            title={thoughtOpen ? 'Collapse thought panel' : 'Expand thought panel'}
          >
            <PanelIcon size={15} strokeWidth={1.8} />
          </button>
        </header>
        <div className={s.messages}>
          <div className={s.landing}>
            <div className={s.landingIcon}><Brain size={28} strokeWidth={1.4} /></div>
            <p className={s.landingTitle}>InterroAI</p>
            <p className={s.landingDesc}>Open a project folder to start a conversation with the agent.</p>
          </div>
        </div>
        <div className={s.inputArea}>
          <div className={s.inputBox}>
            <textarea className={s.textarea} placeholder="Select a project to begin…" disabled rows={1} />
            <div className={s.inputFooter}>
              <div className={s.inputLeft}>
                <button className={s.iconBtn} disabled><Plus size={16} strokeWidth={2} /></button>
                <div style={{position: 'relative'}}>
                  <button className={s.modelBtn} disabled>
                    <span>{currentModelLabel}</span><ChevronDown size={12} strokeWidth={2} />
                  </button>
                </div>
              </div>
              <button className={s.sendBtn} disabled><ArrowUp size={15} strokeWidth={2.5} /></button>
            </div>
          </div>
        </div>
      </div>
    )
  }

  /* ── Project open ── */
  const folderParts = project.folderPath.split('/')
  const shortPath = folderParts.slice(-3).join('/')

  return (
    <div className={s.panel}>
      {/* Header */}
      <header className={s.header}>
        <div className={s.headerInner}>
          <FolderOpen size={13} strokeWidth={1.8} className={s.headerIcon} />
          <span className={s.headerTitle}>{project.folderName}</span>
          <span className={s.headerPath}>{shortPath}</span>
        </div>
        <button
          className={`${s.panelToggle} ${thoughtOpen ? s.panelToggleActive : ''}`}
          onClick={onToggleThought}
          title={thoughtOpen ? 'Collapse thought panel' : 'Expand thought panel'}
        >
          <PanelIcon size={15} strokeWidth={1.8} />
        </button>
      </header>

      {/* Messages */}
      <div className={s.messages}>
        {messages.length === 0 ? (
          <ProjectEmptyState project={project} />
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={s.message}>
              {msg.role === 'user' ? (
                <div className={s.userWrap}>
                  <div className={s.userBubble}>{msg.content}</div>
                </div>
              ) : msg.subtype === 'error' ? (
                <div className={s.agentWrap}>
                  <div className={`${s.agentMd} ${s.agentMdError}`}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                </div>
              ) : (
                <div className={s.agentWrap}>
                  <div className={s.agentMd}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                </div>
              )}
            </div>
          ))
        )}

        {isLoading && <ThinkingBubble />}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className={s.inputArea}>
        <div className={s.inputBox}>
          <textarea
            ref={textareaRef}
            className={s.textarea}
            placeholder={placeholder}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading || project.indexStatus !== 'done'}
            rows={1}
          />
          <div className={s.inputFooter}>
            <div className={s.inputLeft}>
              <button className={s.iconBtn} title="Attach file" disabled={isLoading}>
                <Plus size={16} strokeWidth={2} />
              </button>
              <div style={{position: 'relative'}}>
                <button
                  className={s.modelBtn}
                  disabled={isLoading}
                  onClick={() => setShowModelDropdown(!showModelDropdown)}
                  title="Which model runs the task"
                >
                  <span>{currentModelLabel}</span>
                  <ChevronDown size={12} strokeWidth={2} />
                </button>
                {showModelDropdown && (
                  <div className={s.modelDropdown} style={dropdownStyle}>
                    {MODELS.map(m => (
                      <button
                        key={m.id}
                        onClick={() => { setSelectedModel(m.id); setShowModelDropdown(false); }}
                        style={optionStyle(selectedModel === m.id)}
                      >
                        {m.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
            <button
              className={`${s.sendBtn} ${canSend ? s.sendBtnActive : ''}`}
              disabled={!canSend}
              onClick={send}
              title="Send"
            >
              <ArrowUp size={15} strokeWidth={2.5} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
