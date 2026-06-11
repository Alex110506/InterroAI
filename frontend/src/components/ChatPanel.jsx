import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Plus, ChevronDown, ArrowUp, Brain, FolderOpen,
  PanelRight, PanelRightClose,
  GitBranch, FileCode2, Layers, AlertTriangle, CheckCircle2, Sparkles, X,
} from 'lucide-react'
import { api } from '../lib/api'
import s from './ChatPanel.module.css'

/* ─── Per-project message history (runtime, no persistence yet) ─────── */
const projectMessages = {}

/* ─── Embed step indicators ─────────────────────────────────────────── */
const STEP_LABELS = { A: 'Scanning files', B: 'Chunking code', C: 'Generating embeddings', D: 'Storing vectors' }

function EmbedProgress({ embedStatus, embedSteps, embedProgress }) {
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
        Embedding failed — check your API key in Settings
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
  const { indexStatus, index, folderName, embedStatus, embedSteps, embedProgress } = project

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
        The agent will ask you a few targeted questions to clarify your request
        before starting work on <strong>{folderName}</strong>.
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

      <EmbedProgress embedStatus={embedStatus} embedSteps={embedSteps} embedProgress={embedProgress} />
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
  { id: 'auto',               label: 'Auto' },
  { id: 'gpt-5.5-high-effort', label: 'GPT-5.5 High Effort' },
  { id: 'gpt-5.5-low-effort',  label: 'GPT-5.5 Low Effort' },
  { id: 'gpt-5.4-high-effort', label: 'GPT-5.4 High Effort' },
  { id: 'gpt-5.4-low-effort',  label: 'GPT-5.4 Low Effort' },
  { id: 'gpt-5.4-mini',        label: 'GPT-5.4 Mini' },
]

/* ─── Main ChatPanel ─────────────────────────────────────────────────── */
export default function ChatPanel({ activeId, projects, thoughtOpen, onToggleThought, onThoughtEvent, clearThought, onBusyChange }) {
  const [input, setInput] = useState('')
  const [selectedModel, setSelectedModel] = useState('auto')
  const [showModelDropdown, setShowModelDropdown] = useState(false)
  const [phase, setPhase] = useState('idle')   // 'idle' | 'interrogating' | 'ready'
  const [isLoading, setIsLoading] = useState(false)
  const [, forceUpdate] = useState(0)
  const textareaRef = useRef(null)
  const bottomRef = useRef(null)
  const wsRef = useRef(null)

  const project = projects.find((p) => p.id === activeId) ?? null

  /* ── Surface "model is working" state to the app shell (drives RunningWave) ── */
  useEffect(() => {
    onBusyChange?.(isLoading || phase === 'ready')
  }, [isLoading, phase, onBusyChange])

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
    setPhase('idle')
    setIsLoading(false)
    setInput('')
    bottomRef.current?.scrollIntoView({ behavior: 'instant' })
  }, [activeId])

  const messages = activeId ? (projectMessages[activeId] ?? []) : []

  const addMessage = useCallback((msg) => {
    if (!activeId) return
    if (!projectMessages[activeId]) projectMessages[activeId] = []
    projectMessages[activeId].push({ id: Date.now() + Math.random(), ...msg })
    forceUpdate((n) => n + 1)
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)
  }, [activeId])

  /* ── WebSocket message handler ── */
  const handleWsMessage = useCallback((event) => {
    if (event.type === 'message') {
      setIsLoading(false)
      addMessage({ role: 'agent', subtype: 'message', content: event.content })
      setPhase('idle')
      wsRef.current?.close()
      wsRef.current = null

    } else if (event.type === 'question') {
      setIsLoading(false)
      setPhase('interrogating')
      addMessage({ role: 'agent', subtype: 'question', content: event.question, turn: event.turn })

    } else if (event.type === 'ready') {
      // Coder is starting — keep WebSocket open, keep loading indicator
      if (event.did_interrogate) {
        addMessage({ role: 'agent', subtype: 'ready', content: event.refined_prompt })
      }
      setPhase('ready')

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
      setPhase('idle')
      wsRef.current?.close()
      wsRef.current = null

    } else if (event.type === 'error') {
      setIsLoading(false)
      addMessage({ role: 'agent', subtype: 'error', content: event.message })
      setPhase('idle')
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [addMessage, onThoughtEvent])

  /* ── Send handler ── */
  const send = () => {
    const text = input.trim()
    if (!text || !activeId || isLoading || project?.indexStatus !== 'done') return

    addMessage({ role: 'user', content: text })
    setInput('')
    setIsLoading(true)
    clearThought?.()

    if (phase === 'idle' || phase === 'ready') {
      // Open socket and let the backend decide: direct answer or interrogation
      // Phase stays 'idle' until the backend sends a 'question' event
      const ws = api.openChatSocket()
      wsRef.current = ws

      ws.onopen = () => {
        ws.send(JSON.stringify({
          type: 'start',
          project_path: project.folderPath,
          project_index: project.index ?? {},
          message: text,
          model: selectedModel,
        }))
      }

      ws.onmessage = (e) => handleWsMessage(JSON.parse(e.data))

      ws.onerror = () => {
        setIsLoading(false)
        addMessage({ role: 'agent', subtype: 'error', content: 'Connection to backend failed. Is the server running?' })
        setPhase('idle')
      }

      ws.onclose = () => {
        setIsLoading((prev) => (prev ? false : prev))
      }

    } else if (phase === 'interrogating') {
      // Send answer to the active session
      wsRef.current?.send(JSON.stringify({ type: 'answer', message: text }))
    }
  }

  const cancelInterrogation = () => {
    if (!wsRef.current) return
    setIsLoading(true)
    wsRef.current.send(JSON.stringify({ type: 'force_ready' }))
    // The socket will receive the normal 'ready' event and resolve as usual
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
    phase === 'interrogating' ? 'Answer the question…' :
    `Ask about ${project.folderName}…`

  const currentModelLabel = MODELS.find(m => m.id === selectedModel)?.label ?? 'Auto'

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
    <div className={`${s.panel} ${phase === 'interrogating' ? s.panelInterrogating : ''}`}>
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

      {/* Interrogation mode bar */}
      {phase === 'interrogating' && (
        <div className={s.interrogationBar}>
          <span className={s.interrogationDot} />
          Interrogation mode
          <span className={s.interrogationHint}>Answer the questions to refine your request</span>
          <button
            className={s.cancelBtn}
            onClick={cancelInterrogation}
            disabled={isLoading}
            title="Generate refined prompt from answers so far"
          >
            <X size={12} strokeWidth={2.5} />
            Cancel
          </button>
        </div>
      )}

      {/* Messages */}
      <div className={`${s.messages} ${phase === 'interrogating' ? s.messagesInterrogating : ''}`}>
        {messages.length === 0 ? (
          <ProjectEmptyState project={project} />
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={s.message}>
              {msg.role === 'user' ? (
                <div className={s.userWrap}>
                  <div className={s.userBubble}>{msg.content}</div>
                </div>
              ) : msg.subtype === 'question' ? (
                <div className={s.agentWrap}>
                  <span className={s.questionLabel}>
                    <Sparkles size={10} strokeWidth={2} />
                    Clarifying question {msg.turn}
                  </span>
                  <div className={s.agentMd}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                </div>
              ) : msg.subtype === 'ready' ? (
                <div className={s.agentWrap}>
                  <div className={s.refinedCard}>
                    <div className={s.refinedHeader}>
                      <CheckCircle2 size={13} strokeWidth={2} />
                      Prompt refined — ready to execute
                    </div>
                    <p className={s.refinedContent}>{msg.content}</p>
                  </div>
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
        <div className={`${s.inputBox} ${phase === 'interrogating' ? s.inputBoxActive : ''}`}>
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
                >
                  <span>{currentModelLabel}</span>
                  <ChevronDown size={12} strokeWidth={2} />
                </button>
                {showModelDropdown && (
                  <div className={s.modelDropdown} style={{
                    position: 'absolute', bottom: '100%', left: 0, 
                    backgroundColor: 'var(--bg)', border: '1px solid var(--border)',
                    borderRadius: '6px', padding: '4px', zIndex: 10, marginBottom: '4px',
                    width: 'max-content', display: 'flex', flexDirection: 'column', gap: '2px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                  }}>
                    {MODELS.map(m => (
                      <button 
                        key={m.id}
                        onClick={() => { setSelectedModel(m.id); setShowModelDropdown(false); }}
                        style={{
                          background: selectedModel === m.id ? 'var(--bg-2)' : 'transparent',
                          border: 'none', color: 'var(--text-1)', padding: '6px 12px',
                          textAlign: 'left', borderRadius: '4px', cursor: 'pointer',
                          fontSize: '13px'
                        }}
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
