import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import ChatPanel from './components/ChatPanel'
import ThoughtPanel from './components/ThoughtPanel'
import SettingsModal from './components/SettingsModal'
import { api } from './lib/api'

export default function App() {
  const [projects, setProjects] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [thoughtOpen, setThoughtOpen] = useState(true)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settings, setSettings] = useState({ name: '', hasApiKey: false })

  useEffect(() => {
    api.getSettings()
      .then((data) => setSettings({ name: data.name, hasApiKey: data.has_api_key }))
      .catch(() => {})
  }, [])

  const handleNewProject = async () => {
    const folderPath = await window.electronAPI.selectFolder()
    if (!folderPath) return

    const existing = projects.find((p) => p.folderPath === folderPath)
    if (existing) { setActiveId(existing.id); return }

    const folderName = folderPath.split('/').pop()
    const id = `proj-${Date.now()}`

    setProjects((prev) => [...prev, {
      id, folderPath, folderName,
      indexStatus: 'indexing',
      index: null,
      embedStatus: 'idle',   // 'idle' | 'running' | 'done' | 'error'
      embedSteps: null,      // { A, B, C, D } each 'idle'|'running'|'done'
      embedProgress: {},     // { files, chunks, embedded, total, stored }
    }])
    setActiveId(id)

    // ── Phase 1: file tree + git context ─────────────────────────────────────
    let phaseOneDone = false
    try {
      const index = await api.indexProject(folderPath)
      setProjects((prev) =>
        prev.map((p) => p.id === id ? { ...p, indexStatus: 'done', index } : p)
      )
      phaseOneDone = true
    } catch {
      setProjects((prev) =>
        prev.map((p) => p.id === id ? { ...p, indexStatus: 'error' } : p)
      )
    }

    if (!phaseOneDone) return

    // ── Phase 2: RAG embedding ────────────────────────────────────────────────
    setProjects((prev) =>
      prev.map((p) => p.id === id ? {
        ...p,
        embedStatus: 'running',
        embedSteps: { A: 'idle', B: 'idle', C: 'idle', D: 'idle' },
        embedProgress: {},
      } : p)
    )

    try {
      await api.embedProject(folderPath, (event) => {
        setProjects((prev) => prev.map((p) => {
          if (p.id !== id) return p
          const steps = { ...(p.embedSteps ?? {}) }
          const progress = { ...p.embedProgress }
          const { step, status } = event

          if (['A', 'B', 'C', 'D'].includes(step)) {
            if (status === 'start') steps[step] = 'running'
            else if (status === 'done') steps[step] = 'done'
            else if (status === 'progress') steps[step] = 'running'
          }
          if (step === 'A' && event.files != null) progress.files = event.files
          if (step === 'B' && event.chunks != null) progress.chunks = event.chunks
          if (step === 'C' && event.embedded != null) {
            progress.embedded = event.embedded
            progress.total = event.total
          }
          if (step === 'D' && event.stored != null) progress.stored = event.stored

          return { ...p, embedSteps: steps, embedProgress: progress }
        }))
      })
      setProjects((prev) =>
        prev.map((p) => p.id === id ? { ...p, embedStatus: 'done' } : p)
      )
    } catch {
      setProjects((prev) =>
        prev.map((p) => p.id === id ? { ...p, embedStatus: 'error' } : p)
      )
    }
  }

  const handleDeleteProject = (id) => {
    setProjects((prev) => prev.filter((p) => p.id !== id))
    if (activeId === id) setActiveId(null)
  }

  const handleSettingsSaved = (next) => {
    setSettings(next)
    setSettingsOpen(false)
  }

  return (
    <div
      className="layout"
      style={{ '--thought-col': thoughtOpen ? 'var(--thought-w)' : '0px' }}
    >
      <Sidebar
        projects={projects}
        activeId={activeId}
        onSelect={setActiveId}
        onNewProject={handleNewProject}
        onDeleteProject={handleDeleteProject}
        onOpenSettings={() => setSettingsOpen(true)}
      />
      <ChatPanel
        activeId={activeId}
        projects={projects}
        thoughtOpen={thoughtOpen}
        onToggleThought={() => setThoughtOpen((o) => !o)}
      />
      <ThoughtPanel isOpen={thoughtOpen} />

      {settingsOpen && (
        <SettingsModal
          settings={settings}
          onSave={handleSettingsSaved}
          onClose={() => setSettingsOpen(false)}
        />
      )}
    </div>
  )
}
