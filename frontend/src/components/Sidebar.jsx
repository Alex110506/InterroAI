import { Plus, Clock, Cpu, FolderOpen, Settings, X } from 'lucide-react'
import s from './Sidebar.module.css'

export default function Sidebar({ projects, activeId, onSelect, onNewProject, onOpenSettings, onDeleteProject }) {
  return (
    <aside className={s.sidebar}>
      {/* macOS traffic light spacer */}
      <div className={s.drag} />

      {/* Top navigation */}
      <nav className={s.nav}>
        <button className={s.newProject} onClick={onNewProject}>
          <Plus size={15} strokeWidth={2} />
          <span>New project</span>
        </button>
      </nav>

      {/* Project list */}
      <div className={s.projectList}>
        {projects.length === 0 ? (
          <div className={s.empty}>
            <FolderOpen size={18} strokeWidth={1.5} className={s.emptyIcon} />
            <p className={s.emptyText}>No projects yet</p>
            <p className={s.emptyHint}>Click "New project" to open a folder</p>
          </div>
        ) : (
          <>
            <span className={s.sectionLabel}>Projects</span>
            {projects.map((p) => (
              <div
                key={p.id}
                className={`${s.project} ${activeId === p.id ? s.projectActive : ''}`}
                onClick={() => onSelect(p.id)}
                title={p.folderPath}
              >
                <FolderOpen size={13} strokeWidth={1.8} className={s.folderIcon} />
                <span className={s.projectName}>{p.folderName}</span>
                <button
                  className={s.deleteBtn}
                  onClick={(e) => { e.stopPropagation(); onDeleteProject(p.id) }}
                  title="Remove project"
                >
                  <X size={12} strokeWidth={2.2} />
                </button>
              </div>
            ))}
          </>
        )}
      </div>

      {/* Bottom bar */}
      <div className={s.bottom}>
        <button className={s.settingsBtn} onClick={onOpenSettings}>
          <Settings size={14} strokeWidth={1.8} />
          <span>Settings</span>
        </button>
      </div>
    </aside>
  )
}
