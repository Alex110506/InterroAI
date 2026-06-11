import { useState, useEffect, useRef } from 'react'
import { X, User, Key, Eye, EyeOff, Save, AlertCircle, CheckCircle2 } from 'lucide-react'
import { api } from '../lib/api'
import s from './SettingsModal.module.css'

export default function SettingsModal({ settings, onSave, onClose }) {
  const [name, setName] = useState(settings.name ?? '')
  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [status, setStatus] = useState(null) // null | 'saving' | 'success' | 'error'
  const [errorMsg, setErrorMsg] = useState('')
  const nameRef = useRef(null)

  // Close on Escape
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  useEffect(() => { nameRef.current?.focus() }, [])

  const handleSave = async () => {
    setStatus('saving')
    setErrorMsg('')
    try {
      await api.saveSettings({ name: name.trim(), apiKey: apiKey.trim() })
      setStatus('success')
      setTimeout(() => {
        onSave({
          name: name.trim(),
          hasApiKey: settings.hasApiKey || !!apiKey.trim(),
        })
      }, 600)
    } catch {
      setStatus('error')
      setErrorMsg('Could not reach the backend. Make sure it is running on port 8000.')
    }
  }

  const isSaving = status === 'saving'

  return (
    <div
      className={s.backdrop}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className={s.modal} role="dialog" aria-modal="true">
        {/* Header */}
        <div className={s.header}>
          <span className={s.title}>Settings</span>
          <button className={s.closeBtn} onClick={onClose} title="Close">
            <X size={16} strokeWidth={2} />
          </button>
        </div>

        {/* Body */}
        <div className={s.body}>
          {status === 'error' && (
            <div className={s.errorBanner}>
              <AlertCircle size={14} strokeWidth={1.8} />
              {errorMsg}
            </div>
          )}

          {/* Name */}
          <div className={s.field}>
            <label className={s.label} htmlFor="s-name">
              <User size={13} strokeWidth={1.8} />
              Your name
            </label>
            <input
              ref={nameRef}
              id="s-name"
              type="text"
              className={s.input}
              placeholder="Alex"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleSave() }}
              autoComplete="off"
            />
          </div>

          {/* API key */}
          <div className={s.field}>
            <label className={s.label} htmlFor="s-key">
              <Key size={13} strokeWidth={1.8} />
              OpenAI API key
              {settings.hasApiKey && (
                <span className={s.savedBadge}>
                  <CheckCircle2 size={11} strokeWidth={2} />
                  Saved in keychain
                </span>
              )}
            </label>
            <div className={s.inputRow}>
              <input
                id="s-key"
                type={showKey ? 'text' : 'password'}
                className={`${s.input} ${s.mono}`}
                placeholder={settings.hasApiKey ? 'Enter a new key to update…' : 'sk-…'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleSave() }}
                autoComplete="off"
                spellCheck={false}
              />
              <button
                className={s.eyeBtn}
                onClick={() => setShowKey((v) => !v)}
                type="button"
                title={showKey ? 'Hide' : 'Show'}
              >
                {showKey
                  ? <EyeOff size={14} strokeWidth={1.8} />
                  : <Eye size={14} strokeWidth={1.8} />}
              </button>
            </div>
            <p className={s.hint}>
              Stored in the macOS Keychain via <code>keyring</code>. Never sent anywhere except the OpenAI API.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className={s.footer}>
          <button className={s.cancelBtn} onClick={onClose}>Cancel</button>
          <button
            className={`${s.saveBtn} ${status === 'success' ? s.saveBtnSuccess : ''}`}
            onClick={handleSave}
            disabled={isSaving}
          >
            {status === 'success'
              ? <><CheckCircle2 size={14} strokeWidth={2} /> Saved</>
              : <><Save size={14} strokeWidth={2} /> {isSaving ? 'Saving…' : 'Save'}</>}
          </button>
        </div>
      </div>
    </div>
  )
}
