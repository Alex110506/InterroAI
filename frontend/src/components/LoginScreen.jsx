import { useState } from 'react'
import { Brain, Github, Loader2, AlertCircle } from 'lucide-react'
import { api } from '../lib/api'
import s from './LoginScreen.module.css'

/* Shown in cloud mode until the runtime holds a session. The browser half of
 * sign-in runs in the main process; this screen only starts it and waits. */
export default function LoginScreen({ onSignedIn }) {
  const [status, setStatus] = useState('idle') // 'idle' | 'waiting' | 'error'
  const [error, setError] = useState('')

  const start = async () => {
    setStatus('waiting')
    setError('')
    try {
      onSignedIn(await api.signIn())
    } catch (err) {
      if (err.code === 'cancelled') {
        setStatus('idle')
        return
      }
      setError(err.message)
      setStatus('error')
    }
  }

  return (
    <div className={s.screen}>
      <div className={s.drag} />
      <div className={s.card}>
        <div className={s.logo}>
          <Brain size={28} strokeWidth={1.4} />
        </div>
        <h1 className={s.title}>Sign in to InterroAI</h1>
        <p className={s.desc}>
          Your files stay on this computer and the agent edits them here. The cloud keeps a search
          index of your project, never the code itself.
        </p>

        {status === 'error' && (
          <div className={s.error}>
            <AlertCircle size={14} strokeWidth={1.8} />
            <span>{error}</span>
          </div>
        )}

        {status === 'waiting' ? (
          <>
            <button className={s.primary} disabled>
              <Loader2 size={15} strokeWidth={2} className={s.spin} />
              Waiting for your browser…
            </button>
            <button className={s.secondary} onClick={() => api.cancelSignIn()}>
              Cancel
            </button>
          </>
        ) : (
          <button className={s.primary} onClick={start}>
            <Github size={15} strokeWidth={2} />
            Sign in with GitHub
          </button>
        )}

        <p className={s.hint}>Finish signing in in the browser tab that opens, then come back here.</p>
      </div>
    </div>
  )
}
