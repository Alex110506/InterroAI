// The renderer's client for the local runtime this app started.
//
// The main process owns the runtime: it picks the port and the launch token and
// hands both over through `window.electronAPI.getRuntime()`. Every request
// carries the token, as `?token=` on WebSocket URLs, which cannot carry headers.
// Opened in a plain browser for UI work, this falls back to VITE_BACKEND_URL and
// a runtime started by hand, which needs no token.

const FALLBACK_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

let runtimePromise = null

function runtime() {
  if (!runtimePromise) {
    runtimePromise = window.electronAPI?.getRuntime
      ? window.electronAPI.getRuntime()
      : Promise.resolve({ baseUrl: FALLBACK_URL, token: null })
  }
  return runtimePromise
}

// An error from the runtime, carrying the stable `code` it sent when there is one
// (`not_signed_in`, `quota_exceeded`, …) so callers can act on it.
export class ApiError extends Error {
  constructor(message, { code = null, status = null } = {}) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
  }
}

async function request(path, options = {}) {
  const { baseUrl, token } = await runtime()
  const headers = {
    'Content-Type': 'application/json',
    ...(token ? { 'X-Interroai-Token': token } : {}),
    ...options.headers,
  }

  let res
  try {
    res = await fetch(`${baseUrl}${path}`, { ...options, headers })
  } catch {
    throw new ApiError('The InterroAI runtime is not responding.', { code: 'runtime_unreachable' })
  }
  if (res.status === 204) return null

  const body = await res.json().catch(() => null)
  if (!res.ok) {
    const detail = body?.detail
    const message = typeof detail === 'string' ? detail : detail?.message ?? `${res.status} ${res.statusText}`
    throw new ApiError(message, { code: detail?.code ?? null, status: res.status })
  }
  return body
}

async function socket(path) {
  const { baseUrl, token } = await runtime()
  const url = new URL(path, baseUrl.replace(/^http/, 'ws'))
  if (token) url.searchParams.set('token', token)
  return new WebSocket(url)
}

export const api = {
  health: () => request('/health'),

  getSettings: () => request('/api/settings'),

  saveSettings: ({ name }) =>
    request('/api/settings', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),

  getSession: () => request('/api/session'),

  signOut: () => request('/api/session', { method: 'DELETE' }),

  // The browser half runs in the main process. This resolves once the runtime holds the session.
  signIn: async () => {
    if (!window.electronAPI?.signIn) {
      throw new ApiError('Signing in needs the InterroAI desktop app.', { code: 'no_desktop_app' })
    }
    const result = await window.electronAPI.signIn()
    if (!result.ok) throw new ApiError(result.message, { code: result.code })
    return result.session
  },

  cancelSignIn: () => window.electronAPI?.cancelSignIn?.(),

  indexProject: (path) =>
    request('/api/projects/index', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),

  openChatSocket: () => socket('/api/chat/ws'),

  embedProject: async (path, onEvent) => {
    const ws = await socket('/api/projects/ws/embed')
    return new Promise((resolve, reject) => {
      ws.onopen = () => ws.send(JSON.stringify({ path }))
      ws.onmessage = (e) => {
        const event = JSON.parse(e.data)
        onEvent(event)
        if (event.step === 'done') { ws.close(); resolve() }
        if (event.step === 'error') {
          ws.close()
          reject(new ApiError(event.message, { code: event.code ?? null }))
        }
      }
      ws.onerror = () =>
        reject(new ApiError('Could not connect to the InterroAI runtime.', { code: 'runtime_unreachable' }))
    })
  },
}
