const BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const api = {
  health: () => request('/health'),

  getSettings: () => request('/api/settings'),

  saveSettings: ({ name, apiKey }) =>
    request('/api/settings', {
      method: 'POST',
      body: JSON.stringify({ name, apiKey }),
    }),

  indexProject: (path) =>
    request('/api/projects/index', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),

  openChatSocket: () => {
    const wsBase = (import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000')
      .replace(/^http/, 'ws')
    return new WebSocket(`${wsBase}/api/chat/ws`)
  },

  embedProject: (path, onEvent) =>
    new Promise((resolve, reject) => {
      const wsBase = (import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000')
        .replace(/^http/, 'ws')
      const ws = new WebSocket(`${wsBase}/api/projects/ws/embed`)
      ws.onopen = () => ws.send(JSON.stringify({ path }))
      ws.onmessage = (e) => {
        const event = JSON.parse(e.data)
        onEvent(event)
        if (event.step === 'done') { ws.close(); resolve() }
        if (event.step === 'error') { ws.close(); reject(new Error(event.message)) }
      }
      ws.onerror = () => reject(new Error('WebSocket connection failed'))
    }),
}
