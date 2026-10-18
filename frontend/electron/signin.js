// Signing in to the InterroAI cloud: the browser half, run by the main process.
//
//   1. Ask the runtime where to sign in: the Auth0 tenant, this app's client id
//      and the Cloud API's audience (GET /api/session, from the API's /auth/config).
//   2. Make a PKCE verifier and its challenge, and a state value.
//   3. Listen on 127.0.0.1 for the browser to come back, on the first free port
//      of LOOPBACK_PORTS. Auth0 matches callback URLs exactly, port included, so
//      each of these is registered there; a random port would be refused.
//   4. Open Auth0's /authorize in the user's browser, sent straight on to GitHub.
//   5. Auth0 sends the browser back here with an authorization code.
//   6. Hand the code, the verifier and the redirect URI to the runtime
//      (PUT /api/session), which redeems them at Auth0 and keeps the tokens.
//      They never pass through this process.
//
// The verifier stays in this process until step 6, and step 6 goes only to the
// runtime this app started. A code intercepted on its way through the browser
// is useless without it.

const crypto = require('node:crypto')
const http = require('node:http')

const TIMEOUT_MS = 5 * 60 * 1000
// Each is registered in Auth0 as an allowed callback, http://127.0.0.1:<port>/callback.
const LOOPBACK_PORTS = [47213, 47214, 47215]
const SCOPES = 'openid profile offline_access'

const REFUSALS = {
  access_denied: 'Sign-in was cancelled, or GitHub refused it.',
  server_error: 'GitHub sign-in failed. Try again in a moment.',
  temporarily_unavailable: 'Sign-in is unavailable right now. Try again in a moment.',
}

class SignInError extends Error {
  constructor(code, message) {
    super(message)
    this.name = 'SignInError'
    this.code = code
  }
}

const base64url = (bytes) => Buffer.from(bytes).toString('base64url')

function pkcePair() {
  const verifier = base64url(crypto.randomBytes(32))
  const challenge = base64url(crypto.createHash('sha256').update(verifier).digest())
  return { verifier, challenge }
}

const escapeHtml = (text) =>
  String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c])

function page(title, message) {
  return `<!doctype html><meta charset="utf-8"><title>InterroAI</title>
<style>body{margin:0;height:100vh;display:grid;place-items:center;background:#0d0e18;color:#eef0fa;font-family:-apple-system,BlinkMacSystemFont,Inter,sans-serif}main{max-width:420px;padding:24px;text-align:center}h1{font-size:18px;font-weight:600;margin:0 0 8px}p{margin:0;color:#aab0c8;font-size:14px;line-height:1.5}</style>
<main><h1>${escapeHtml(title)}</h1><p>${escapeHtml(message)}</p></main>`
}

function listenOn(port) {
  return new Promise((resolve, reject) => {
    const server = http.createServer()
    server.once('error', reject)
    server.listen(port, '127.0.0.1', () => resolve(server))
  })
}

async function listen(ports) {
  for (const port of ports) {
    try {
      return await listenOn(port)
    } catch (err) {
      if (err.code !== 'EADDRINUSE') throw err
    }
  }
  throw new SignInError(
    'port_in_use',
    `InterroAI could not wait for the sign-in to come back: ports ${ports.join(', ')} are all in use.`,
  )
}

// Resolves with the login code when the browser comes back with this sign-in's
// state; rejects on a refusal, the timeout, or the app cancelling.
function waitForCallback(server, { state, timeoutMs, signal }) {
  return new Promise((resolve, reject) => {
    const finish = (settle) => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', onAbort)
      settle()
    }
    const onAbort = () => finish(() => reject(new SignInError('cancelled', 'Sign-in was cancelled.')))
    const timer = setTimeout(
      () => finish(() => reject(new SignInError('timeout', 'Sign-in timed out. Try again.'))),
      timeoutMs,
    )
    if (signal?.aborted) {
      onAbort()
      return
    }
    signal?.addEventListener('abort', onAbort, { once: true })

    server.on('request', (req, res) => {
      const reply = (status, title, message) => {
        res.writeHead(status, { 'Content-Type': 'text/html; charset=utf-8', Connection: 'close' })
        res.end(page(title, message))
      }
      const url = new URL(req.url, 'http://127.0.0.1')
      if (url.pathname !== '/callback') {
        res.writeHead(404, { Connection: 'close' })
        res.end()
        return
      }
      if (url.searchParams.get('state') !== state) {
        // Not the sign-in this app started. Turn it away and keep waiting.
        reply(400, 'Sign-in failed', 'This link does not belong to the sign-in the app started. Start again from InterroAI.')
        return
      }

      const error = url.searchParams.get('error')
      const code = url.searchParams.get('code')
      if (error || !code) {
        const message = REFUSALS[error] ?? 'Sign-in failed.'
        reply(403, 'Sign-in failed', message)
        finish(() => reject(new SignInError(error ?? 'invalid_request', message)))
        return
      }
      finish(() => resolve({ code, reply }))
    })
  })
}

async function signIn({
  runtime,
  openExternal,
  fetchImpl = fetch,
  timeoutMs = TIMEOUT_MS,
  signal,
  ports = LOOPBACK_PORTS,
} = {}) {
  const headers = { 'Content-Type': 'application/json', 'X-Interroai-Token': runtime.token }
  const session = await (await fetchImpl(`${runtime.baseUrl}/api/session`, { headers })).json()
  if (!session.auth0) {
    throw new SignInError('cloud_unavailable', 'The InterroAI cloud cannot be reached. Check your connection and try again.')
  }

  const { verifier, challenge } = pkcePair()
  const state = base64url(crypto.randomBytes(16))
  const server = await listen(ports)
  try {
    const redirectUri = `http://127.0.0.1:${server.address().port}/callback`
    const query = new URLSearchParams({
      response_type: 'code',
      client_id: session.auth0.client_id,
      audience: session.auth0.audience,
      scope: SCOPES,
      // Straight to GitHub, past Auth0's own login page: it is the only way in.
      connection: 'github',
      redirect_uri: redirectUri,
      code_challenge: challenge,
      code_challenge_method: 'S256',
      state,
    })
    const callback = waitForCallback(server, { state, timeoutMs, signal })
    // Handled here too, so a cancel arriving while the browser opens is not an
    // unhandled rejection; the await below still sees it.
    callback.catch(() => {})
    await openExternal(`https://${session.auth0.domain}/authorize?${query}`)
    const { code, reply } = await callback

    const res = await fetchImpl(`${runtime.baseUrl}/api/session`, {
      method: 'PUT',
      headers,
      body: JSON.stringify({ code, code_verifier: verifier, redirect_uri: redirectUri }),
    })
    const body = await res.json().catch(() => ({}))
    if (!res.ok) {
      const message = body.detail?.message ?? 'The app could not finish signing in.'
      reply(400, 'Sign-in failed', message)
      throw new SignInError(body.detail?.code ?? 'sign_in_failed', message)
    }
    reply(200, 'Signed in to InterroAI', 'You can close this tab and go back to the app.')
    return body
  } finally {
    server.close()
    server.closeIdleConnections?.()
  }
}

module.exports = { LOOPBACK_PORTS, SignInError, pkcePair, signIn }
