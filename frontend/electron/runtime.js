// The local runtime (backend/), started and owned by the Electron main process.
//
// It listens on a free port on 127.0.0.1 and answers only requests carrying the
// launch token made here (backend/api/guard.py), so no other program on this
// machine, and no web page, can drive the agent. The renderer learns the
// address and the token over IPC; see preload.js.
//
// Development only for now: it runs `python3` from PATH (or INTERROAI_PYTHON)
// in the repo's backend/ folder. A packaged app would ship its own runtime.

const { spawn } = require('node:child_process')
const crypto = require('node:crypto')
const net = require('node:net')
const path = require('node:path')

const BACKEND_DIR = path.join(__dirname, '..', '..', 'backend')

function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address()
      server.close(() => resolve(port))
    })
  })
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

// Resolves once GET /health answers. `exited` reports why the process stopped,
// if it has, so a crash on startup fails at once instead of at the timeout.
async function waitForHealth(baseUrl, { timeoutMs = 60_000, intervalMs = 250, exited = () => null } = {}) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const exit = exited()
    if (exit) throw new Error(`The InterroAI runtime stopped while starting (${exit}).`)
    try {
      const res = await fetch(`${baseUrl}/health`)
      if (res.ok) return
    } catch {
      // Not listening yet.
    }
    await sleep(intervalMs)
  }
  throw new Error(`The InterroAI runtime did not start within ${Math.round(timeoutMs / 1000)} seconds.`)
}

async function startRuntime({
  python = process.env.INTERROAI_PYTHON || 'python3',
  cwd = BACKEND_DIR,
  env = process.env,
  allowedOrigins = '*',
  output = process,
} = {}) {
  const port = await findFreePort()
  const token = crypto.randomBytes(32).toString('hex')
  const baseUrl = `http://127.0.0.1:${port}`

  const child = spawn(
    python,
    // No access log: WebSocket URLs carry the token, and the log would print it.
    ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', String(port), '--no-access-log'],
    {
      cwd,
      env: { ...env, INTERROAI_LAUNCH_TOKEN: token, INTERROAI_ALLOWED_ORIGINS: allowedOrigins },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  )

  let exit = null
  child.on('error', (err) => { exit = exit ?? err.message })
  child.on('exit', (code, signal) => { exit = exit ?? `exit ${signal ?? code}` })
  child.stdout.on('data', (chunk) => output.stdout.write(`[runtime] ${chunk}`))
  child.stderr.on('data', (chunk) => output.stderr.write(`[runtime] ${chunk}`))

  try {
    await waitForHealth(baseUrl, { exited: () => exit })
  } catch (err) {
    if (!exit) child.kill()
    throw err
  }

  return {
    baseUrl,
    token,
    stop() {
      if (!exit) child.kill()
    },
  }
}

module.exports = { BACKEND_DIR, findFreePort, startRuntime, waitForHealth }
