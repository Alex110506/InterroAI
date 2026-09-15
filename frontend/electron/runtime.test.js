// Starting the runtime. The last test starts the real backend, so it needs
// python3 with the backend's dependencies installed; it is skipped otherwise.
// Run with: npm test

const assert = require('node:assert/strict')
const { spawnSync } = require('node:child_process')
const fs = require('node:fs')
const http = require('node:http')
const net = require('node:net')
const os = require('node:os')
const path = require('node:path')
const { test } = require('node:test')

const { BACKEND_DIR, findFreePort, startRuntime, waitForHealth } = require('./runtime')

const quiet = { stdout: { write() {} }, stderr: { write() {} } }

function serve(handler) {
  return new Promise((resolve) => {
    const server = http.createServer(handler)
    server.listen(0, '127.0.0.1', () =>
      resolve({ server, baseUrl: `http://127.0.0.1:${server.address().port}` }),
    )
  })
}

test('a free port can be listened on', async () => {
  const port = await findFreePort()
  const server = net.createServer()
  await new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(port, '127.0.0.1', resolve)
  })
  server.close()
})

test('waiting for health returns once the runtime answers', async () => {
  let calls = 0
  const { server, baseUrl } = await serve((req, res) => {
    calls += 1
    res.statusCode = calls < 3 ? 503 : 200
    res.end('{}')
  })
  try {
    await waitForHealth(baseUrl, { intervalMs: 10 })
    assert.equal(calls, 3)
  } finally {
    server.closeAllConnections()
    server.close()
  }
})

test('waiting for health gives up at once when the process has stopped', async () => {
  const port = await findFreePort()
  await assert.rejects(
    waitForHealth(`http://127.0.0.1:${port}`, { intervalMs: 10, exited: () => 'exit 1' }),
    /stopped while starting/,
  )
})

const python = process.env.INTERROAI_PYTHON || 'python3'
const backendReady = spawnSync(python, ['-c', 'import fastapi, uvicorn'], { cwd: BACKEND_DIR }).status === 0

test(
  'the real runtime answers only requests that carry its launch token',
  { skip: !backendReady && `${python} with the backend dependencies is not available`, timeout: 120_000 },
  async () => {
    // A throwaway home, so the runtime writes nothing into the developer's ~/.interroai.
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'interroai-runtime-test-'))
    const runtime = await startRuntime({
      env: { ...process.env, HOME: home, INTERROAI_MODE: 'local' },
      allowedOrigins: 'http://localhost:5173',
      output: quiet,
    })
    try {
      assert.equal((await fetch(`${runtime.baseUrl}/health`)).status, 200)
      assert.equal((await fetch(`${runtime.baseUrl}/api/session`)).status, 401)

      const answered = await fetch(`${runtime.baseUrl}/api/session`, {
        headers: { 'X-Interroai-Token': runtime.token },
      })
      assert.equal(answered.status, 200)
      assert.equal((await answered.json()).mode, 'local')
    } finally {
      runtime.stop()
      fs.rmSync(home, { recursive: true, force: true })
    }
  },
)
