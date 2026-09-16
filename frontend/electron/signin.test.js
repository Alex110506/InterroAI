// The browser half of sign-in, against a stand-in runtime and a scripted browser.
// Run with: npm test

const assert = require('node:assert/strict')
const crypto = require('node:crypto')
const http = require('node:http')
const { test } = require('node:test')

const { pkcePair, signIn } = require('./signin')

const CLOUD = 'https://cloud.example'
const TOKEN = 'launch-token'

// The runtime's /api/session, answering as a cloud runtime would.
async function fakeRuntime({ mode = 'cloud', acceptCode = 'login-code' } = {}) {
  const calls = []
  const server = http.createServer(async (req, res) => {
    let raw = ''
    for await (const chunk of req) raw += chunk
    calls.push({ method: req.method, token: req.headers['x-interroai-token'], body: raw ? JSON.parse(raw) : null })
    res.setHeader('Content-Type', 'application/json')
    if (req.method === 'GET') {
      res.end(JSON.stringify({ mode, api_url: mode === 'cloud' ? `${CLOUD}/` : null, signed_in: false }))
    } else if (JSON.parse(raw).code === acceptCode) {
      res.end(JSON.stringify({ mode: 'cloud', api_url: CLOUD, signed_in: true, login: 'octocat', avatar_url: null }))
    } else {
      res.statusCode = 400
      res.end(JSON.stringify({ detail: { code: 'invalid_grant', message: 'The sign-in code is invalid or has expired.' } }))
    }
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const close = () => {
    server.closeAllConnections()
    server.close()
  }
  return { calls, close, runtime: { baseUrl: `http://127.0.0.1:${server.address().port}`, token: TOKEN } }
}

// A browser that, once opened, goes straight back to the app with each set of
// query parameters in turn, as the Cloud API would send it. The last visit is
// not awaited: the app answers it only after finishing sign-in.
function scriptedBrowser(...visits) {
  const browser = { opened: [], pages: [] }
  browser.openExternal = async (url) => {
    browser.opened.push(url)
    const start = new URL(url)
    for (const [index, params] of visits.entries()) {
      const back = new URL(start.searchParams.get('redirect_uri'))
      back.search = new URLSearchParams({ state: start.searchParams.get('state'), ...params }).toString()
      const visit = fetch(back).then(async (r) => ({ status: r.status, html: await r.text() }))
      browser.pages.push(visit)
      if (index < visits.length - 1) await visit
    }
  }
  return browser
}

const sha256 = (text) => crypto.createHash('sha256').update(text).digest('base64url')

test('a verifier and its challenge are an S256 pair', () => {
  const { verifier, challenge } = pkcePair()
  assert.match(verifier, /^[A-Za-z0-9_-]{43}$/)
  assert.equal(challenge, sha256(verifier))
})

test('the login code comes back to the app and goes to the runtime with its verifier', async () => {
  const { calls, close, runtime } = await fakeRuntime()
  const browser = scriptedBrowser({ code: 'login-code' })
  try {
    const session = await signIn({ runtime, openExternal: browser.openExternal })

    assert.equal(session.login, 'octocat')
    const opened = new URL(browser.opened[0])
    assert.equal(`${opened.origin}${opened.pathname}`, `${CLOUD}/auth/github/start`)
    assert.equal(opened.searchParams.get('code_challenge_method'), 'S256')
    assert.match(opened.searchParams.get('redirect_uri'), /^http:\/\/127\.0\.0\.1:\d+\/callback$/)

    const put = calls.find((call) => call.method === 'PUT')
    assert.equal(put.body.code, 'login-code')
    assert.equal(sha256(put.body.code_verifier), opened.searchParams.get('code_challenge'))
    assert.ok(calls.every((call) => call.token === TOKEN), 'every call to the runtime carries the launch token')

    const page = await browser.pages[0]
    assert.equal(page.status, 200)
    assert.match(page.html, /Signed in/)
  } finally {
    close()
  }
})

test('a refusal on the way back fails the sign-in with its code', async () => {
  const { calls, close, runtime } = await fakeRuntime()
  const browser = scriptedBrowser({ error: 'access_denied' })
  try {
    await assert.rejects(signIn({ runtime, openExternal: browser.openExternal }), { code: 'access_denied' })
    assert.ok(!calls.some((call) => call.method === 'PUT'), 'nothing is redeemed')
    assert.equal((await browser.pages[0]).status, 403)
  } finally {
    close()
  }
})

test('a visit with some other state is turned away, and the real one still signs in', async () => {
  const { close, runtime } = await fakeRuntime()
  const browser = scriptedBrowser({ state: 'forged', code: 'stolen-code' }, { code: 'login-code' })
  try {
    const session = await signIn({ runtime, openExternal: browser.openExternal })
    assert.equal(session.login, 'octocat')
    assert.equal((await browser.pages[0]).status, 400)
  } finally {
    close()
  }
})

test('the runtime refusing the code fails the sign-in, and the browser is told', async () => {
  const { close, runtime } = await fakeRuntime({ acceptCode: 'some-other-code' })
  const browser = scriptedBrowser({ code: 'login-code' })
  try {
    await assert.rejects(signIn({ runtime, openExternal: browser.openExternal }), { code: 'invalid_grant' })
    assert.equal((await browser.pages[0]).status, 400)
  } finally {
    close()
  }
})

test('a runtime in local mode does not start a sign-in', async () => {
  const { close, runtime } = await fakeRuntime({ mode: 'local' })
  const browser = scriptedBrowser({ code: 'login-code' })
  try {
    await assert.rejects(signIn({ runtime, openExternal: browser.openExternal }), { code: 'local_mode' })
    assert.equal(browser.opened.length, 0)
  } finally {
    close()
  }
})

test('a sign-in nobody finishes times out', async () => {
  const { close, runtime } = await fakeRuntime()
  try {
    await assert.rejects(signIn({ runtime, openExternal: async () => {}, timeoutMs: 50 }), { code: 'timeout' })
  } finally {
    close()
  }
})

test('the app can cancel a sign-in in progress', async () => {
  const { close, runtime } = await fakeRuntime()
  const controller = new AbortController()
  try {
    const pending = signIn({
      runtime,
      openExternal: async () => { setTimeout(() => controller.abort(), 20) },
      signal: controller.signal,
    })
    await assert.rejects(pending, { code: 'cancelled' })
  } finally {
    close()
  }
})
