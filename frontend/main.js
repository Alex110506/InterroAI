const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron')
const path = require('node:path')

const { startRuntime } = require('./electron/runtime')
const { signIn } = require('./electron/signin')

const isDev = process.env.NODE_ENV === 'development'
const DEV_URL = 'http://localhost:5173'

// The runtime this app started, and the sign-in in progress, if any.
let runtime = null
let signInInProgress = null

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 960,
    minHeight: 640,
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 16, y: 18 },
    transparent: true,
    vibrancy: 'under-window',
    visualEffectState: 'active',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  })

  if (isDev) {
    win.loadURL(DEV_URL)
  } else {
    win.loadFile(path.join(__dirname, 'dist/index.html'))
  }
}

function registerIpc() {
  // Where the runtime listens, and the token it requires on every request.
  ipcMain.handle('runtime-info', () => ({ baseUrl: runtime.baseUrl, token: runtime.token }))

  ipcMain.handle('select-folder', async (event) => {
    const win = BrowserWindow.fromWebContents(event.sender)
    const { canceled, filePaths } = await dialog.showOpenDialog(win, {
      title: 'Select project folder',
      buttonLabel: 'Open project',
      properties: ['openDirectory', 'createDirectory']
    })
    return canceled ? null : filePaths[0]
  })

  // Errors cross IPC as plain values: a thrown error would lose its `code` on the way.
  ipcMain.handle('cloud-sign-in', async () => {
    signInInProgress?.abort()
    const controller = new AbortController()
    signInInProgress = controller
    try {
      const session = await signIn({
        runtime,
        openExternal: (url) => shell.openExternal(url),
        signal: controller.signal
      })
      return { ok: true, session }
    } catch (err) {
      return { ok: false, code: err.code ?? 'sign_in_failed', message: err.message }
    } finally {
      if (signInInProgress === controller) signInInProgress = null
    }
  })

  ipcMain.handle('cloud-sign-in-cancel', () => {
    signInInProgress?.abort()
  })
}

app.whenReady().then(async () => {
  try {
    // A page loaded from a file sends the origin "null"; in development it is Vite's.
    runtime = await startRuntime({ allowedOrigins: isDev ? DEV_URL : 'null' })
  } catch (err) {
    dialog.showErrorBox('InterroAI could not start', err.message)
    app.quit()
    return
  }
  registerIpc()
  createWindow()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  if (runtime && BrowserWindow.getAllWindows().length === 0) createWindow()
})

app.on('will-quit', () => {
  runtime?.stop()
})
