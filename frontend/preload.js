const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  platform: process.platform,
  selectFolder: () => ipcRenderer.invoke('select-folder'),
  // Where the runtime this app started is listening, and the launch token it requires.
  getRuntime: () => ipcRenderer.invoke('runtime-info'),
  // Runs the browser sign-in. Resolves to { ok: true, session } or { ok: false, code, message }.
  signIn: () => ipcRenderer.invoke('cloud-sign-in'),
  cancelSignIn: () => ipcRenderer.invoke('cloud-sign-in-cancel')
})
