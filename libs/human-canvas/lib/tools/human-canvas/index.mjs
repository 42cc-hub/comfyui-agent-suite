// human-canvas / lib/tools/human-canvas/index.mjs
// agent 侧 API：开画布 → 通知人 → 【让出回合】→ 人画完提交并回复 → 读回结构化输入。
//
// 与 ps 工具同一套非阻塞约定：openHumanCanvas() 立即返回 URL，绝不在这里等人。
//
// 服务跑在 detached 子进程里（不随 agent 回合结束而死），位置记在状态文件里，
// 因此后续任意进程都能用 canvasStatus() 读到结果。
import { spawn } from 'node:child_process'
import { closeSync, openSync } from 'node:fs'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { DEFAULT_PORT } from './server.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const SERVER = join(HERE, 'server.mjs')
const LEGACY_STATE = join(HERE, 'canvas-state.json')

// 状态属运行期数据，不该住在包目录里：那样会被打进发布物、被重装/更新覆盖，
// 也会被同一台机器上的多个 profile 共用。默认落到用户数据目录，并保留环境变量覆盖
// （测试靠它做隔离，见 test/human-canvas/smoke.mjs）。
function defaultStatePath() {
  const base = process.env.LOCALAPPDATA || process.env.XDG_STATE_HOME || join(homedir(), '.local', 'state')
  return join(base, 'human-canvas', 'canvas-state.json')
}

const STATE = process.env.HUMAN_CANVAS_STATE || defaultStatePath()
// task → outDir 的登记表。单槽服务被下一个任务 reconfigure 之后，上一轮的结果仍能按 task 找回。
const TASKS = process.env.HUMAN_CANVAS_TASKS || join(dirname(STATE), 'canvas-tasks.json')
const TASK_HISTORY = 50

const readState = async () => {
  for (const f of new Set([STATE, LEGACY_STATE])) {
    try { return JSON.parse(await readFile(f, 'utf8')) } catch { /* 试下一个 */ }
  }
  return null
}

async function readTasks() {
  try { return JSON.parse(await readFile(TASKS, 'utf8')) } catch { return [] }
}

async function rememberTask(entry) {
  const list = (await readTasks()).filter((t) => t.task !== entry.task)
  list.unshift(entry)
  await mkdir(dirname(TASKS), { recursive: true })
  await writeFile(TASKS, JSON.stringify(list.slice(0, TASK_HISTORY), null, 1), 'utf8')
}

async function writeState(obj) {
  await mkdir(dirname(STATE), { recursive: true })
  await writeFile(STATE, JSON.stringify(obj, null, 1), 'utf8')
  // 一次性迁移：新状态写成功就把包目录里那份旧的清掉，避免下次误读
  if (LEGACY_STATE !== STATE) await rm(LEGACY_STATE, { force: true })
}

async function alive(port) {
  if (!port) return false
  try {
    const r = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(800) })
    if (!r.ok) return false
    // 端口可能被别的服务占用，认身份不认端口
    return (await r.json())?.tool === 'human-canvas'
  } catch { return false }
}

/**
 * 找当前该用的服务。状态文件是单槽的——另一个进程（比如冒烟测试）开过画布
 * 就会把它覆盖/删掉，所以状态文件失效时必须能按端口扫回来，否则人提交了 agent 读不到。
 */
async function findServer(task) {
  const st = await readState()
  if (st && await alive(st.port)) return st
  for (let p = DEFAULT_PORT; p < DEFAULT_PORT + 20; p++) {
    if (!(await alive(p))) continue
    try {
      const t = await (await fetch(`http://127.0.0.1:${p}/task`, { signal: AbortSignal.timeout(1500) })).json()
      const h = await (await fetch(`http://127.0.0.1:${p}/health`, { signal: AbortSignal.timeout(1500) })).json()
      if (!task || t.task === task) return { port: p, task: t.task, image: t.image, outDir: null, pid: h.pid }
    } catch { /* 下一个端口 */ }
  }
  return null
}

/** 开一轮人在画布上的输入。返回 {url, port, task, outDir}，不阻塞等人。 */
export async function openHumanCanvas({ image, outDir, task, allow = [], port = DEFAULT_PORT, seed = null, open = true, fresh = false } = {}) {
  if (!image) throw new Error('openHumanCanvas: image is required')
  const t = task || 't-' + Date.now().toString(36)
  const out = resolve(outDir || join(HERE, 'out', t))

  const prev = fresh ? null : await readState()
  if (prev && await alive(prev.port)) {
    // 复用已在跑的服务，只换这一轮的图/task/输出目录
    const r = await fetch(`http://127.0.0.1:${prev.port}/config`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: resolve(image), outDir: out, task: t, allow, seed }),
    })
    if (!r.ok) throw new Error('reconfigure failed: ' + r.status)
    const url = `http://127.0.0.1:${prev.port}/`
    await rememberTask({ task: t, outDir: out, image: resolve(image), port: prev.port, started: Date.now() })
    if (open) launch(url)
    return { url, port: prev.port, task: t, outDir: out, reused: true }
  }

  await mkdir(out, { recursive: true })
  const seedFile = seed ? join(out, 'seed.json') : null
  if (seedFile) await writeFile(seedFile, JSON.stringify(seed), 'utf8')

  const argv = [SERVER, '--image', resolve(image), '--out', out, '--task', t, '--port', String(port)]
  if (allow.length) argv.push('--allow', allow.map((d) => resolve(d)).join(','))
  if (seedFile) argv.push('--seed', seedFile)

  // 子进程日志落盘：stdio:'ignore' 时服务起不来只剩一句 "did not come up"，
  // 看不到任何栈——那是没法排查的。
  const logPath = join(out, 'server.log')
  const logFd = openSync(logPath, 'a')
  let child
  try {
    child = spawn(process.execPath, argv, { detached: true, stdio: ['ignore', logFd, logFd], windowsHide: true })
  } finally {
    closeSync(logFd)   // 子进程已持有自己的副本，父进程不必继续握着
  }
  child.unref()

  let actual = 0
  for (let i = 0; i < 50; i++) {
    await new Promise((r) => setTimeout(r, 120))
    for (let p = port; p < port + 20; p++) {
      if (await alive(p)) { actual = p; break }
    }
    if (actual) break
  }
  if (!actual) {
    let tail = ''
    try { tail = (await readFile(logPath, 'utf8')).slice(-800).trim() } catch { /* 没有就算了 */ }
    throw new Error(`human-canvas server did not come up (log: ${logPath})${tail ? '\n' + tail : ''}`)
  }

  const url = `http://127.0.0.1:${actual}/`
  await writeState({ port: actual, task: t, image: resolve(image), outDir: out, pid: child.pid, started: Date.now() })
  await rememberTask({ task: t, outDir: out, image: resolve(image), port: actual, started: Date.now() })
  if (open) launch(url)
  return { url, port: actual, task: t, outDir: out, reused: false }
}

/** 直接从某个 outDir 读一轮结果（不依赖服务还活着、也没被下一个任务改掉配置）。 */
async function readResultFromDisk(task, outDir) {
  try {
    const token = (await readFile(join(outDir, 'done.txt'), 'utf8')).trim()
    if (token !== task) return null
    const annotation = JSON.parse(await readFile(join(outDir, 'annotation.json'), 'utf8'))
    return { done: true, task: token, outDir, annotation }
  } catch {
    return null
  }
}

/** 读回人的输入。done=false 表示还没提交（人可能还在画）。 */
export async function canvasStatus(task) {
  // 先按登记表读盘：这样即使服务已被下一个任务 reconfigure、甚至已经关掉，
  // 上一轮已提交的结果依然拿得到（单槽服务不再吞结果）。
  if (task) {
    const known = (await readTasks()).find((t) => t.task === task)
    if (known?.outDir) {
      const hit = await readResultFromDisk(task, known.outDir)
      if (hit) return { ...hit, url: known.port ? `http://127.0.0.1:${known.port}/` : null }
    }
  }
  const st = await findServer(task)
  if (!st) return { done: false, error: 'canvas server not running' }
  const q = encodeURIComponent(task ?? st.task ?? '')
  const r = await fetch(`http://127.0.0.1:${st.port}/status?task=${q}`, { signal: AbortSignal.timeout(3000) })
  return { ...(await r.json()), url: `http://127.0.0.1:${st.port}/` }
}

/** 清掉上一轮产物（新一轮开始前调用，避免旧结果被当成本轮）。 */
export async function resetHumanCanvas() {
  const st = await findServer()
  if (!st) return { ok: false, error: 'canvas server not running' }
  const r = await fetch(`http://127.0.0.1:${st.port}/reset`, { method: 'POST', signal: AbortSignal.timeout(3000) })
  return r.json()
}

/** 关掉服务（人已经不需要画了）。状态文件丢了也能靠 /health 自报的 pid 关掉。 */
export async function closeHumanCanvas() {
  const st = await findServer()
  if (st?.pid) { try { process.kill(st.pid) } catch { /* 已退出 */ } }
  await rm(STATE, { force: true })
  return { ok: true, killed: st?.pid ?? null }
}

export async function canvasUrl() {
  const st = await findServer()
  return st ? `http://127.0.0.1:${st.port}/` : null
}

function launch(url) {
  try {
    const c = spawn('cmd', ['/c', 'start', '', url], { detached: true, stdio: 'ignore', windowsHide: true })
    c.unref()
  } catch { /* 打不开就让人自己点 URL */ }
}
