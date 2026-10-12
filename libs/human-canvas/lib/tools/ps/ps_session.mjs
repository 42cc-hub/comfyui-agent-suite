// dsh-ps / lib/ps_session.mjs
// Node 侧的常驻 PS 会话客户端：spawn ps_bridge.ps1（持有 Photoshop COM），
// 通过 NDJSON 收发 ExtendScript，一次启动、多次操作。
//
// 三条不变量，都是为了「不让 agent 永久挂住」：
//   1) 每条命令都有超时——PS 的模态对话框会让 bridge 永久阻塞，这是 PS 自动化最常见的卡点
//   2) ready 等待有超时，且桥退出/spawn 失败时 reject，不会永远 pending
//   3) 桥死了要 reject 掉所有在飞的命令，而不是让它们悬着
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const __dirname = dirname(fileURLToPath(import.meta.url))
// 可覆盖：测试要能塞一个「装着不动」的假桥来验证超时路径，而不是真去卡 Photoshop。
const BRIDGE_PS1 = process.env.DSH_PS_BRIDGE || join(__dirname, 'ps_bridge.ps1')

const READY_TIMEOUT_MS = Number(process.env.DSH_PS_READY_TIMEOUT_MS || 30000)
const CMD_TIMEOUT_MS = Number(process.env.DSH_PS_CMD_TIMEOUT_MS || 60000)

let proc = null
let rl = null
let pending = new Map()
let seq = 0
let readyPromise = null
let readyResolve = null
let readyReject = null
let readyTimer = null

function newReadyPromise() {
  readyPromise = new Promise((res, rej) => { readyResolve = res; readyReject = rej })
  // 没人 await 时也不要变成 unhandled rejection；真正的错误仍由 await 抛出。
  readyPromise.catch(() => {})
  return readyPromise
}

function settleReady(error) {
  if (readyTimer) { clearTimeout(readyTimer); readyTimer = null }
  if (error) {
    if (readyReject) { const rej = readyReject; readyReject = null; readyResolve = null; rej(error) }
  } else if (readyResolve) {
    const res = readyResolve; readyResolve = null; readyReject = null; res()
  }
}

/** 桥进程消失（正常退出、崩溃、spawn 失败）时统一收尾：ready 与所有在飞命令都不能悬着。 */
function failSession(error) {
  settleReady(error)
  for (const [, p] of pending) p.rej(error)
  pending.clear()
  proc = null
  rl = null
}

export function psReady() {
  startPS()
  return readyPromise
}

export function startPS() {
  if (proc) return proc
  newReadyPromise()
  const child = spawn('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', BRIDGE_PS1], {
    stdio: ['pipe', 'pipe', 'pipe'],
  })
  proc = child
  rl = createInterface({ input: child.stdout })
  // 每条回调都先确认自己仍属于当前会话。stopPS() 会先把 proc 置空再等旧桥退出，
  // 旧桥的 'exit' 若晚于新桥启动到达，会把新会话的 proc / pending 一起清掉。
  const live = () => proc === child
  rl.on('line', (l) => {
    if (!live()) return
    let m
    try { m = JSON.parse(l) } catch { return }
    if (m.ready) { settleReady(null); return }
    if (m.id && pending.has(m.id)) {
      const p = pending.get(m.id); pending.delete(m.id)
      if (m.e) p.rej(new Error(m.e)); else p.res(m.r)
    }
  })
  child.stderr.on('data', (d) => process.stderr.write('[ps_bridge] ' + d))
  // spawn 失败（powershell 不在、路径不可执行）只会触发 'error'，不会触发 'exit'
  child.on('error', (e) => { if (live()) failSession(new Error('ps_bridge 启动失败: ' + e.message)) })
  child.on('exit', (code) => { if (live()) failSession(new Error(`ps_bridge exited (code ${code})`)) })
  readyTimer = setTimeout(() => {
    if (live()) failSession(new Error(`ps_bridge 未在 ${READY_TIMEOUT_MS}ms 内就绪（PS 未启动或 COM 未响应）`))
  }, READY_TIMEOUT_MS)
  readyTimer.unref?.()
  return child
}

export function stopPS() {
  if (!proc) return
  const p = proc
  proc = null
  // 主动停会话时，ready 与在飞命令都不能悬着——上面的 live() 守卫已经让旧桥的
  // 'exit' 不再负责收尾，所以在这里收。
  settleReady(new Error('ps_bridge 已停止'))
  for (const [, cmdState] of pending) cmdState.rej(new Error('ps_bridge 已停止'))
  pending.clear()
  try { p.stdin.write(JSON.stringify({ id: ++seq, type: 'quit' }) + '\n') } catch { /* 已经死了 */ }
  setTimeout(() => { try { p.kill() } catch { /* 已退出 */ } }, 300).unref()
}

/**
 * 发送一段 ExtendScript，返回 PS 的返回字符串或抛错。
 * @param script ExtendScript 源码。
 * @param opts.timeoutMs 覆盖默认超时。
 */
export async function cmd(script, { timeoutMs = CMD_TIMEOUT_MS } = {}) {
  await psReady()   // 未启动就先启动；ready 失败会在这里抛出，而不是后面 proc.stdin 上的 TypeError
  return new Promise((res, rej) => {
    const id = ++seq
    let timer = null
    const settle = (fn, value) => { if (timer) clearTimeout(timer); fn(value) }
    timer = setTimeout(() => {
      pending.delete(id)
      rej(new Error(
        `ps_bridge 命令超时（${timeoutMs}ms, id=${id}）。Photoshop 通常是被模态对话框挡住了` +
        '（缺字体 / 色彩配置 / 文件已被占用）——去 PS 里关掉它再重试，或调大 DSH_PS_CMD_TIMEOUT_MS。',
      ))
    }, timeoutMs)
    timer.unref?.()
    pending.set(id, {
      res: (v) => settle(res, v),
      rej: (e) => settle(rej, e),
    })
    try { proc.stdin.write(JSON.stringify({ id, script }) + '\n') }
    catch (e) { if (timer) clearTimeout(timer); pending.delete(id); rej(e) }
  })
}

// 打开一个文档（com=aPath），返回文档名称/尺寸。
export async function openImage(absPath) {
  const js = 'var f = new File(' + JSON.stringify(absPath) + '); var d = app.open(f); (Number(d.width))+"x"+(Number(d.height));'
  const r = await cmd(js)
  return r
}
