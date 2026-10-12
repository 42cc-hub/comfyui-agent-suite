// human-canvas / lib/notify.mjs
// Windows 系统托盘气泡通知原语（无需注册，最稳）。
// 这里曾有一个写 dsh-cc-connect outbox 的 qqNotify()；2026-09-12 随该插件卸载一并删除，
// QQ 侧改由 dsh-im 的主动投递承担（设置 → IM机器人，或 delivery RPC 的 message.send）。
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const __dirname = dirname(fileURLToPath(import.meta.url))
const BALLOON_PS1 = join(__dirname, '..', 'notify', 'win-balloon.ps1')

/** 气泡最长等多久（默认 duration+2s，上限 20s）。等不到就当作失败，别把 agent 拖住。 */
const WAIT_SLACK_MS = 2000
const WAIT_CAP_MS = 20000

/**
 * 发一条 Windows 系统通知（托盘气泡）。
 * 关键：不要 detached/unref——父进程需保持存活到气泡显示完，否则会被父进程退出带掉。
 *
 * 返回 { pid, done }：done 在子进程结束时给出退出码。调用方若关心"到底发出去没有"，
 * 就 await 它（见 windowsBalloonAndWait）；不关心就忽略。
 * @param {object} opts { title, message, duration(ms) }
 */
export function windowsBalloon({ title = 'DSH Agent', message = '', duration = 10000 } = {}) {
  const child = spawn('powershell.exe', [
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', BALLOON_PS1,
    '-Title', title,
    '-Message', message,
    '-Duration', String(duration),
  ], { stdio: 'ignore' })
  const done = new Promise((res) => {
    child.on('error', (e) => res({ code: null, error: e.message }))
    child.on('exit', (code) => res({ code, error: null }))
  })
  return { pid: child.pid, done }
}

/**
 * 发通知并等到有结果。整个非阻塞流程都押在这次 ping 上，所以工具层必须能知道它成没成：
 * 之前无条件回 "notified"，万一 PowerShell 起不来，agent 会以为自己叫过人，
 * 然后一直等一个永远不会来的人。
 * @returns {Promise<{ok: boolean, pid: number|undefined, code: number|null, error: string|null}>}
 */
export async function windowsBalloonAndWait(opts = {}) {
  const duration = Number(opts.duration ?? 10000)
  const budget = Math.min(duration + WAIT_SLACK_MS, WAIT_CAP_MS)
  const { pid, done } = windowsBalloon(opts)
  let timer = null
  const timedOut = new Promise((res) => {
    timer = setTimeout(() => res({ code: null, error: `等待气泡结果超时（${budget}ms）` }), budget)
    timer.unref?.()
  })
  const result = await Promise.race([done, timedOut])
  if (timer) clearTimeout(timer)
  return { ok: result.error === null && result.code === 0, pid, code: result.code, error: result.error }
}
