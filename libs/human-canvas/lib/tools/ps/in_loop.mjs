// dsh-ps / lib/in_loop.mjs — 「人在回路」核心（非阻塞模型）
// 核心约定：agent 绝不前台阻塞等人。
//   正确流程：通知 → ask(闸门) → openInPS(打开+置顶) → 【让出回合】→ 人去编辑并原图保存 →
//   人回复 → agent 恢复 → checkChanged(mtime/哈希 验证) + 读原路径 → 继续。
import { stat, readFile, readdir } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { spawn, execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const NOTIFY = await import('../../notify.mjs')

export const IN_LOOP_BUDGET = 5
let requests = 0
export function canRequest() { return requests < IN_LOOP_BUDGET }
export function countRequest() { requests++ }

export function notifyHuman({ title = 'DSH 需要你', message = '', duration = 10000 } = {}) {
  NOTIFY.windowsBalloon({ title, message, duration })
  return true
}

/**
 * 找本机的 Photoshop.exe。
 * 原来是硬编码绝对路径，PS 换版本/换盘就静默失效（spawn 失败发生在 detached 子进程里，
 * 调用方什么也看不到）。这里按 DSH_PS_EXE → 注册表 → 常见安装根目录 依次探测，失败时把
 * 试过的路径全部报出来。
 */
let cachedExe = null

async function readdirSafe(dir, opts) {
  try { return await readdir(dir, opts) } catch { return [] }
}

/** 从注册表读 Adobe 记录的 ApplicationPath（32/64 位两处都看）。 */
function registryCandidates() {
  const out = []
  const keys = [
    'HKLM\\SOFTWARE\\Adobe\\Photoshop',
    'HKLM\\SOFTWARE\\WOW6432Node\\Adobe\\Photoshop',
  ]
  for (const key of keys) {
    let text = ''
    try {
      text = execFileSync('reg.exe', ['query', key, '/s', '/v', 'ApplicationPath'],
        { encoding: 'utf8', timeout: 5000, windowsHide: true })
    } catch { continue }
    for (const line of text.split(/\r?\n/)) {
      const m = line.match(/ApplicationPath\s+REG_SZ\s+(.+?)\s*$/i)
      if (m) out.push(join(m[1].trim(), 'Photoshop.exe'))
    }
  }
  return out
}

async function collectExe(dir, depth, out) {
  if (depth < 0) return
  for (const e of await readdirSafe(dir, { withFileTypes: true })) {
    const full = join(dir, e.name)
    if (e.isFile() && /^photoshop.*\.exe$/i.test(e.name)) out.push(full)
    else if (e.isDirectory()) await collectExe(full, depth - 1, out)
  }
}

/** 扫盘根下名字含 adobe 的目录，外加两个标准 Program Files\Adobe。 */
async function scanCandidates() {
  const out = []
  for (const d of ['C:', 'D:', 'E:', 'F:', 'G:', 'H:']) {
    const base = d + '\\'
    if (!existsSync(base)) continue
    for (const name of await readdirSafe(base)) {
      if (/adobe/i.test(name)) await collectExe(join(base, name), 4, out)
    }
    for (const pf of ['Program Files', 'Program Files (x86)']) {
      const adobe = join(base, pf, 'Adobe')
      if (existsSync(adobe)) await collectExe(adobe, 2, out)
    }
  }
  return out
}

export async function resolvePSExe() {
  if (cachedExe && existsSync(cachedExe)) return cachedExe
  const tried = []
  const consider = (p) => {
    if (!p) return false
    if (existsSync(p)) { cachedExe = p; return true }
    tried.push(p)
    return false
  }
  if (consider(process.env.DSH_PS_EXE)) return cachedExe
  for (const p of registryCandidates()) if (consider(p)) return cachedExe
  for (const p of await scanCandidates()) if (consider(p)) return cachedExe
  throw new Error(
    '找不到 Photoshop.exe。试过：\n' + (tried.length ? tried.join('\n') : '(无候选)') +
    '\n可用环境变量 DSH_PS_EXE 指定完整路径。',
  )
}

export async function getMtime(path) {
  try { return (await stat(path)).mtimeMs } catch { return 0 }
}

/**
 * 文件指纹。只看 mtime 会两头误判：打开又原样保存也会改 mtime（假阳性），
 * 「另存到别处再拷回来」可能让 mtime 不变（假阴性）。哈希能给出确定答案。
 */
export async function getFingerprint(path) {
  try {
    const st = await stat(path)
    if (!st.isFile()) return { exists: false, mtime: 0, size: 0, sha256: null }
    const sha256 = createHash('sha256').update(await readFile(path)).digest('hex')
    return { exists: true, mtime: st.mtimeMs, size: st.size, sha256 }
  } catch {
    return { exists: false, mtime: 0, size: 0, sha256: null }
  }
}

// 打开原图让人编辑：写一个 .jsx(app.open+置顶)，用 `Photoshop.exe -r <jsx>` 启动时执行，
// PS 打开该图并保持，node 立即退出、无挂起进程。
export async function openInPS(imagePath) {
  const exe = await resolvePSExe()
  const jsx = 'app.open(new File(' + JSON.stringify(imagePath) + ')); app.bringToFront();'
  const jsxPath = join(tmpdir(), 'dsh_ps_open_' + Date.now() + '.jsx')
  writeFileSync(jsxPath, jsx, 'utf8')
  const child = spawn(exe, ['-r', jsxPath], { detached: true, stdio: 'ignore' })
  child.unref()
  return 'open:' + imagePath
}

/**
 * 一次性验证（非阻塞）：人保存后内容是否已变。
 * 传了 baseSha256 就以哈希为准（确定），否则退回 mtime 比较（含 150ms 容差）。
 */
export async function checkChanged(path, baseMtime, baseSha256 = null) {
  const fp = await getFingerprint(path)
  if (!fp.exists) return { changed: false, byMtime: false, byHash: null, ...fp }
  const byMtime = fp.mtime > baseMtime + 150
  const byHash = baseSha256 ? fp.sha256 !== baseSha256 : null
  return { changed: byHash === null ? byMtime : byHash, byMtime, byHash, ...fp }
}

// 结果就在原路径
export function readResult(path) { return path }
