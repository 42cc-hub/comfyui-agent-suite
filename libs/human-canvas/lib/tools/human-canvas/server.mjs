// human-canvas / lib/tools/human-canvas/server.mjs
// 人机输入画布服务：供图 + 收「人在画布上的输入」= 栅格图层 mask + 矢量图(点/边) + 每项 label。
//
// 与旧 mask-editor 的三处关键差异（都是旧版会咬人的地方）：
//   1) /img 只服务白名单内的文件（注册过的图 + --allow 目录），不再是任意绝对路径可读
//   2) done 标记里带 task token：上一轮的结果不会被当成本轮
//   3) 提交是单个 JSON（各图层 PNG 走 dataURI），服务端负责拆成文件落盘
//
// 设计取向：人给的是「精度可变」的输入——可以花很久描准，也可以几笔粗略圈一下。
// 因此服务端只负责忠实保存（含 alpha 覆盖率），不做任何"智能"处理；如何用是 agent 的事。
import { createServer } from 'node:http'
import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { dirname, extname, join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const EDITOR = join(HERE, 'editor.html')

const MIME = {
  '.html': 'text/html; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.gif': 'image/gif', '.bmp': 'image/bmp',
}

export const DEFAULT_PORT = 8140

// 全透明图层也会有一张合法 PNG；这个阈值只是"压缩后小到不可能是真笔迹"的启发式，
// 不是精确的空判定。要精确就得解 PNG 看 alpha，代价不值得。
const EMPTY_LAYER_PNG_BYTES = 200

let server = null
let cfg = { outDir: join(HERE, 'out'), allow: [], task: null, image: null, seed: null }

function underDir(p, dir) {
  const a = resolve(p)
  const b = resolve(dir)
  return a === b || a.startsWith(b.endsWith(sep) ? b : b + sep)
}

function allowed(p) {
  if (!p) return false
  if (cfg.image && resolve(p) === resolve(cfg.image)) return true
  return cfg.allow.some((d) => underDir(p, d))
}

// 只删本工具自己写出来的文件。旧实现是 rm -rf 整个 outDir —— agent 只要把 out_dir
// 指到一个装着别的东西的目录（合法入参，没人拦），人在画布上点一次"提交"就会把它清空。
const OWNED_ARTIFACT = /^(?:layer_[^\\/]*\.png|annotation\.json|done\.txt|seed\.json)$/

async function cleanOwnedArtifacts(dir) {
  let names
  try { names = await readdir(dir) } catch { names = [] }   // 目录不存在=本来就干净
  await Promise.all(names
    .filter((n) => OWNED_ARTIFACT.test(n))
    .map((n) => rm(join(dir, n), { force: true })))
  await mkdir(dir, { recursive: true })
}

// 无 Origin 的调用（agent 自己、curl）放行；带了 Origin 就必须是 loopback。
// 这样跨站页面没法借用户的浏览器触发 /reset、/submit、/config。
function originAllowed(req) {
  const origin = req.headers.origin
  if (!origin) return true
  try {
    const { hostname } = new URL(origin)
    return hostname === '127.0.0.1' || hostname === 'localhost' || hostname === '::1'
  } catch { return false }
}

function readBody(req, limit = 64 * 1024 * 1024) {
  return new Promise((res, rej) => {
    const chunks = []
    let n = 0
    req.on('data', (c) => {
      n += c.length
      if (n > limit) { rej(new Error('body too large')); req.destroy(); return }
      chunks.push(c)
    })
    req.on('end', () => res(Buffer.concat(chunks)))
    req.on('error', rej)
  })
}

function send(res, code, body, type = 'text/plain; charset=utf-8') {
  res.writeHead(code, { 'Content-Type': type, 'Cache-Control': 'no-store' })
  res.end(body)
}

const json = (res, code, obj) => send(res, code, JSON.stringify(obj), 'application/json; charset=utf-8')

async function writeAnnotation(payload) {
  await cleanOwnedArtifacts(cfg.outDir)

  const layers = []
  for (const L of payload.raster?.layers ?? []) {
    const entry = {
      id: L.id, name: L.name ?? L.id, label: L.label ?? '',
      color: L.color ?? '#ff5050', opacity: L.opacity ?? 1, visible: L.visible !== false,
      file: null, empty: true,
    }
    if (typeof L.png === 'string' && L.png.startsWith('data:image/png;base64,')) {
      const buf = Buffer.from(L.png.slice('data:image/png;base64,'.length), 'base64')
      entry.file = `layer_${L.id}.png`
      await writeFile(join(cfg.outDir, entry.file), buf)
      entry.empty = buf.length < EMPTY_LAYER_PNG_BYTES
    }
    layers.push(entry)
  }

  const annotation = {
    tool: 'human-canvas',
    version: 1,
    task: payload.task,
    image: cfg.image,
    size: payload.size ?? null,
    submitted_at: Date.now(),
    raster: { layers },
    vector: {
      nodes: (payload.vector?.nodes ?? []).map((n) => ({
        id: n.id, x: n.x, y: n.y, label: n.label ?? '',
      })),
      edges: (payload.vector?.edges ?? []).map((e) => ({
        id: e.id, a: e.a, b: e.b, label: e.label ?? '',
      })),
    },
    notes: payload.notes ?? '',
  }
  await writeFile(join(cfg.outDir, 'annotation.json'), JSON.stringify(annotation, null, 1), 'utf8')
  // task token 一并写入：读回时校验，避免上一轮的结果被当成本轮
  await writeFile(join(cfg.outDir, 'done.txt'), String(payload.task ?? ''), 'utf8')
  return annotation
}

export function createCanvasServer() {
  return createServer(async (req, res) => {
    const u = new URL(req.url, 'http://x')
    const p = u.pathname
    try {
      if (req.method === 'POST' && !originAllowed(req)) return send(res, 403, 'cross-origin refused')
      // pid 自报：状态文件丢了也能被 closeHumanCanvas 关掉
      if (p === '/health') return json(res, 200, { ok: true, tool: 'human-canvas', pid: process.pid, task: cfg.task })
      if (p === '/') return send(res, 200, await readFile(EDITOR, 'utf8'), MIME['.html'])
      if (p === '/task') {
        return json(res, 200, { task: cfg.task, image: cfg.image ? '/img?p=' + encodeURIComponent(cfg.image) : null, seed: cfg.seed })
      }
      if (p === '/img') {
        const f = u.searchParams.get('p')
        if (!allowed(f)) return send(res, 403, 'forbidden')
        const buf = await readFile(f)
        return send(res, 200, buf, MIME[extname(f).toLowerCase()] || 'application/octet-stream')
      }
      if (p === '/status') {
        const want = u.searchParams.get('task') ?? cfg.task
        try {
          const token = (await readFile(join(cfg.outDir, 'done.txt'), 'utf8')).trim()
          if (token !== want) return json(res, 200, { done: false, stale: Boolean(token) })
          const annotation = JSON.parse(await readFile(join(cfg.outDir, 'annotation.json'), 'utf8'))
          return json(res, 200, { done: true, task: token, outDir: cfg.outDir, annotation })
        } catch {
          return json(res, 200, { done: false })
        }
      }
      if (p === '/config' && req.method === 'POST') {
        // 复用已启动的服务开始新一轮：换图/换 task/换输出目录，页面每 5s 轮询 /task 会提示人重开
        const body = JSON.parse((await readBody(req)).toString('utf8') || '{}')
        configure(body)
        await mkdir(cfg.outDir, { recursive: true })
        return json(res, 200, { ok: true, task: cfg.task, image: cfg.image, outDir: cfg.outDir })
      }
      if (p === '/reset' && req.method === 'POST') {
        await cleanOwnedArtifacts(cfg.outDir)
        return json(res, 200, { ok: true })
      }
      if (p === '/submit' && req.method === 'POST') {
        const body = JSON.parse((await readBody(req)).toString('utf8') || '{}')
        if (!cfg.task || body.task !== cfg.task) {
          return json(res, 409, { ok: false, error: 'task mismatch', expected: cfg.task, got: body.task })
        }
        const annotation = await writeAnnotation(body)
        return json(res, 200, {
          ok: true, outDir: cfg.outDir,
          layers: annotation.raster.layers.length,
          nodes: annotation.vector.nodes.length,
          edges: annotation.vector.edges.length,
        })
      }
      return send(res, 404, 'not found')
    } catch (e) {
      return send(res, 500, String(e?.stack || e))
    }
  })
}

/** 启动服务；端口被占用时顺延，返回实际端口。 */
export async function startServer({ port = DEFAULT_PORT, outDir, allow = [], task, image, seed } = {}) {
  if (server) { configure({ outDir, allow, task, image, seed }); return { port: server.address().port } }
  configure({ outDir, allow, task, image, seed })
  await mkdir(cfg.outDir, { recursive: true })

  for (let candidate = port; candidate < port + 20; candidate++) {
    const s = createCanvasServer()
    const result = await new Promise((res) => {
      // 不要在 'error' 监听器里 throw：那是未捕获异常，会把 detached 进程直接打死，
      // 而且（因为 stdio 被重定向）除非看日志否则什么也看不到。收进结果里在外面抛。
      s.once('error', (e) => res({ error: e }))
      s.listen(candidate, '127.0.0.1', () => res({ ok: true }))
    })
    if (result.error) {
      if (result.error.code === 'EADDRINUSE') continue
      throw result.error
    }
    server = s
    return { port: candidate }
  }
  throw new Error(`no free port in ${port}..${port + 19}`)
}

export function configure({ outDir, allow, task, image, seed } = {}) {
  if (outDir) cfg.outDir = resolve(outDir)
  if (allow) cfg.allow = allow.map((d) => resolve(d))
  if (task !== undefined) cfg.task = task
  if (image !== undefined) cfg.image = image ? resolve(image) : null
  if (seed !== undefined) cfg.seed = seed
}

export function serverPort() { return server?.address()?.port ?? null }
export function serverConfig() { return { ...cfg } }

export async function stopServer() {
  if (!server) return
  const s = server
  server = null
  await new Promise((res) => s.close(res))
}

// ---- CLI：以 detached 子进程跑，服务不随 agent 的回合结束而消失 -------------------
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = {}
  for (let i = 2; i < process.argv.length - 1; i += 2) args[process.argv[i].replace(/^--/, '')] = process.argv[i + 1]
  const seed = args.seed ? JSON.parse(await readFile(args.seed, 'utf8')) : null
  const { port } = await startServer({
    port: Number(args.port || DEFAULT_PORT),
    outDir: args.out,
    allow: (args.allow || '').split(',').map((s) => s.trim()).filter(Boolean),
    task: args.task,
    image: args.image,
    seed,
  })
  console.log(`human-canvas on http://127.0.0.1:${port}/  task=${args.task}`)
}
