// @dsh-external/dsh-man-in-loop — host half.
//
// 人在回路工具集：把「agent 需要人帮忙」做成可调用的工具。三个工具、两条通道：
//   man_in_loop_notify  把人叫回来（Windows 系统气泡）
//   human_canvas        人在浏览器画布上给【粗略二维输入】(涂 mask / 摆点连线，带 label)
//   ps_edit             人在 Photoshop 里做【真正的图像编辑】（改原文件）
//
// 非阻塞铁律：open 类动作立刻返回，然后【让出回合】；人回复后再用 status/check 读回。
// 前台阻塞会卡死 agent、看不到用户消息——这是本工具集存在的全部理由。
import { isAbsolute, resolve as resolvePath } from 'node:path'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { windowsBalloonAndWait } from './notify.mjs'
import { checkChanged, getFingerprint, openInPS } from './tools/ps/in_loop.mjs'
import {
  canvasStatus, closeHumanCanvas, openHumanCanvas, resetHumanCanvas,
} from './tools/human-canvas/index.mjs'

// 库导出照旧可用：脚本里 import { openHumanCanvas } 不受影响
export * from './index.js'

export const name = '@dsh-external/dsh-man-in-loop'
export const inject = ['tools']

const text = (value) => [{ type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value, null, 1) }]
const out = { schema: { type: 'string' }, render: (_args, value) => text(value) }

function sessionCwd(exec) {
  return exec?.agent?.session?.header?.cwd ?? process.cwd()
}
function absPath(p, exec) {
  const s = String(p ?? '').trim()
  if (!s) throw new Error('image is required')
  return isAbsolute(s) ? s : resolvePath(sessionCwd(exec), s)
}
// seed_json 是手写的 JSON 字符串，打错一个字符不该只丢一个裸 SyntaxError 出去。
function parseSeed(raw) {
  if (!raw) return null
  try {
    return JSON.parse(raw)
  } catch (e) {
    throw new Error(
      `seed_json 不是合法 JSON（${e.message}）。期望形如 ` +
      '{"layers":[{"name":"前景","label":"主体","color":"#ff5050"}]} 或 {"nodes":[...],"edges":[...]}',
    )
  }
}

export function apply(ctx) {
  ctx.effect(() => ctx.tools.register(defineTool({
    name: 'man_in_loop_notify',
    description: 'Ping the human out-of-band (Windows tray balloon) when their input is needed. Waits for the balloon to finish and reports whether it actually fired.',
    parameters: {
      message: { type: 'string', required: true, description: 'What you need from them, one line.' },
      title: { type: 'string', description: 'Balloon title.' },
      duration: { type: 'integer', description: 'Balloon display ms, default 10000.' },
    },
    output: out,
    async execute(args) {
      const title = args.title || 'DSH 需要你'
      const r = await windowsBalloonAndWait({ title, message: args.message, duration: args.duration ?? 10000 })
      if (!r.ok) {
        return `通知可能没送达（pid=${r.pid} code=${r.code} ${r.error ?? ''}）。` +
          '不要假设人已经看到——改用别的途径叫人，或稍后重发。'
      }
      return `notified (balloon): ${args.message}`
    },
  })), 'man-in-loop: notify')

  ctx.effect(() => ctx.tools.register(defineTool({
    name: 'human_canvas',
    description: 'Browser canvas for ROUGH human input: paint labelled mask layers, or place labelled graph points/edges. Non-blocking.',
    parameters: {
      action: { type: 'string', required: true, description: 'open | status | reset | close' },
      image: { type: 'string', description: 'open: image path (absolute or workspace-relative).' },
      out_dir: { type: 'string', description: 'open: annotation output dir.' },
      task: { type: 'string', description: 'open/status: task token; auto-generated on open.' },
      seed_json: { type: 'string', description: 'open: JSON for preset layers/vector graph, e.g. {"layers":[{"name":"前景","label":"主体","color":"#ff5050"}]} or {"nodes":[{"id":"n1","x":10,"y":20,"label":"left_ankle"}],"edges":[]}.' },
      allow: { type: 'array', items: { type: 'string' }, description: 'open: extra dirs the canvas may serve images from.' },
      open_browser: { type: 'boolean', description: 'open: launch default browser, default true.' },
    },
    output: out,
    async execute(args, exec) {
      const action = String(args.action || '').toLowerCase()
      if (action === 'open') {
        const image = absPath(args.image, exec)
        const outDir = args.out_dir
          ? absPath(args.out_dir, exec)
          : resolvePath(sessionCwd(exec), 'agent_work/human-canvas', String(args.task || `t-${Date.now().toString(36)}`))
        const r = await openHumanCanvas({
          image, outDir, task: args.task,
          seed: parseSeed(args.seed_json),
          allow: (args.allow ?? []).map((d) => absPath(d, exec)),
          open: args.open_browser !== false,
        })
        return [
          `canvas open: ${r.url}`,
          `task: ${r.task}`,
          `out: ${r.outDir}`,
          'Now yield the turn: tell the human what to paint/place, and read back later with action=status.',
        ].join('\n')
      }
      if (action === 'status') return JSON.stringify(await canvasStatus(args.task), null, 1)
      if (action === 'reset') return JSON.stringify(await resetHumanCanvas(), null, 1)
      if (action === 'close') return JSON.stringify(await closeHumanCanvas(), null, 1)
      throw new Error(`unknown action: ${args.action} (open|status|reset|close)`)
    },
  })), 'man-in-loop: human-canvas')

  ctx.effect(() => ctx.tools.register(defineTool({
    name: 'ps_edit',
    description: 'Hand an image to the human in Photoshop for real editing (non-blocking), then verify they saved it. The human may not be watching the screen — pair this with man_in_loop_notify.',
    parameters: {
      action: { type: 'string', required: true, description: 'open | check' },
      image: { type: 'string', required: true, description: 'Image path (absolute or workspace-relative).' },
      since_mtime: { type: 'number', description: 'check: mtime returned by open.' },
      since_sha256: { type: 'string', description: 'check: sha256 returned by open. When given, changed is decided by content, not by the clock.' },
    },
    output: out,
    async execute(args, exec) {
      const action = String(args.action || '').toLowerCase()
      const image = absPath(args.image, exec)
      if (action === 'open') {
        const fp = await getFingerprint(image)
        if (!fp.exists) throw new Error(`image not found: ${image}`)
        await openInPS(image)
        return [
          `opened in Photoshop: ${image}`,
          `mtime: ${fp.mtime}`,
          `sha256: ${fp.sha256}`,
          `size: ${fp.size}`,
          'Now yield the turn: tell the human to edit and SAVE THE SAME FILE, then reply.',
          'Read back later with action=check, passing since_mtime and since_sha256.',
        ].join('\n')
      }
      if (action === 'check') {
        // 只看 mtime 两头都会错：打开又原样保存也改 mtime（假阳性），另存再拷回来可能不改（假阴性）。
        // 给了 sha256 就以内容为准。
        const r = args.since_mtime == null && args.since_sha256 == null
          ? { changed: null, ...(await getFingerprint(image)) }
          : await checkChanged(image, args.since_mtime ?? 0, args.since_sha256 ?? null)
        return JSON.stringify({ path: image, ...r }, null, 1)
      }
      throw new Error(`unknown action: ${args.action} (open|check)`)
    },
  })), 'man-in-loop: ps')

  // 选型引导：让 agent 知道两条通道的分工，别拿重锤砸钉子
  ctx.get('systemPrompt')?.section?.({
    name: 'tool:man-in-loop',
    order: 150,
    text: [
      'Human-in-the-loop has two channels; pick the lighter one.',
      'human_canvas = the human supplies ROUGH 2D input (labelled mask layers, labelled points/edges).',
      'Use it for matting, inpaint regions, and fixing/extending detected keypoints: the human can be sloppy,',
      'you refine (edge-aware refinement, GrabCut-style trimaps, fitting) and they can also be surgical when it matters.',
      'ps_edit = the human does REAL image editing in Photoshop (retouch, layers/channels) on the original file;',
      'requires Photoshop on this host. Prefer human_canvas unless the task genuinely needs PS.',
      'Both are non-blocking: open, then YIELD THE TURN and state exactly what the human must do;',
      'when they reply, read back via human_canvas action=status or ps_edit action=check. Never block waiting on them.',
    ].join(' '),
  })
}
