#!/usr/bin/env node
// Reusable ComfyUI generate -> poll -> save.
// Usage: node comfy_generate.mjs <workflow.json> <out.png> [--seed N]
//   <workflow.json> is either a bare node graph {id:{class_type,inputs}} or {prompt:{...}}
//   env COMFY_BASE overrides the server URL (default http://127.0.0.1:7187)
import fs from 'fs';

const [,, wfPath, outPath] = process.argv;
const base = process.env.COMFY_BASE || 'http://127.0.0.1:7187';
if (!wfPath || !outPath) { console.error('usage: node comfy_generate.mjs <workflow.json> <out.png> [--seed N]'); process.exit(1); }
const seedIdx = process.argv.indexOf('--seed');
const fixedSeed = seedIdx >= 0 ? Number(process.argv[seedIdx + 1]) : null;

const raw = JSON.parse(fs.readFileSync(wfPath, 'utf8'));
const prompt = raw.prompt || raw;

if (fixedSeed != null) {
  for (const n of Object.values(prompt)) {
    if (n.class_type === 'KSampler' && n.inputs && 'seed' in n.inputs) n.inputs.seed = fixedSeed;
  }
}

async function j(method, path, body) {
  const r = await fetch(base + path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(30000),
  });
  if (!r.ok) throw new Error(path + ' ' + r.status + ': ' + (await r.text()).slice(0, 400));
  return r.json();
}

const sub = await j('POST', '/prompt', { prompt, client_id: 'dsh-comfy' });
if (!sub.prompt_id) { console.error('no prompt_id ' + JSON.stringify(sub)); process.exit(1); }
if (sub.node_errors && Object.keys(sub.node_errors).length) { console.error('node_errors ' + JSON.stringify(sub.node_errors)); process.exit(2); }
const pid = sub.prompt_id;
console.log('submitted ' + pid);

let done = false, hist;
for (let i = 0; i < 150; i++) {
  hist = await j('GET', '/history/' + pid);
  const rec = hist[pid];
  // 报错要先判：否则"跑到一半失败"会被下面的完成判定误当成"跑完了但没图"
  if (rec && rec.status && rec.status.status_str === 'error') { console.error('gen error ' + JSON.stringify(rec.status)); process.exit(3); }
  // 完成 = **有图**。有些节点只有纯 UI 输出（如 OpenposePreprocessor 的 openpose_json），
  // 按"outputs 非空"判完成会提前跳出，然后死在 no image in outputs（2026-10-15 踩到）。
  const outs = (rec && rec.outputs) || {};
  if (Object.values(outs).some((o) => o.images && o.images.length)) { done = true; break; }
  await new Promise((r) => setTimeout(r, 2000));
}
if (!done) { console.error('not done in time'); process.exit(4); }

let img = null;
for (const o of Object.values(hist[pid].outputs)) {
  if (o.images && o.images.length) { img = o.images[0]; break; }
}
if (!img) { console.error('no image in outputs ' + JSON.stringify(hist[pid].outputs).slice(0, 300)); process.exit(5); }
const q = `filename=${encodeURIComponent(img.filename)}&subfolder=${encodeURIComponent(img.subfolder || '')}&type=${encodeURIComponent(img.type || 'output')}`;
const vr = await fetch(base + '/view?' + q, { signal: AbortSignal.timeout(30000) });
const buf = Buffer.from(await vr.arrayBuffer());
fs.writeFileSync(outPath, buf);
console.log('saved ' + outPath + ' ' + buf.length + ' ' + img.filename);
