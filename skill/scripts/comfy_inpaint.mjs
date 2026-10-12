// 局部重绘驱动：LoadImage(base) + LoadImage(mask) → VAEEncode → SetLatentNoiseMask → KSampler(denoise)
//
// 与整图生图的区别：只有 mask 内的 latent 被加噪，mask 外的像素原样保留。
// 因此提示词可以（也应该）把 mask 外的人物简化成 "1girl at right side" 之类，
// 避免模型把没在重绘的角色特征也改写一遍（污染）。
//
// usage: node run_inpaint.mjs <jobs.json>
import fs from 'fs';
import path from 'path';

const base = process.env.COMFY_BASE || 'http://127.0.0.1:7187';
const jobsPath = path.resolve(process.argv[2] || 'inpaint_jobs.json');
const spec = JSON.parse(fs.readFileSync(jobsPath, 'utf8'));
const jobsDir = path.dirname(jobsPath);
const outDir = path.resolve(jobsDir, spec.outDir || 'inpaint_out');
fs.mkdirSync(outDir, { recursive: true });

async function api(method, p, body, timeout = 120000) {
  const r = await fetch(base + p, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(timeout),
  });
  if (!r.ok) throw new Error(p + ' ' + r.status + ': ' + (await r.text()).slice(0, 300));
  return r.json();
}

async function upload(localPath) {
  const buf = fs.readFileSync(localPath);
  const fd = new FormData();
  fd.append('image', new Blob([buf], { type: 'image/png' }), path.basename(localPath));
  fd.append('subfolder', 'dsh_ctrl');
  fd.append('overwrite', 'true');
  const r = await fetch(base + '/upload/image', { method: 'POST', body: fd, signal: AbortSignal.timeout(60000) });
  if (!r.ok) throw new Error('upload ' + r.status);
  const j = await r.json();
  return j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
}

function buildGraph(job) {
  const g = {
    1: { class_type: 'LoadImage', inputs: { image: job.baseRemote } },
    2: { class_type: 'LoadImage', inputs: { image: job.maskRemote } },
    3: { class_type: 'ImageToMask', inputs: { image: ['2', 0], channel: 'red' } },
    4: { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: spec.base } },
    5: { class_type: 'VAEEncode', inputs: { pixels: ['1', 0], vae: ['4', 2] } },
    6: { class_type: 'SetLatentNoiseMask', inputs: { samples: ['5', 0], mask: ['3', 0] } },
    7: { class_type: 'CLIPTextEncode', inputs: { text: job.positive, clip: ['4', 1] } },
    8: { class_type: 'CLIPTextEncode', inputs: { text: job.negative, clip: ['4', 1] } },
  };
  let pos = ['7', 0], neg = ['8', 0];
  (job.controls ?? []).forEach((c, i) => {
    const cn = `cn${i}`, im = `img${i}`, ap = `ap${i}`;
    g[cn] = { class_type: 'ControlNetLoader', inputs: { control_net_name: c.model } };
    g[im] = { class_type: 'LoadImage', inputs: { image: c.remote } };
    g[ap] = {
      class_type: 'ControlNetApplyAdvanced',
      inputs: {
        positive: pos, negative: neg, control_net: [cn, 0], image: [im, 0],
        strength: c.strength ?? 0.7, start_percent: c.start ?? 0.0, end_percent: c.end ?? 1.0,
      },
    };
    pos = [ap, 0]; neg = [ap, 1];
  });
  g['10'] = {
    class_type: 'KSampler',
    inputs: {
      seed: job.seed, steps: job.steps ?? 28, cfg: job.cfg ?? 6,
      sampler_name: job.sampler ?? 'dpmpp_2m', scheduler: job.scheduler ?? 'karras',
      denoise: job.denoise ?? 0.7, model: ['4', 0], positive: pos, negative: neg, latent_image: ['6', 0],
    },
  };
  g['11'] = { class_type: 'VAEDecode', inputs: { samples: ['10', 0], vae: ['4', 2] } };
  g['12'] = { class_type: 'PreviewImage', inputs: { images: ['11', 0] } };
  return g;
}

async function runOne(job) {
  job.baseRemote = await upload(path.resolve(jobsDir, job.base));
  job.maskRemote = await upload(path.resolve(jobsDir, job.mask));
  for (const c of job.controls ?? []) c.remote = await upload(path.resolve(jobsDir, c.image));

  const sub = await api('POST', '/prompt', { prompt: buildGraph(job), client_id: 'dsh-inpaint' });
  if (!sub.prompt_id) throw new Error('no prompt_id ' + JSON.stringify(sub).slice(0, 300));
  if (sub.node_errors && Object.keys(sub.node_errors).length) {
    throw new Error('node_errors ' + JSON.stringify(sub.node_errors).slice(0, 400));
  }
  const pid = sub.prompt_id;
  for (let i = 0; i < 600; i++) {
    const hist = await api('GET', '/history/' + pid);
    const rec = hist[pid];
    if (rec && rec.status && rec.status.status_str === 'error') {
      throw new Error('exec error ' + JSON.stringify(rec.status).slice(0, 600));
    }
    if (rec && rec.outputs && Object.keys(rec.outputs).length) {
      let img = null;
      for (const o of Object.values(rec.outputs)) if (o.images && o.images.length) { img = o.images[0]; break; }
      if (!img) throw new Error('no image in outputs');
      const q = `filename=${encodeURIComponent(img.filename)}&subfolder=${encodeURIComponent(img.subfolder || '')}&type=${encodeURIComponent(img.type || 'temp')}`;
      const vr = await fetch(base + '/view?' + q, { signal: AbortSignal.timeout(60000) });
      const buf = Buffer.from(await vr.arrayBuffer());
      const out = path.join(outDir, `${job.id}.png`);
      fs.writeFileSync(out, buf);
      return { id: job.id, seed: job.seed, denoise: job.denoise, file: out, bytes: buf.length };
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  throw new Error('timeout');
}

const results = [];
for (const job of spec.jobs) {
  const t0 = Date.now();
  try {
    const r = await runOne(job);
    r.seconds = Math.round((Date.now() - t0) / 1000);
    results.push(r);
    console.log(`OK  ${r.id}  denoise=${r.denoise}  ${r.seconds}s  ${r.file}`);
  } catch (e) {
    results.push({ id: job.id, error: String(e.message || e) });
    console.log(`ERR ${job.id}  ${String(e.message || e).slice(0, 300)}`);
  }
}
fs.writeFileSync(path.join(jobsDir, 'inpaint_results.json'), JSON.stringify(results, null, 2));
console.log(`done: ${results.filter((r) => !r.error).length}/${results.length}`);
