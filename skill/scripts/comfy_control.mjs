// ControlNet / LLLite runner for Blender-authored control maps.
//
// pipeline "sdxl"  : SDXL checkpoint + ControlNetLoader + ControlNetApplyAdvanced chain
// pipeline "anima" : Anima UNET/CLIP/VAE + AnimaLLLiteApply_sdscripts chain
//
// Every path in the job file (outDir, control images, init image) resolves against
// the JOB FILE's directory, so this script can live anywhere.
// Control images are uploaded into ComfyUI's input/dsh_ctrl subfolder (delete it
// afterwards) and results are pulled via PreviewImage -> /view?type=temp, so
// nothing is written to ComfyUI's output/.
//
// usage: node comfy_control.mjs <jobs.json>
import fs from 'fs';
import path from 'path';

const base = process.env.COMFY_BASE || 'http://127.0.0.1:7187';
const jobsPath = path.resolve(process.argv[2] || 'jobs.json');
const spec = JSON.parse(fs.readFileSync(jobsPath, 'utf8'));
const jobsDir = path.dirname(jobsPath);
const outDir = path.resolve(jobsDir, spec.outDir || 'gen');
fs.mkdirSync(outDir, { recursive: true });

async function api(method, p, body, timeout = 120000) {
  const r = await fetch(base + p, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(timeout),
  });
  if (!r.ok) throw new Error(p + ' ' + r.status + ': ' + (await r.text()).slice(0, 400));
  return r.json();
}

async function upload(localPath) {
  const buf = fs.readFileSync(localPath);
  const fd = new FormData();
  fd.append('image', new Blob([buf], { type: 'image/png' }), path.basename(localPath));
  fd.append('subfolder', 'dsh_ctrl');
  fd.append('overwrite', 'true');
  const r = await fetch(base + '/upload/image', { method: 'POST', body: fd, signal: AbortSignal.timeout(60000) });
  if (!r.ok) throw new Error('upload ' + r.status + ' ' + (await r.text()).slice(0, 200));
  const j = await r.json();
  return j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
}

function buildGraph(job) {
  const g = {};
  const W = job.width ?? 832;
  const H = job.height ?? 1216;
  const sampler = {
    seed: job.seed,
    steps: job.steps ?? 28,
    cfg: job.cfg ?? 6,
    sampler_name: job.sampler ?? 'dpmpp_2m',
    scheduler: job.scheduler ?? 'karras',
    denoise: 1,
  };

  if (spec.pipeline === 'sdxl') {
    g['1'] = { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: spec.base } };
    g['2'] = { class_type: 'CLIPTextEncode', inputs: { text: job.positive, clip: ['1', 1] } };
    g['3'] = { class_type: 'CLIPTextEncode', inputs: { text: job.negative, clip: ['1', 1] } };
    if (job.init) {
      // img2img: start from the Blender clay render so composition survives at low denoise
      g['init'] = { class_type: 'LoadImage', inputs: { image: job.init.remote } };
      g['4'] = { class_type: 'VAEEncode', inputs: { pixels: ['init', 0], vae: ['1', 2] } };
      sampler.denoise = job.init.denoise ?? 0.6;
    } else {
      g['4'] = { class_type: 'EmptyLatentImage', inputs: { width: W, height: H, batch_size: 1 } };
    }
    let pos = ['2', 0];
    let neg = ['3', 0];
    job.controls.forEach((c, i) => {
      const cn = `cn${i}`;
      const im = `img${i}`;
      const ap = `ap${i}`;
      g[cn] = { class_type: 'ControlNetLoader', inputs: { control_net_name: c.model } };
      g[im] = { class_type: 'LoadImage', inputs: { image: c.remote } };
      g[ap] = {
        class_type: 'ControlNetApplyAdvanced',
        inputs: {
          positive: pos, negative: neg, control_net: [cn, 0], image: [im, 0],
          strength: c.strength ?? 0.75, start_percent: c.start ?? 0.0, end_percent: c.end ?? 1.0,
        },
      };
      pos = [ap, 0];
      neg = [ap, 1];
    });
    g['5'] = { class_type: 'KSampler', inputs: { ...sampler, model: ['1', 0], positive: pos, negative: neg, latent_image: ['4', 0] } };
    g['6'] = { class_type: 'VAEDecode', inputs: { samples: ['5', 0], vae: ['1', 2] } };
    g['9'] = { class_type: 'PreviewImage', inputs: { images: ['6', 0] } };
    return g;
  }

  // anima
  g['1'] = { class_type: 'UNETLoader', inputs: { unet_name: spec.unet, weight_dtype: 'default' } };
  g['2'] = { class_type: 'CLIPLoader', inputs: { clip_name: spec.clip, type: spec.clipType ?? 'stable_diffusion', device: 'default' } };
  g['3'] = { class_type: 'VAELoader', inputs: { vae_name: spec.vae } };
  g['4'] = { class_type: 'CLIPTextEncode', inputs: { text: job.positive, clip: ['2', 0] } };
  g['5'] = { class_type: 'CLIPTextEncode', inputs: { text: job.negative, clip: ['2', 0] } };
  if (job.init) {
    g['init'] = { class_type: 'LoadImage', inputs: { image: job.init.remote } };
    g['6'] = { class_type: 'VAEEncode', inputs: { pixels: ['init', 0], vae: ['3', 0] } };
    sampler.denoise = job.init.denoise ?? 0.6;
  } else {
    g['6'] = { class_type: 'EmptyLatentImage', inputs: { width: W, height: H, batch_size: 1 } };
  }
  let model = ['1', 0];
  job.controls.forEach((c, i) => {
    const im = `img${i}`;
    const ll = `ll${i}`;
    g[im] = { class_type: 'LoadImage', inputs: { image: c.remote } };
    g[ll] = {
      class_type: 'AnimaLLLiteApply_sdscripts',
      inputs: {
        model, lllite_name: c.model, image: [im, 0],
        strength: c.strength ?? 1.0, start_percent: c.start ?? 0.0, end_percent: c.end ?? 1.0,
        preserve_wrapper: true,
      },
    };
    model = [ll, 0];
  });
  g['7'] = {
    class_type: 'KSampler',
    inputs: { ...sampler, sampler_name: job.sampler ?? 'euler', scheduler: job.scheduler ?? 'simple', model, positive: ['4', 0], negative: ['5', 0], latent_image: ['6', 0] },
  };
  g['8'] = { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['3', 0] } };
  g['9'] = { class_type: 'PreviewImage', inputs: { images: ['8', 0] } };
  return g;
}

async function runOne(job) {
  for (const c of job.controls) {
    if (!c.remote) c.remote = await upload(path.isAbsolute(c.image) ? c.image : path.resolve(jobsDir, c.image));
  }
  if (job.init && !job.init.remote) {
    job.init.remote = await upload(path.isAbsolute(job.init.image) ? job.init.image : path.resolve(jobsDir, job.init.image));
  }
  const graph = buildGraph(job);
  const sub = await api('POST', '/prompt', { prompt: graph, client_id: 'dsh-blender-ctrl' });
  if (!sub.prompt_id) throw new Error('no prompt_id ' + JSON.stringify(sub).slice(0, 300));
  if (sub.node_errors && Object.keys(sub.node_errors).length) throw new Error('node_errors ' + JSON.stringify(sub.node_errors).slice(0, 500));
  const pid = sub.prompt_id;

  for (let i = 0; i < 600; i++) {
    const hist = await api('GET', '/history/' + pid);
    const rec = hist[pid];
    if (rec && rec.status && rec.status.status_str === 'error') {
      throw new Error('exec error ' + JSON.stringify(rec.status).slice(0, 700));
    }
    if (rec && rec.outputs && Object.keys(rec.outputs).length) {
      let img = null;
      for (const o of Object.values(rec.outputs)) if (o.images && o.images.length) { img = o.images[0]; break; }
      if (!img) throw new Error('no image in outputs');
      const q = `filename=${encodeURIComponent(img.filename)}&subfolder=${encodeURIComponent(img.subfolder || '')}&type=${encodeURIComponent(img.type || 'temp')}`;
      const vr = await fetch(base + '/view?' + q, { signal: AbortSignal.timeout(60000) });
      if (!vr.ok) throw new Error('view ' + vr.status);
      const buf = Buffer.from(await vr.arrayBuffer());
      const out = path.join(outDir, `${job.id}.png`);
      fs.writeFileSync(out, buf);
      return { id: job.id, seed: job.seed, file: out, bytes: buf.length };
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
    console.log(`OK  ${r.id}  seed=${r.seed}  ${r.seconds}s  ${r.file}`);
  } catch (e) {
    results.push({ id: job.id, error: String(e.message || e) });
    console.log(`ERR ${job.id}  ${String(e.message || e).slice(0, 400)}`);
  }
}
fs.writeFileSync(path.join(jobsDir, `${spec.pipeline}_results.json`), JSON.stringify(results, null, 2));
console.log(`done: ${results.filter((r) => !r.error).length}/${results.length}`);
