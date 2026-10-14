#!/usr/bin/env python
"""绳子 mask 推理入口（固定 CLI，产线统一调用）。

  python scripts/rope_predict.py --src <图片或目录> --out <目录> [--conf 0.565]

产物三件套（<name> = 图片主名，空格转下划线）：
  <out>/<name>_mask.png                  二值 0/255（重绘蒙版）
  <out>/<name>_conf.png                  0~255 灰度置信度（可直接当软蒙版，省掉羽化）
  <overlay-dir>/<name>_overlay.png       给人看的叠加图（默认 .runtime/rope_cv/overlay）

模型（两条路，`--weights` 显式指定时以你指定的为准）：
  * `.pt`  UNetPre（预训练骨干 + 薄解码器）——**当前交付版**
    `rope_unet_resnet18_g14.pt`：resnet18 骨干不冻结 + RGB + ±20° 旋转，**79 张人工标注**训练，
    留出集 F1 0.9178 / 召回 0.9812（旧 30 图版 0.848 / 0.838）。需 torch。阈值存在 .pt 里（`--conf` 可覆盖）。
    见 `docs/rope_runs_check_2026-10-13.md` §7 与 `docs/rope_C_result.md` §2c。
  * `.npz` k×k 邻域 HSV 摊平 → 小 MLP（纯 numpy，不需要 torch）——**兜底路径，一直保留可用**。
默认权重 = 有权重文件且有 torch 就用 .pt，否则自动退回 MLP（`default_weights()`）。
逐行分块做稠密滑窗，避免 2364×1773 图一次性展开 (H*W*k*k*3) 撑爆内存。
"""
import argparse
import glob
import os
import time

import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W_DIR = os.path.join(ROOT, 'data', 'masks', 'rope', 'weights')


def default_weights():
    """默认交付权重：有 torch 且交付权重在位，就用 UNetPre(resnet18 不冻结)；否则退回不依赖 torch 的 MLP 兜底。

    2026-10-14 换权重：`rope_unet_resnet18_g14.pt`（G14 配置 = resnet18 不冻结 + RGB + ±20° 旋转，
    **79 张人工标注**训练）取代 `rope_unet_resnet18_full.pt`（30 张，thr 0.92）。同一留出集 s01~s05、
    同一 `rope_eval.py` 口径：

    | 交付权重 | P | R | F1 | 笔数（误检块+漏检段） |
    |---|---|---|---|---|
    | 旧 full（30 图） | 0.899 | 0.838 | 0.848 | 298（182+116） |
    | 新 g14（79 图） | 0.862 | **0.9812** | **0.9178** | 340（294+46） |

    即**从"漏绳"变成"更敢报"**（漏检段 −60%、误检块 +62%，笔数 +14%）——按本项目"宁可略偏大、
    不要漏阴影段"的偏好选它。**交付 s0**（预注册的 seed 0，不按留出集挑选）；另外两个 seed：
    F1 0.9211 / 0.9441，笔数 279 / 281。旧权重仍在盘上，`--weights` 可指回去。
    """
    for pt in (os.path.join(W_DIR, 'rope_unet_resnet18_g14.pt'),
               os.path.join(W_DIR, 'rope_unet_resnet18_full.pt')):
        if os.path.exists(pt):
            try:
                import torch  # noqa: F401
                return pt
            except Exception:
                break
    return os.path.join(W_DIR, 'rope_mlp_k9_4img.npz')


DEFAULT_W = default_weights()


def load_unet_weights(path):
    """加载交付版 UNetPre（.pt）：返回 {'kind':'unet', 'model':..., 'thr':..., 'tile':...}

    早期实验导出的 fold 权重只存了 sd/thr，没存 encoder/mid —— 这里从 state_dict 反推，
    否则一加载就 KeyError（等于"实验跑出来的模型不能在产线入口里用"）。
    """
    import torch
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import rope_unet as RU
    z = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu',
                   weights_only=False)
    sd = z['sd']
    enc = z.get('encoder')
    if enc is None:
        ks = list(sd.keys())
        if any('relative_position_bias_table' in k for k in ks):
            enc = 'swin_t'                       # Swin：enc.body.<stage>.<blk>.attn.*
        elif any(k.startswith('enc.body.0.norm1.') for k in ks):
            enc = 'vit_b_16'                     # ViT：enc.body.<blk>.norm1.*
        elif any(k.startswith('enc.body.model.') for k in ks):
            enc = 'yolo11n_seg'
        elif any(k.startswith('enc.body.features.') for k in ks):
            enc = 'mobilenet_v3_small'
        elif any(k.startswith('enc.body.') for k in ks):
            enc = 'resnet18'
        else:
            raise ValueError('无法从权重里判断 backbone：%s' % path)
    mid = z.get('mid')
    if mid is None:
        mid = int(sd['proj.0.weight'].shape[0]) if 'proj.0.weight' in sd else 128
    m = RU.UNetPre(z['base'], enc, mid=int(mid), in_ch=int(z.get('in_ch', 3)))
    m.load_state_dict(sd)
    m = m.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    return {'kind': 'unet', 'model': m, 'thr': float(z['thr']), 'tile': int(z.get('tile', 512)),
            'encoder': enc, 'n_images': z.get('n_images'),
            'in_mode': z.get('in_mode', 'hsv')}


def dense_prob_unet(m, hsv, tile=512, overlap=64):
    """UNetPre 的分块稠密推理（hsv 是 0~1 的 float32，HWC）。"""
    import torch
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import rope_unet as RU
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return RU.predict_full(m['model'].to(dev), hsv, tile=tile, overlap=overlap,
                           in_mode=m.get('in_mode', 'hsv'))


def load_model(path):
    if str(path).lower().endswith('.pt'):
        return load_unet_weights(path)
    z = np.load(path)
    return {'w1': z['w1'], 'b1': z['b1'], 'w2': z['w2'], 'b2': z['b2'], 'w3': z['w3'],
            'b3': z['b3'], 'mean': z['mean'], 'std': z['std'], 'k': int(z['k'][0]),
            'thr': float(z['thr'][0])}


def forward(m, X):
    h = np.maximum(X @ m['w1'] + m['b1'], 0.0)
    h = np.maximum(h @ m['w2'] + m['b2'], 0.0)
    z = (h @ m['w3'] + m['b3']).ravel()
    return 1.0 / (1.0 + np.exp(-z))


def dense_prob(hsv, m, chunk_rows=48, backend='auto'):
    """整图稠密滑窗推理，返回 float32 概率图。

    两条后端**数学等价**：同一个 MLP，第一层就是 k×k 卷积（权重按训练时的 (dy,dx,c) 序重排），
    后两层是 1×1 卷积。
      torch → 真卷积，GPU 上 ~10ms/Mpx（生产用，需 me2/sdenv 这类带 torch 的环境）
      numpy → 展开滑窗 + matmul，不依赖 torch（兜底，实测 ~8600 ms/Mpx，慢约一千倍）
    """
    k = m['k']
    r = k // 2
    h, w = hsv.shape[:2]
    if backend in ('auto', 'torch'):
        try:
            import torch
            import torch.nn.functional as F
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            # 训练特征序 (dy,dx,c)：第一层就是 k×k 卷积。标准化是**逐 tap** 的
            # (mu/sd 也是 3kk 维)，没法当成"每通道一次"来做，所以把它折进卷积权重：
            #   sum W*(x-mu)/sd + b = sum (W/sd)*x + (b - sum W*mu/sd)
            kk = k * k
            W1r = m['w1'].reshape(k, k, 3, -1)
            mu_r = m['mean'][0].reshape(k, k, 3)[..., None]
            sd_r = m['std'][0].reshape(k, k, 3)[..., None]
            W1n = W1r / sd_r
            b1n = m['b1'] - (W1r * mu_r / sd_r).sum(axis=(0, 1, 2))
            W1 = torch.from_numpy(np.ascontiguousarray(
                W1n.transpose(3, 2, 0, 1))).to(dev)                            # (48,3,k,k)
            b1 = torch.from_numpy(b1n).to(dev)
            W2 = torch.from_numpy(np.ascontiguousarray(m['w2'].T[:, :, None, None])).to(dev)  # (24,48,1,1)
            b2 = torch.from_numpy(m['b2']).to(dev)
            W3 = torch.from_numpy(np.ascontiguousarray(m['w3'].T[:, :, None, None])).to(dev)  # (1,24,1,1)
            b3 = torch.from_numpy(m['b3']).to(dev)
            x = torch.from_numpy(np.ascontiguousarray(hsv.transpose(2, 0, 1))).to(dev)[None]
            out = []
            step = 1024
            with torch.no_grad():
                for y0 in range(0, h, step):
                    y1 = min(h, y0 + step)
                    xb = F.pad(x[:, :, y0:y1], (r, r, r, r), mode='replicate')
                    hn = torch.relu(F.conv2d(xb, W1, b1))
                    hn = torch.relu(F.conv2d(hn, W2, b2))
                    out.append(torch.sigmoid(F.conv2d(hn, W3, b3))[0, 0].cpu().numpy())
            return np.concatenate(out, 0).astype(np.float32)
        except Exception as e:
            if backend == 'torch':
                raise
            print('  (torch 后端不可用，退回 numpy: %s)' % e)
    pad = np.pad(hsv, ((r, r), (r, r), (0, 0)), mode='edge')
    prob = np.zeros((h, w), np.float32)
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        win = np.lib.stride_tricks.sliding_window_view(pad[y0:y1 + 2 * r], (k, k), axis=(0, 1))
        X = win.transpose(0, 1, 3, 4, 2).reshape(-1, k * k * 3)
        X = ((X - m['mean']) / m['std']).astype(np.float32)
        p = np.empty(len(X), np.float32)
        step = 20000                                   # 同时限制临时数组大小
        for i in range(0, len(X), step):
            p[i:i + step] = forward(m, X[i:i + step])
        prob[y0:y1] = p.reshape(y1 - y0, w)
    return prob


def overlay(img, mask):
    vis = img.copy()
    m = mask > 0
    if m.any():
        vis[m] = (0.45 * vis[m] + 0.55 * np.array([60, 60, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours((m * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (255, 255, 0), 1, cv2.LINE_AA)
    return vis


def resolve(src):
    if any(c in src for c in '*?['):
        return sorted(glob.glob(src))
    if os.path.isdir(src):
        out = []
        for e in ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp'):
            out += glob.glob(os.path.join(src, e))
        return sorted(out)
    return [src]


def main():
    ap = argparse.ArgumentParser(description='绳子 mask 推理（固定接口）')
    ap.add_argument('--src', required=True, help='图片路径 / 目录 / glob')
    ap.add_argument('--out', required=True, help='mask 与 conf 的输出目录')
    ap.add_argument('--overlay-dir', default=os.path.join(ROOT, '.runtime', 'rope_cv', 'overlay'))
    ap.add_argument('--weights', default=DEFAULT_W)
    ap.add_argument('--conf', type=float, default=None, help='二值阈值；默认用权重里存的 bestThr')
    ap.add_argument('--min-area', type=int, default=32, help='去掉小于该面积的碎块（px）')
    ap.add_argument('--conf-blur', type=float, default=0.0, help='置信度图羽化 sigma；0=不羽化')
    ap.add_argument('--backend', default='auto', choices=['auto','torch','numpy'], help='auto=有 torch 就用 GPU')
    ap.add_argument('--save-prob', action='store_true', help='额外存 <name>_prob.npy')
    a = ap.parse_args()

    m = load_model(a.weights)
    thr = m['thr'] if a.conf is None else a.conf
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.overlay_dir, exist_ok=True)
    files = resolve(a.src)
    if isinstance(m, dict) and m.get('kind') == 'unet':
        print('权重 %s (UNetPre/%s, thr=%.3f)  待处理 %d 张'
              % (os.path.basename(a.weights), m.get('encoder'), thr, len(files)))
    else:
        print('权重 %s (k=%d, thr=%.3f)  待处理 %d 张'
              % (os.path.basename(a.weights), m['k'], thr, len(files)))
    for fp in files:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            print('读取失败:', fp)
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        t0 = time.time()
        if isinstance(m, dict) and m.get('kind') == 'unet':
            prob = dense_prob_unet(m, hsv, tile=m.get('tile', 512))
        else:
            prob = dense_prob(hsv, m, backend=a.backend)
        dt = time.time() - t0
        mask = (prob >= thr).astype(np.uint8) * 255
        if a.min_area > 1:
            n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            drop = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < a.min_area]
            if drop:
                mask[np.isin(cc, drop)] = 0
        conf = np.clip(prob, 0, 1) * 255
        if a.conf_blur:
            conf = cv2.GaussianBlur(conf, (0, 0), a.conf_blur)
            conf[mask > 0] = 255
        conf = conf.astype(np.uint8)
        base = os.path.splitext(os.path.basename(fp))[0].replace(' ', '_')
        cv2.imwrite(os.path.join(a.out, base + '_mask.png'), mask)
        cv2.imwrite(os.path.join(a.out, base + '_conf.png'), conf)
        cv2.imwrite(os.path.join(a.overlay_dir, base + '_overlay.png'), overlay(img, mask))
        if a.save_prob:
            np.save(os.path.join(a.out, base + '_prob.npy'), prob.astype(np.float16))
        ncomp = int(cv2.connectedComponents((mask > 0).astype(np.uint8), 8)[0]) - 1
        print('[%s] %dx%d 覆盖率 %.3f%%  连通块 %d  thr %.3f  稠密推理 %.1fs (%.1f ms/Mpx)'
              % (base, img.shape[1], img.shape[0], 100 * (mask > 0).mean(), ncomp, thr, dt,
                 1000 * dt / (img.shape[0] * img.shape[1] / 1e6)))


if __name__ == '__main__':
    main()
