#!/usr/bin/env python
"""路线 A：扩散式 mask 精修器（SegRefiner 式离散扩散，T 步）——训练/推理入口。

任务形式（用户提的那个）：输入 **原图 + 粗 mask**，输出**正确 mask**；粗 mask 被当作"加噪的真值"，
用离散扩散（每像素 fine/coarse 两态、单向转移）迭代 T 步逐步纠正。**论文 T=6，推理很便宜。**

网络：4+1 通道输入（RGB + 当前 mask + 时间步常量平面）→ 小 U-Net（base 32，3 级）→ 1 通道 logits。
损失：masked BCE + α·texture loss（预测与真值的梯度幅值 L1；论文 α=5，对细结构连续性有帮助）。

训练对由 `scripts/rope_make_coarse.py` 生成（真值 + 形态学/断裂/误检块扰动），**不需要新标注**。

    # 训练（3 seed 见文档；单次约 5~7 分钟）
    python scripts\\rope_refiner.py train \\
        --coarse data/masks/rope/coarse --split train --steps 3000 --seed 0 --T 6 \\
        --tile 256 --base 32 --batch 8 --alpha-texture 5 --out .runtime/rope_cv/RF_E1_u

    # 推理：把**现有模型**的粗 mask 精修一遍（--src 指向 rope_predict 的输出目录）
    python scripts\\rope_refiner.py apply \\
        --weights .runtime/rope_cv/RF_E1_u/refiner.pt --src <粗 mask 目录> --out <精修输出目录>
"""
import argparse
import glob
import io
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'masks', 'rope', 'raw_ps')
ORIG = os.path.join(ROOT, '.runtime', 'rope_cv', 'ps_work', '_orig')


def find_image(name):
    for e in ('.png', '.jpg', '.jpeg', '.webp'):
        p = os.path.join(ORIG, name + e)
        if os.path.exists(p):
            return p
    g = sorted(glob.glob(os.path.join(ORIG, name + '_*')))
    return g[0] if g else None


# ---------------------------------------------------------------- 数据
class Pairs:
    def __init__(self, coarse_dir, split, tile=256, seed=0):
        man = json.load(io.open(os.path.join(coarse_dir, 'manifest.json'), encoding='utf-8'))
        self.items = []
        for m in man:
            if m['split'] != split:
                continue
            img = find_image(m['name'])
            lab = os.path.join(RAW, m['name'] + '_labels.png')
            cp = os.path.join(ROOT, m['path'])
            if img and os.path.exists(lab) and os.path.exists(cp):
                self.items.append((m['name'], img, lab, cp))
        self.tile = tile
        self.rng = np.random.RandomState(seed)
        self.cache = {}

    def _get(self, name):
        if name not in self.cache:
            img = None
            for nm, ip, lp, cp in self.items:
                if nm == name:
                    img = cv2.imread(ip, cv2.IMREAD_COLOR)
                    lab = cv2.imread(lp, 0)
                    coa = cv2.imread(cp, 0)
                    h = min(img.shape[0], lab.shape[0], coa.shape[0])
                    w = min(img.shape[1], lab.shape[1], coa.shape[1])
                    self.cache[name] = (img[:h, :w], (lab[:h, :w] == 1), (coa[:h, :w] > 0))
                    break
        return self.cache[name]

    def batch(self, bs):
        xs, gts, cos = [], [], []
        for _ in range(bs):
            name = self.items[self.rng.randint(len(self.items))][0]
            img, gt, coa = self._get(name)
            h, w = gt.shape
            s = self.tile
            if h < s or w < s:
                img = cv2.resize(img, (max(s, w), max(s, h)))
                gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_NEAREST) > 0
                coa = cv2.resize(coa.astype(np.uint8), (img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST) > 0
                h, w = gt.shape
            if self.rng.rand() < 0.5 and gt.sum() > 0:
                ys, xs2 = np.where(gt)
                i = self.rng.randint(len(ys))
                y0, x0 = int(ys[i]) - s // 2, int(xs2[i]) - s // 2
            else:
                y0, x0 = self.rng.randint(0, h - s + 1), self.rng.randint(0, w - s + 1)
            y0 = int(np.clip(y0, 0, h - s)); x0 = int(np.clip(x0, 0, w - s))
            patch = img[y0:y0 + s, x0:x0 + s][:, :, ::-1].astype(np.float32) / 255.0  # RGB
            xs.append(patch.transpose(2, 0, 1))
            gts.append(gt[y0:y0 + s, x0:x0 + s].astype(np.float32))
            cos.append(coa[y0:y0 + s, x0:x0 + s].astype(np.float32))
        return (torch.from_numpy(np.stack(xs)).float(),
                torch.from_numpy(np.stack(gts)).float(),
                torch.from_numpy(np.stack(cos)).float())


# ---------------------------------------------------------------- 网络
class Blk(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, padding=1)
        self.n1 = nn.GroupNorm(8 if o % 8 == 0 else 1, o)
        self.c2 = nn.Conv2d(o, o, 3, padding=1)
        self.n2 = nn.GroupNorm(8 if o % 8 == 0 else 1, o)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        return F.relu(self.n2(self.c2(h)))


class Refiner(nn.Module):
    """小 U-Net：输入 5 通道（RGB + 当前 mask + 时间步），输出 1 通道 logits。"""

    def __init__(self, base=32):
        super().__init__()
        b = base
        self.d1 = Blk(5, b)
        self.d2 = Blk(b, b * 2)
        self.d3 = Blk(b * 2, b * 4)
        self.bott = Blk(b * 4, b * 4)
        self.u3 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.c3 = Blk(b * 6, b * 2)      # 上采样 b*2 + skip e3(b*4) = b*6
        self.u2 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.c2 = Blk(b * 3, b)          # 上采样 b + skip e2(b*2) = b*3
        self.u1 = nn.ConvTranspose2d(b, b, 2, stride=2)
        self.c1 = Blk(b * 2, b)          # 上采样 b + skip e1(b) = b*2
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x, m, t01):
        if m.dim() == 3:                 # (B,H,W) → (B,1,H,W)
            m = m[:, None]
        tt = torch.full_like(m, float(t01))
        h = torch.cat([x, m, tt], 1)
        e1 = self.d1(h)
        e2 = self.d2(F.max_pool2d(e1, 2))
        e3 = self.d3(F.max_pool2d(e2, 2))
        bo = self.bott(F.max_pool2d(e3, 2))
        y = self.c3(torch.cat([self.u3(bo), e3], 1))
        y = self.c2(torch.cat([self.u2(y), e2], 1))
        y = self.c1(torch.cat([self.u1(y), e1], 1))
        return self.head(y)[:, 0]



class DitRefiner(nn.Module):
    """小 DiT 风格去噪器：patch 8、embed dim 可调、4 层、8 头；时间步用正弦 embedding 加性注入。

    与 Refiner（U-Net）**同输入同输出**（RGB + 当前 mask + 时间步 → 1 通道 logits），
    所以 A/B 的差别只有网络结构。位置编码用固定 sin-cos（不用可学参数，免得建 optimizer 之后
    才创建参数导致它不被训练）。
    """

    def __init__(self, base=128, patch=8, depth=4, heads=8):
        super().__init__()
        self.patch = patch
        self.embed = base
        self.proj = nn.Conv2d(4, base, patch, stride=patch)       # 4 通道：RGB + mask
        layer = nn.TransformerEncoderLayer(d_model=base, nhead=heads, dim_feedforward=base * 4,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(base)
        self.head = nn.Linear(base, patch * patch)
        self.tmlp = nn.Sequential(nn.Linear(base, base), nn.SiLU(), nn.Linear(base, base))

    @staticmethod
    def temb(t, dim):
        half = max(1, dim // 2)
        freqs = torch.exp(torch.linspace(0, 6.0, half, device=t.device))
        x = t[:, None].float() * freqs[None]
        e = torch.cat([torch.sin(x), torch.cos(x)], 1)
        return F.pad(e, (0, dim - e.shape[1])) if e.shape[1] < dim else e[:, :dim]

    @staticmethod
    def posenc(h, w, c, device):
        ys = torch.linspace(0, 1, h, device=device)
        xs = torch.linspace(0, 1, w, device=device)
        grid = torch.stack(torch.meshgrid(ys, xs, indexing='ij'), -1).reshape(-1, 2)   # (N,2)
        half = c // 2
        freqs = torch.exp(torch.linspace(0, 4.0, half // 2, device=device))
        p = torch.cat([torch.sin(grid[:, :1] * freqs), torch.cos(grid[:, :1] * freqs),
                       torch.sin(grid[:, 1:] * freqs), torch.cos(grid[:, 1:] * freqs)], 1)
        return F.pad(p, (0, c - p.shape[1]))[None] if p.shape[1] < c else p[:, :c][None]

    def forward(self, x, m, t01):
        if m.dim() == 3:
            m = m[:, None]
        z = torch.cat([x, m], 1)
        p = self.proj(z)                                         # (B,C,h,w)
        b, c, h, w = p.shape
        tok = p.flatten(2).transpose(1, 2)                        # (B,N,C)
        tok = tok + self.posenc(h, w, c, tok.device)
        tok = tok + self.tmlp(self.temb(torch.full((b,), float(t01), device=tok.device), c))[:, None]
        tok = self.norm(self.blocks(tok))
        out = self.head(tok)                                      # (B,N,p*p)
        out = out.reshape(b, h, w, self.patch, self.patch)
        out = out.permute(0, 1, 3, 2, 4).reshape(b, h * self.patch, w * self.patch)
        return out


def schedule(T):
    """β̄_t：从 0.8 线性降到 0（t 越大越接近粗 mask）。返回长度 T+1 的数组，末尾 0。"""
    b = np.linspace(0.8, 0.0, T + 1)
    b[-1] = 0.0
    return b


def texture_loss(pred, gt):
    if pred.dim() == 3:              # (B,H,W) → (B,1,H,W)，便于统一写梯度
        pred = pred[:, None]
        gt = gt[:, None]

    def grad(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx.abs().mean() + dy.abs().mean()
    return (grad(pred) - grad(gt)).abs()


def train(a):
    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = Pairs(a.coarse, a.split, tile=a.tile, seed=a.seed)
    print('训练对 %d（split=%s, tile=%d）' % (len(ds.items), a.split, a.tile), flush=True)
    model = (DitRefiner(a.base) if getattr(a, 'net', 'unet') == 'dit'
             else Refiner(a.base)).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print('Refiner 参数量 %.2fM  设备 %s  T=%d' % (n_par / 1e6, dev, a.T), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    bar = schedule(a.T)
    t0 = time.time()
    for st in range(1, a.steps + 1):
        x, gt, coa = ds.batch(a.batch)
        x, gt, coa = x.to(dev), gt.to(dev), coa.to(dev)
        t = int(np.random.randint(a.T))          # 随机时间步
        keep = float(bar[t])                     # 保留为 fine 的概率
        u = torch.rand_like(gt)
        m = torch.where(u < keep, gt, coa)       # 前向：真值 → 粗 mask
        logit = model(x, m, t / max(1, a.T))
        w = torch.ones_like(gt)
        bce = (F.binary_cross_entropy_with_logits(logit, gt, reduction='none') * w).mean()
        tex = texture_loss(torch.sigmoid(logit), gt)
        loss = bce + a.alpha_texture * tex
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 200 == 0:
            print('  step %d/%d  loss %.4f (bce %.4f tex %.4f)  %.0fs'
                  % (st, a.steps, float(loss), float(bce), float(tex), time.time() - t0), flush=True)
    torch.save({'sd': model.state_dict(), 'base': a.base, 'T': a.T, 'tile': a.tile,
                'net': getattr(a, 'net', 'unet'),
                'alpha_texture': a.alpha_texture, 'steps': a.steps, 'seed': a.seed}, a.weights)
    print('已保存 %s' % a.weights, flush=True)


def refine_image(model, img_bgr, coarse, T, base, tile):
    """整图精修：分块推理（重叠平均），离散扩散 T 步（反向转移见 SegRefiner Eq.11）。"""
    dev = next(model.parameters()).device
    h, w = coarse.shape
    m = coarse.astype(np.float32)
    fine = np.zeros((h, w), bool)
    img = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    bar = schedule(T)
    step = max(1, tile // 2)
    for t in range(T - 1, -1, -1):
        pred = np.zeros((h, w), np.float32)
        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
                y0b, x0b = max(0, y1 - tile), max(0, x1 - tile)
                pi = torch.from_numpy(img[y0b:y1, x0b:x1].transpose(2, 0, 1)[None]).float().to(dev)
                mi = torch.from_numpy(m[y0b:y1, x0b:x1][None]).float().to(dev)
                with torch.no_grad():
                    out = torch.sigmoid(model(pi, mi, t / max(1, T)))
                pred[y0b:y1, x0b:x1] = out[0].cpu().numpy()
        fine_pred = pred >= 0.5
        pf = np.clip(pred, 1e-6, 1 - 1e-6)
        b_t, b_prev = bar[t], bar[max(0, t - 1)]
        p_trans = np.clip(pf * (b_prev - b_t) / (1 - pf * b_t), 0.0, 1.0)
        newly = np.random.rand(h, w) < p_trans
        m = np.where(fine, fine_pred.astype(np.float32),
                     np.where(newly, fine_pred.astype(np.float32), m))
        fine = fine | newly | fine_pred
    return (m >= 0.5).astype(np.uint8) * 255


def apply(a):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = torch.load(a.weights, map_location=dev, weights_only=False)
    model = (DitRefiner(z.get('base', 128)) if z.get('net', 'unet') == 'dit'
             else Refiner(z['base'])).to(dev).eval()
    model.load_state_dict(z['sd'])
    T = int(a.T or z['T'])
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.src, '*_mask.png')))
    print('待精修 %d 张（T=%d）' % (len(files), T), flush=True)
    for p in files:
        name = os.path.basename(p)[:-len('_mask.png')]
        ip = find_image(name)
        if ip is None:
            print('  跳过（找不到原图）', name)
            continue
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        coa = cv2.imread(p, 0)
        h = min(img.shape[0], coa.shape[0]); w = min(img.shape[1], coa.shape[1])
        out = refine_image(model, img[:h, :w], (coa[:h, :w] > 0), T, z['base'], z['tile'])
        cv2.imwrite(os.path.join(a.out, name + '_mask.png'), out)
        print('  %s 覆盖率 %.2f%%' % (name, 100.0 * (out > 0).mean()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--coarse', default=os.path.join(ROOT, 'data', 'masks', 'rope', 'coarse'))
    t.add_argument('--split', default='train')
    t.add_argument('--steps', type=int, default=3000)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--T', type=int, default=6)
    t.add_argument('--tile', type=int, default=256)
    t.add_argument('--base', type=int, default=32,
                   help='unet 通道基数；--net dit 时是 transformer embed dim（dit 建议 128）')
    t.add_argument('--net', default='unet', choices=['unet', 'dit'])
    t.add_argument('--batch', type=int, default=8)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--alpha-texture', type=float, default=5.0)
    t.add_argument('--out', required=True)
    t.add_argument('--weights', default='')
    p = sub.add_parser('apply')
    p.add_argument('--weights', required=True)
    p.add_argument('--src', required=True, help='粗 mask 目录（rope_predict 的 --out）')
    p.add_argument('--out', required=True)
    p.add_argument('--T', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'train':
        os.makedirs(a.out, exist_ok=True)
        if not a.weights:
            a.weights = os.path.join(a.out, 'refiner.pt')
        train(a)
        # 训练完自动在**验证图**上精修一次，便于快速看效果（不碰最终评测图）
        man = json.load(io.open(os.path.join(a.coarse, 'manifest.json'), encoding='utf-8'))
        names = sorted({m['name'] for m in man if m['split'] == 'val'})
        src = os.path.join(a.out, 'val_coarse')
        os.makedirs(src, exist_ok=True)
        for n in names:
            lab = cv2.imread(os.path.join(RAW, n + '_labels.png'), 0)
            cp = sorted(glob.glob(os.path.join(a.coarse, 'val', n + '_c0.png')))
            if lab is not None and cp:
                cv2.imwrite(os.path.join(src, n + '_mask.png'), cv2.imread(cp[0], 0))
        a.src = src
        a.out = os.path.join(os.path.dirname(src), 'val_refined')
        a.T = 0
        apply(a)
    else:
        apply(a)


if __name__ == '__main__':
    main()
