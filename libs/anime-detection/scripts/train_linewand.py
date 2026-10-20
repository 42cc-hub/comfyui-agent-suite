r"""轻量"超级魔棒"：输入彩色切片 → 输出 **1 通道线稿概率**（目标 = 人工边界带 ∩ 图像边缘）。

为什么是 1 通道 + 线稿（用户 2026-10-14 的思路）：输出"数量不定的块"很难，输出"一条线"容易得多；
拿到线稿后 agent 自己 flooding 就能选块。**线稿闭合性**在这里不是硬要求 —— 下游把线当作
**代价屏障**（线=高代价）而不是"墙"，漏一小段也不会整块漏出去（见 scripts/colorblock.py --mode compete）。

为什么目标用 `line_edge` 而不是人工边界本身（实测四联图看出来的）：
人工边界里混着**画法伪影**（矩形选区、笔刷边），模型会去学笔刷；`line_edge` 只保留
"人工认定该分 ∧ 图像真有边"的那些，正是用户说的"提供细碎细节与渐变融并的范本"。

数据：`data/colorblock_ds/{originals,targets}`；**按源图**留出（同一张图切出的 9 块会泄漏，绝不能随机切）。
产物：`data/colorblock_linewand/<name>/{model.pt, pred/<stem>_prob.png, log.txt}`

用法：python scripts/train_linewand.py [--steps 1500] [--name lw_v1] [--holdout ink20260928]
"""
import argparse
import glob
import io
import os
import random
import sys

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, 'data')
DS = os.path.join(DATA, 'colorblock_ds')


class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co),
                               nn.ReLU(inplace=True))

    def forward(self, x):
        return self.b(x)


class TinyUNet(nn.Module):
    """base=24 ⇒ 约 0.5M 参数，256 瓦片上一秒几十步（6GB 卡够用）。"""

    def __init__(self, base=24, cin=3):
        super().__init__()
        self.d1 = ConvBlock(cin, base)
        self.d2 = ConvBlock(base, base * 2)
        self.d3 = ConvBlock(base * 2, base * 4)
        self.b = ConvBlock(base * 4, base * 8)
        self.u3 = ConvBlock(base * 8 + base * 4, base * 4)
        self.u2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.u1 = ConvBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # 坑：三层 pooling ⇒ 输入必须能被 8 整除，而切片边长是 600/667（奇数也能出现）
        # ⇒ 先 replicate 补齐到 8 的倍数、算完再裁回原尺寸。
        _, _, H, W = x.shape
        ph, pw = (-H) % 8, (-W) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        b = self.b(self.pool(x3))
        y = self.u3(torch.cat([F.interpolate(b, scale_factor=2, mode='nearest'), x3], 1))
        y = self.u2(torch.cat([F.interpolate(y, scale_factor=2, mode='nearest'), x2], 1))
        y = self.u1(torch.cat([F.interpolate(y, scale_factor=2, mode='nearest'), x1], 1))
        return self.out(y)[:, :, :H, :W]


def load_items():
    items = []
    for op in sorted(glob.glob(os.path.join(DS, 'originals', '*.png'))):
        stem = os.path.splitext(os.path.basename(op))[0]
        lp = os.path.join(DS, 'targets', stem + '_line_edge.png')
        if not os.path.exists(lp):
            continue                                  # 未标注
        img = cv2.imread(op, cv2.IMREAD_COLOR)
        line = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        if img is None or line is None:
            continue
        items.append({'stem': stem, 'src': stem.split('__')[0], 'img': img, 'line': (line > 127)})
    return items


def f1(pred, gt, eps=1e-6):
    tp = float((pred & gt).sum())
    return 2 * tp / (pred.sum() + gt.sum() + eps), tp / (pred.sum() + eps), tp / (gt.sum() + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--base', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--pos-weight', type=float, default=8.0, help='线只占 ~1.6%%，正样本要加权')
    ap.add_argument('--name', default='lw_v1')
    ap.add_argument('--holdout', nargs='*', default=['ink20260928', 'odette20260928'],
                    help='按**源图前缀**留出（同一张图切出的 9 块必须同侧，否则泄漏）')
    ap.add_argument('--val-every', type=int, default=250)
    a = ap.parse_args()
    out = os.path.join(DATA, 'colorblock_linewand', a.name)
    os.makedirs(os.path.join(out, 'pred'), exist_ok=True)
    items = load_items()
    if not items:
        raise SystemExit('没有已标注的切片（先跑 colorblock_targets.py）')
    ho = set(a.holdout)
    tr = [it for it in items if it['src'] not in ho]
    va = [it for it in items if it['src'] in ho]
    print('切片 %d：训练 %d（源 %s）/ 留出 %d（源 %s）'
          % (len(items), len(tr), sorted(set(i['src'] for i in tr)),
             len(va), sorted(set(i['src'] for i in va))))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = TinyUNet(a.base).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    rng = random.Random(20261014)
    log = io.open(os.path.join(out, 'log.txt'), 'w', encoding='utf-8', newline='\n')

    def tiles(batch_items):
        xs, ys = [], []
        for _ in range(a.batch):
            it = rng.choice(batch_items)
            H, W = it['img'].shape[:2]
            t = a.tile
            x0, y0 = rng.randrange(0, max(1, W - t)), rng.randrange(0, max(1, H - t))
            x0, y0 = min(x0, max(0, W - t)), min(y0, max(0, H - t))
            xs.append(it['img'][y0:y0 + t, x0:x0 + t])
            ys.append(it['line'][y0:y0 + t, x0:x0 + t])
        x = np.stack(xs).astype(np.float32) / 255.0
        y = np.stack(ys).astype(np.float32)[:, None]
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
        y = torch.from_numpy(y).to(dev)
        return x, y

    @torch.no_grad()
    def evaluate(step):
        net.eval()
        rows = []
        for it in va or items:
            img = it['img'].astype(np.float32) / 255.0
            x = torch.from_numpy(img).permute(2, 0, 1)[None].to(dev)
            p = torch.sigmoid(net(x))[0, 0].cpu().numpy()
            for thr in (0.3, 0.5):
                f, pr, rc = f1(p > thr, it['line'])
                rows.append((thr, f, pr, rc))
            cv2.imwrite(os.path.join(out, 'pred', it['stem'] + '_prob.png'),
                        (p * 255).astype(np.uint8))
        net.train()
        for thr in (0.3, 0.5):
            r = [x for x in rows if x[0] == thr]
            line = '  step %5d  thr %.1f  线 F1 %.3f  P %.3f  R %.3f  (n=%d)' % (
                step, thr, np.mean([x[1] for x in r]), np.mean([x[2] for x in r]),
                np.mean([x[3] for x in r]), len(r))
            print(line)
            log.write(line + '\n')
            log.flush()
        return rows

    best = 0.0
    for step in range(1, a.steps + 1):
        x, y = tiles(tr)
        logits = net(x)
        bce = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=torch.tensor(a.pos_weight, device=dev))
        prob = torch.sigmoid(logits)
        dice = 1 - (2 * (prob * y).sum() + 1) / (prob.sum() + y.sum() + 1)
        loss = bce + dice
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print('  step %5d  loss %.4f (bce %.4f dice %.4f)' % (step, loss.item(), bce.item(), dice.item()))
        if step % a.val_every == 0 or step == a.steps:
            rows = evaluate(step)
            f_at_5 = np.mean([x[1] for x in rows if x[0] == 0.5])
            if f_at_5 >= best:
                best = f_at_5
                torch.save({'sd': net.state_dict(), 'base': a.base, 'thr': 0.5,
                            'tile': a.tile, 'step': step, 'val_f1': float(f_at_5)}, 
                           os.path.join(out, 'model.pt'))
                log.write('  ↑ 保存 model.pt（val F1 %.3f）\n' % f_at_5)
                log.flush()
    print('最佳 val 线 F1 %.3f → %s' % (best, os.path.join(out, 'model.pt')))
    log.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
