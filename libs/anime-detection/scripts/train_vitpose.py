"""微调强骨干 ViTPose（top-down）：**面部用人工真值拉回来，身体用自动池轻度修正**。

设计（对应 2026-10-13 用户的决定）：
  * 面点 5 个（鼻/双眼/双耳）：只有**人工真值**里有 vis>=1，自动池的面点按"宁可漏标"已全部置 vis=0
    ⇒ 天然不参与监督，脸部完全由人的标注决定；
  * 身体 12 个：自动池（= dwpose 的输出）+ 人工真值一起监；自动样本权重低（1×）、人工高（重复 K 次），
    这就是"让 ViTPose 从 dwpose 那里学动画躯干"，而不是小模型从零猜；
  * backbone 可冻结（--freeze-backbone）只训解码头 ⇒ 最保守的"轻度修正"；否则用很小的 lr（默认 2e-5）。
  * 裁图框：优先用 M2a 人框（和推理一致），没有就用骨架外接框 pad 25%；训练时加尺度/平移/翻转抖动。

损失：heatmap MSE，**逐关节掩码**（真值 vis<1 的关节不产生梯度），面部关节权重 3×。
评测：与 scripts/pose_eval_vs_human.py 同口径 —— 留出集上 GT vis>=1 关节的中位像素误差（用 M2a 框推理）。

用法：
  python scripts/train_vitpose.py --epochs 40 --human-repeat 5            # 正式
  python scripts/train_vitpose.py --epochs 1 --limit 40 --batch 8 --smoke # 冒烟
"""
import argparse
import csv
import io
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from export_labels import COCO17_FLIP, COCO17_NAMES                          # noqa: E402

DATA = os.path.join(PROJ, 'data')
GT = os.path.join(DATA, 'pose_gt')
PRE = os.path.join(DATA, 'prelabel_face_full')
VITPOSE_DIR = os.path.join(PROJ, 'models', 'vitpose')
BODY17 = ('nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
          'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle')
# 真值/预标注里的名字（BODY18 缩写）→ 模型输出的 COCO-17 名字。
# **必须统一到 COCO 名字**：make_target / 翻转 / 评测都按 COCO17_NAMES 索引；
# 混用会让除 nose 以外的关节全部对不上（冒烟实测：只训到 1 个关节，翻转样本直接没有目标）。
B2C = {'nose': 'nose', 'l_eye': 'left_eye', 'r_eye': 'right_eye', 'l_ear': 'left_ear', 'r_ear': 'right_ear',
       'l_shoulder': 'left_shoulder', 'r_shoulder': 'right_shoulder', 'l_elbow': 'left_elbow',
       'r_elbow': 'right_elbow', 'l_wrist': 'left_wrist', 'r_wrist': 'right_wrist',
       'l_hip': 'left_hip', 'r_hip': 'right_hip', 'l_knee': 'left_knee', 'r_knee': 'right_knee',
       'l_ankle': 'left_ankle', 'r_ankle': 'right_ankle'}
FACE_IDX = [0, 1, 2, 3, 4]
IN_W, IN_H = 192, 256
HM_W, HM_H = IN_W // 4, IN_H // 4


def jload(p):
    return json.load(io.open(p, encoding='utf-8'))


def norm_pts(rec):
    """prelabel / GT 记录 → {chain: {COCO17 名字: (x, y, vis)}}（只取 body，名字统一到 COCO）。"""
    by = {}
    for q in rec.get('points') or []:
        if q.get('cls') != 'body':
            continue
        nm = B2C.get(q['name'].split('.')[-1])
        if nm:          # B2C 只认我们这 17 个关节，其余（手/脚细点）返回 None
            by.setdefault(q.get('chain'), {})[nm] = (float(q['x']), float(q['y']), int(q.get('vis') or 0))
    return by


def box_of(pts, pad=0.25, iw=None, ih=None):
    v = [p for p in pts.values() if p[2] >= 1]
    if len(v) < 2:
        return None
    xs, ys = [p[0] for p in v], [p[1] for p in v]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    x0, y0, x1, y1 = x0 - pad * w, y0 - pad * h, x1 + pad * w, y1 + pad * h
    if iw:
        x0, x1 = max(0.0, x0), min(float(iw), x1)
    if ih:
        y0, y1 = max(0.0, y0), min(float(ih), y1)
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def person_box(rec, ref_box):
    """M2a 人框里与骨架框 IoU 最大的那个（和推理口径一致）。"""
    best, bb = 0.0, None
    for b in rec.get('boxes') or []:
        if b.get('cls') != 'person':
            continue
        x0, y0, x1, y1 = b['box']
        ix = max(0.0, min(x1, ref_box[2]) - max(x0, ref_box[0]))
        iy = max(0.0, min(y1, ref_box[3]) - max(y0, ref_box[1]))
        inter = ix * iy
        ua = (x1 - x0) * (y1 - y0) + (ref_box[2] - ref_box[0]) * (ref_box[3] - ref_box[1]) - inter
        v = inter / ua if ua > 0 else 0.0
        if v > best:
            best, bb = v, [x0, y0, x1, y1]
    return bb


def build_samples(human_repeat, use_auto, limit=0, min_auto_vis=7, seed=20261013, drop_stems=(),
                  box_src='person', auto_dir=None):
    man = jload(os.path.join(DATA, 'ds_pose17_semi', 'semi_manifest.json'))
    val = set(man['val_stems'])
    drop = set(drop_stems)
    human, auto = [], []
    for p in sorted(os.listdir(GT)):
        if not p.endswith('.json'):
            continue
        stem = p[:-5]
        if stem in val or stem in drop:
            continue
        g, rec = jload(os.path.join(GT, p)), jload(os.path.join(PRE, p))
        iw, ih = [int(v) for v in rec['size'].split('x')]
        for ch, pts in norm_pts(g).items():
            b = box_of(pts, iw=iw, ih=ih)
            if not b:
                continue
            human.append({'stem': stem, 'chain': ch, 'src': rec['src'],
                          'box': (b if box_src == 'label' else (person_box(rec, b) or b)),
                          'pts': pts, 'size': (iw, ih), 'human': True})
    AUTO = auto_dir or PRE
    print('  自动池标签来源：%s' % AUTO)
    if use_auto:
        for p in sorted(os.listdir(AUTO)):
            stem = p[:-5]
            if stem in val or os.path.exists(os.path.join(GT, p)):
                continue
            rec = jload(os.path.join(AUTO, p))
            iw, ih = [int(v) for v in rec['size'].split('x')]
            for ch, pts in norm_pts(rec).items():
                if sum(1 for q in pts.values() if q[2] >= 1) < min_auto_vis:
                    continue
                b = box_of(pts, iw=iw, ih=ih)
                if not b:
                    continue
                auto.append({'stem': stem, 'chain': ch, 'src': rec['src'],
                             'box': (b if box_src == 'label' else (person_box(rec, b) or b)),
                             'pts': pts, 'size': (iw, ih), 'human': False})
    if limit:
        rng = random.Random(seed)
        human = human[:limit]
        auto = rng.sample(auto, min(limit * 3, len(auto)))
    idx = list(range(len(human))) * max(1, human_repeat) + list(range(len(human), len(human) + len(auto)))
    samples = human + auto
    print('样本：人工 %d 个实例（重复 %d× ⇒ %d 样本） + 自动 %d 个实例 ⇒ 每 epoch %d 样本'
          % (len(human), human_repeat, len(human) * max(1, human_repeat), len(auto), len(idx)))
    return samples, idx, val


def warp_matrix(box, size=(IN_W, IN_H)):
    from transformers.models.vitpose.image_processing_vitpose import (
        box_to_center_and_scale, get_warp_matrix)
    center, scale = box_to_center_and_scale(box, image_width=size[0], image_height=size[1],
                                           normalize_factor=200.0)
    return get_warp_matrix(0.0, center * 2.0, np.array(size) - 1.0, scale * 200.0)


def make_target(pts):
    """真值点 → (heatmap[17,HM_H,HM_W], mask[17])；vis<1 的关节 mask=0（不产生梯度）。"""
    hm = np.zeros((17, HM_H, HM_W), dtype=np.float32)
    mask = np.zeros(17, dtype=np.float32)
    yy, xx = np.mgrid[0:HM_H, 0:HM_W]
    for i, nm in enumerate(COCO17_NAMES):
        if nm not in pts or pts[nm][2] < 1:
            continue
        x, y = pts[nm][0] / 4.0, pts[nm][1] / 4.0
        if not (0 <= x < HM_W and 0 <= y < HM_H):
            continue                                  # 裁图后跑到外面 ⇒ 当这张图没有这个关节
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0 ** 2))
        hm[i] = g / max(1e-6, g.max())
        mask[i] = 1.0
    return hm, mask


class Samples(torch.utils.data.Dataset):
    def __init__(self, samples, idx, proc, flip=0.5, jitter=0.18, rot=0.0, color=0.0, debug=False, train=True):
        self.s = samples
        self.idx = idx
        self.proc = proc            # 用 processor 裁图：与推理**逐像素一致**（cv2 在出界处填充方式不同，实测差 3.8）
        self.flip = flip
        self.jitter = jitter
        self.rot = rot              # 旋转增广（度）
        self.color = color          # 亮度/对比度抖动幅度
        self.debug = debug          # True 时额外返回 M 与原始关键点（只给几何自检用）
        self.train = train
        self.cache = {}

    def __len__(self):
        return len(self.idx)

    def _img(self, src):
        if src not in self.cache:
            from PIL import Image
            self.cache[src] = Image.open(src).convert('RGB')
        return self.cache[src]

    def __getitem__(self, k):
        ex = self.s[self.idx[k]]
        img = self._img(ex['src'])
        x0, y0, x1, y1 = ex['box']
        if self.train and self.jitter:
            w, h = x1 - x0, y1 - y0
            sx, sy = 1 + random.uniform(-self.jitter, self.jitter), 1 + random.uniform(-self.jitter, self.jitter)
            cx, cy = (x0 + x1) / 2 + random.uniform(-0.05, 0.05) * w, (y0 + y1) / 2 + random.uniform(-0.05, 0.05) * h
            x0, x1 = cx - w * sx / 2, cx + w * sx / 2
            y0, y1 = cy - h * sy / 2, cy + h * sy / 2
        box = [x0, y0, x1, y1]
        # 旋转增广：必须**先把原图关键点过一次 M 变成裁图坐标，再在裁图里翻转**。
        # 上一版把翻转放在旋转之前，等于对已经裁好的坐标又套了一遍 M ⇒ 目标全错（冒烟后 2 epoch 就 664 px）。
        rot = random.uniform(-self.rot, self.rot) if (self.train and self.rot) else 0.0
        if rot:
            from transformers.models.vitpose.image_processing_vitpose import (
                box_to_center_and_scale, get_warp_matrix)
            center, scale = box_to_center_and_scale(box, image_width=IN_W, image_height=IN_H,
                                                   normalize_factor=200.0)
            M = get_warp_matrix(rot, center * 2.0, np.array((IN_W, IN_H)) - 1.0, scale * 200.0)
            crop = self.proc.affine_transform(np.asarray(img), center, scale, rot,
                                              {'height': IN_H, 'width': IN_W},
                                              input_data_format='channels_last')
        else:
            M = warp_matrix(box)
            crop = np.asarray(self.proc(images=img, boxes=[[box]], return_tensors=None,
                                        do_normalize=False, do_rescale=False,
                                        data_format='channels_last')['pixel_values'][0])
        pts = {}
        for nm, (px, py, v) in ex['pts'].items():
            p = M @ np.array([px, py, 1.0])
            pts[nm] = (float(p[0]), float(p[1]), v)
        crop = np.asarray(crop)
        flipped = bool(self.train and self.flip and random.random() < self.flip)
        if flipped:
            crop = crop[:, ::-1]
            sw = {}
            for i, nm in enumerate(COCO17_NAMES):
                src_nm = COCO17_NAMES[COCO17_FLIP[i]]
                if src_nm in pts:
                    px, py, v = pts[src_nm]
                    sw[nm] = (IN_W - 1 - px, py, v)
            pts = sw
        if self.train and self.color:
            c0 = crop.astype(np.float32)
            c0 = c0 / 255.0 if c0.max() > 1.5 else c0          # scipy 路径返回 0-255，processor 路径已除过
            b = 1 + random.uniform(-self.color, self.color)
            c = random.uniform(-self.color, self.color)
            c0 = np.clip(c0 * b + c, 0, 1)
            if random.random() < 0.3:                          # 轻度对比度
                m = c0.mean()
                c0 = np.clip((c0 - m) * (1 + random.uniform(-0.15, 0.15)) + m, 0, 1)
            crop = c0
        f = np.asarray(crop, dtype=np.float32)
        f = f / 255.0 if f.max() > 1.5 else f
        x = torch.from_numpy(np.ascontiguousarray(f.transpose(2, 0, 1)))
        x = (x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        hm, mask = make_target(pts)
        if self.debug:
            return x, torch.from_numpy(hm), torch.from_numpy(mask), M, dict(ex['pts']), flipped
        return x, torch.from_numpy(hm), torch.from_numpy(mask)


def evaluate(model, proc, val_items, device):
    """留出集：M2a 人框推理 → 与人工真值比关节像素误差；解码用 processor 自带的 DARK 亚像素细化，
    与 eval_common_subset.py 的口径一致（否则训练期看到的数会比真实差一截）。"""
    from transformers.models.vitpose.image_processing_vitpose import box_to_center_and_scale
    errs = []
    model.eval()
    with torch.no_grad():
        for ex in val_items:
            img = __import__('PIL.Image', fromlist=['Image']).open(ex['src']).convert('RGB')
            box = list(ex['box'])
            x = proc(images=img, boxes=[[box]], return_tensors='pt')['pixel_values'].to(device)
            hm = model(pixel_values=x).heatmaps.float()
            center, scale = box_to_center_and_scale(box, image_width=IN_W, image_height=IN_H,
                                                   normalize_factor=200.0)
            preds, _scores = proc.keypoints_from_heatmaps(hm.cpu().numpy(),
                                                          center=np.array([center]), scale=np.array([scale]))
            for i, nm in enumerate(COCO17_NAMES):
                if nm not in ex['pts'] or ex['pts'][nm][2] < 1:
                    continue
                gx, gy, _v = ex['pts'][nm]
                errs.append(float(np.hypot(gx - preds[0][i][0], gy - preds[0][i][1])))
    model.train()
    errs = np.array(errs) if errs else np.array([np.nan])
    return float(np.median(errs)), float(errs.mean()), len(errs)


def items_of(stems):
    """把一批 stem 变成评测项（每个实例一条：M2a 人框 + 真值点）。"""
    out = []
    for stem in stems:
        gp = os.path.join(GT, stem + '.json')
        if not os.path.exists(gp):
            continue
        g, rec = jload(gp), jload(os.path.join(PRE, stem + '.json'))
        iw, ih = [int(v) for v in rec['size'].split('x')]
        for ch, pts in norm_pts(g).items():
            b = box_of(pts, iw=iw, ih=ih)
            if not b:
                continue
            out.append({'stem': stem, 'chain': ch, 'src': rec['src'],
                        'box': person_box(rec, b) or b, 'pts': pts, 'size': (iw, ih)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--human-repeat', type=int, default=5)
    ap.add_argument('--no-auto', action='store_true')
    ap.add_argument('--lr-head', type=float, default=3e-4)
    ap.add_argument('--lr-backbone', type=float, default=2e-5)
    ap.add_argument('--freeze-backbone', action='store_true')
    ap.add_argument('--face-weight', type=float, default=3.0)
    ap.add_argument('--val-every', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0, help='冒烟用：只用这么多人工实例')
    ap.add_argument('--select-count', type=int, default=15, help='从人工训练图里切多少张用于选模型')
    ap.add_argument('--auto-dir', default=None,
                    help='自动池标签目录（默认 data/prelabel_face_full；自训练时换成刷新后的那份）')
    ap.add_argument('--box-src', choices=['person', 'label'], default='person',
                    help='训练裁图框：person=M2a 人框（宽松，与推理一致）；label=标签自身骨架框（更紧）')
    ap.add_argument('--joint-weight-hard', type=float, default=1.0,
                    help='难关节（肘腕膝踝髋）损失权重倍数')
    HARD_IDX = [7, 8, 9, 10, 13, 14, 15, 16, 11, 12]     # 肘/腕/膝/踝/髋（COCO17 下标）
    ap.add_argument('--rot', type=float, default=12.0, help='旋转增广幅度（度）')
    ap.add_argument('--color', type=float, default=0.10, help='亮度抖动幅度')
    ap.add_argument('--name', default=None)
    ap.add_argument('--seed', type=int, default=20261013)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    name = a.name or ('vitpose_ft_%s' % time.strftime('%m%d_%H%M'))
    print('=== 微调 ViTPose-B：%s ===' % name)

    # 选模型集：从"留出集之外的 104 张人工图"里再切一批出来，**模型选择只看它**，
    # 原始 15 张留出集只用于最终报告（否则用报告集选 checkpoint，报出来的数会偏乐观）。
    man0 = jload(os.path.join(DATA, 'ds_pose17_semi', 'semi_manifest.json'))
    pool = sorted(os.path.splitext(f)[0] for f in os.listdir(GT) if f.endswith('.json')
                  and os.path.splitext(f)[0] not in set(man0['val_stems']))
    rng_sel = random.Random(a.seed + 1)
    rng_sel.shuffle(pool)
    select_stems = set(pool[:a.select_count])
    print('人工真值 %d 张：留出 %d（报告） / 选模型 %d / 训练 %d'
          % (len(pool) + len(man0['val_stems']), len(man0['val_stems']), len(select_stems),
             len(pool) - len(select_stems)))

    samples, idx, val_stems = build_samples(a.human_repeat, not a.no_auto, a.limit, seed=a.seed,
                                           drop_stems=select_stems, box_src=a.box_src,
                                           auto_dir=a.auto_dir)
    val_items = items_of(sorted(val_stems))
    select_items = items_of(sorted(select_stems))
    print('留出集实例 %d 个（%d 张图，**只用于最终报告**）；选模型集实例 %d 个（%d 张图，从其余人工图里切）'
          % (len(val_items), len(val_stems), len(select_items), len(select_stems)))

    from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
    proc = VitPoseImageProcessor.from_pretrained(VITPOSE_DIR)
    model = VitPoseForPoseEstimation.from_pretrained(VITPOSE_DIR)
    if a.freeze_backbone:
        for p in model.vitpose.parameters():
            p.requires_grad = False
        print('backbone 冻结，只训解码头')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).train()

    head_params = [p for n, p in model.named_parameters() if n.startswith('vitpose.') is False and p.requires_grad]
    bb_params = [p for n, p in model.named_parameters() if n.startswith('vitpose.') and p.requires_grad]
    groups = []
    if head_params:
        groups.append({'params': head_params, 'lr': a.lr_head})
    if bb_params and a.lr_backbone > 0:
        groups.append({'params': bb_params, 'lr': a.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g['lr'] for g in groups], total_steps=max(1, a.epochs * (len(idx) // a.batch + 1)),
        pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    dl = torch.utils.data.DataLoader(Samples(samples, idx, proc, rot=a.rot, color=a.color), batch_size=a.batch, shuffle=True,
                                     num_workers=0, drop_last=False)
    w = torch.ones(17, device=device)
    w[FACE_IDX] = a.face_weight
    if a.joint_weight_hard != 1.0:
        for _i in HARD_IDX:
            w[_i] *= a.joint_weight_hard
    print('  关节权重：面点 ×%.1f，难关节 ×%.1f' % (a.face_weight, a.joint_weight_hard))
    out_dir = os.path.join(PROJ, 'models', name)
    os.makedirs(out_dir, exist_ok=True)
    log = io.open(os.path.join(out_dir, 'train_log.csv'), 'w', encoding='utf-8', newline='\n')
    log.write('epoch,loss,val_median_px,val_mean_px,n_joints\n')
    best = 1e9
    vm0, vmean0, nj0 = evaluate(model, proc, select_items, device)
    print('训练前（零微调）选模型集中位 %.1f px / 平均 %.1f px（%d 关节）' % (vm0, vmean0, nj0), flush=True)
    log.write('0,0,%.3f,%.3f,%d\n' % (vm0, vmean0, nj0))
    for ep in range(1, a.epochs + 1):
        tot, nb = 0.0, 0
        for x, hm, mask in dl:
            x, hm, mask = x.to(device), hm.to(device), mask.to(device)
            with torch.autocast('cuda', enabled=(device == 'cuda')):
                out = model(pixel_values=x).heatmaps
                mse = F.mse_loss(out.float(), hm, reduction='none').mean(dim=(2, 3))   # (B,17)
                loss = (mse * mask * w[None]).sum() / (mask * w[None]).sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        msg = 'epoch %3d/%d  loss %.4f' % (ep, a.epochs, tot / max(1, nb))
        vm = vmean = float('nan')
        nj = 0
        if ep % a.val_every == 0 or ep == a.epochs:
            vm, vmean, nj = evaluate(model, proc, select_items, device)
            msg += '  选模型集中位 %.1f px / 平均 %.1f px（%d 关节）' % (vm, vmean, nj)
            if vm < best:
                best = vm
                model.save_pretrained(out_dir)
                proc.save_pretrained(out_dir)      # 否则 from_pretrained(dir) 加载不了（缺 preprocessor_config）
                io.open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8', newline='\n').write(
                    json.dumps({'epoch': ep, 'val_median_px': vm, 'val_mean_px': vmean, 'args': vars(a)},
                               ensure_ascii=False, indent=1))
                msg += '  ← 存盘'
        print(msg, flush=True)
        log.write('%d,%.5f,%.3f,%.3f,%d\n' % (ep, tot / max(1, nb), vm, vmean, nj))
        log.flush()
    log.close()
    rv, rmean, rn = evaluate(model, proc, val_items, device)
    try:                                    # 把报告集数字补进 meta.json，便于汇总脚本自动出表
        _mp = os.path.join(out_dir, 'meta.json')
        _m = json.load(io.open(_mp, encoding='utf-8')) if os.path.exists(_mp) else {}
        _m['report_median_px'] = rv
        _m['report_mean_px'] = rmean
        _m['report_n_joints'] = rn
        io.open(_mp, 'w', encoding='utf-8', newline='\n').write(json.dumps(_m, ensure_ascii=False, indent=1))
    except Exception as _e:
        print('  ⚠ 报告集数字写入 meta 失败：%r' % (_e,))
    print('=== 最终报告（15 张留出集，选模型集没碰过它）：中位 %.1f px / 平均 %.1f px（%d 关节）' % (rv, rmean, rn))
    print('最好选模型集中位：%.1f px → %s' % (best, out_dir))
    return 0


if __name__ == '__main__':
    sys.exit(main())
