"""U-Net 版绳子 kernel：大感受野 + 跳跃连接，用"绳索宏观走向"这类上下文信息判别。

和 rope_train.py（k×k 邻域 MLP）的区别：MLP 只看局部 k×k（k=9 时 9px），U-Net 下采样两次后
有效感受野 ~100px，能利用"这根带子是不是沿着绳子的走向连续延伸"。

协议与 MLP 完全一致，保证可比：
  * 按整图 leave-one-image-out（训练图与测试图不重叠）；
  * 阈值只在**训练图的验证块**上选（不碰测试图）；
  * 指标交给 scripts/rope_eval.py 统一算（P/R/F1 + 误检块 + 漏检段 + 预计笔数）。
用 me2 环境（torch+CUDA）；显存按 6GB 共享的口径压小：base=24、256 瓦片、batch 6。

用法：
  python scripts/rope_unet.py --gt data/masks/rope/raw_ps \
      --out .runtime/rope_cv/unet_loo --weights data/masks/rope/weights
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
IMG_DIR = os.path.join(ROOT, '.runtime', 'rope_cv', 'ps_work', '_orig')


def find_image(name):
    for e in ('.png', '.jpg', '.jpeg', '.webp'):
        p = os.path.join(IMG_DIR, name + e)
        if os.path.exists(p):
            return p
    g = sorted(glob.glob(os.path.join(IMG_DIR, name + '_*')))
    return g[0] if g else None


def load_items(gt):
    items = []
    for lp in sorted(glob.glob(os.path.join(gt, '*_labels.png'))):
        name = os.path.basename(lp)[:-len('_labels.png')]
        ip = find_image(name)
        if ip is None:
            continue
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        lab = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        if lab.shape != img.shape[:2]:
            lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        items.append({'name': name, 'img': img, 'hsv': hsv, 'lab': lab,
                      'w': cost_map(lab)})
    return items


def cost_map(lab, edge_w0=0.2, d_ref=6.0, neg_band=2, neg_band_w=0.3):
    """按"到绳子边界的距离"给正/负样本加权（用户提的 cost map）。

    edge_w0：边界像素保留的权重（0 = 完全不管边缘，交给后处理膨胀去覆盖）
    d_ref  ：多远算"中心"（px）；绳子半宽量级即可
    neg_band / neg_band_w：绳边界两侧这么宽的负样本权重降到 neg_band_w（容忍标注错位）
    """
    rope = (lab == 1).astype(np.uint8)
    neg = (lab == 2).astype(np.uint8)
    if rope.sum() == 0:
        return np.ones_like(lab, np.float32)
    dt_in = cv2.distanceTransform(rope, cv2.DIST_L2, 5)
    w_pos = np.clip(dt_in / max(1e-6, d_ref), 0, 1)
    w_pos = edge_w0 + (1.0 - edge_w0) * w_pos
    inv = (1 - rope).astype(np.uint8)
    dt_out = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    w_neg = np.ones_like(w_pos)
    if neg_band > 0:
        near = (dt_out <= neg_band) & (neg > 0)
        w_neg[near] = neg_band_w
    w = np.where(rope > 0, w_pos, w_neg)
    return w.astype(np.float32)


def crop(img, lab, y, x, s):
    """返回 (s,s,3) 的图和 (s,s) 的标签——保持 HWC，转置留到 stack 之后统一做。"""
    h, w = lab.shape
    y = max(0, min(h - s, y))
    x = max(0, min(w - s, x))
    return img[y:y + s, x:x + s], lab[y:y + s, x:x + s]


class UNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        b = base

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.e1 = blk(3, b)
        self.e2 = blk(b, b * 2)
        self.e3 = blk(b * 2, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d2 = blk(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d1 = blk(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(F.max_pool2d(s1, 2))
        s3 = self.e3(F.max_pool2d(s2, 2))
        y = self.d2(torch.cat([self.u2(s3), s2], 1))
        y = self.d1(torch.cat([self.u1(y), s1], 1))
        return self.head(y)


def widen_first_conv(conv, in_ch):
    """把预训练骨干的第一层从 3 通道扩到 in_ch（HSV+RGB 六通道用）。

    两组通道都拿 RGB 预训练权重 ×0.5：对称初始化，激活量级与单支路相当，
    网络自己决定更信哪一支；只拷一路、另一路置零会让对照偏向被拷贝的那一支。
    """
    out_ch = conv.out_channels
    new = nn.Conv2d(in_ch, out_ch, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, bias=conv.bias is not None)
    with torch.no_grad():
        w = conv.weight.data
        for g in range(in_ch // 3):
            new.weight.data[:, 3 * g:3 * g + 3] = w * (3.0 / in_ch)
        if conv.bias is not None:
            new.bias.data.copy_(conv.bias.data)
    return new



class _ViTSeq(nn.Module):
    """torchvision ViT：预处理成 token 序列 → 逐层 encoder block（供 PreEnc 挂 hook 取中间特征）。"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        m = self.m
        # 自己做 patch embedding，绕开 torchvision `_process_input` 里"必须 224"的 assert
        z = m.conv_proj(x).flatten(2).transpose(1, 2)          # (B,N,C)
        b, n, c = z.shape
        pos = m.encoder.pos_embedding
        if pos.shape[1] != n + 1:                              # 位置编码插值到目标网格
            g_old = int(round((pos.shape[1] - 1) ** 0.5))
            g_new = int(round(n ** 0.5))
            cls_pe, grid = pos[:, :1], pos[:, 1:]
            grid = grid.reshape(1, g_old, g_old, c).permute(0, 3, 1, 2)
            grid = F.interpolate(grid, size=(g_new, g_new), mode='bilinear', align_corners=False)
            grid = grid.permute(0, 2, 3, 1).reshape(1, g_new * g_new, c)
            pos = torch.cat([cls_pe, grid], 1)
        z = torch.cat([m.class_token.expand(b, -1, -1), z], 1) + pos
        z = m.encoder.dropout(z)
        for layer in m.encoder.layers:
            z = layer(z)
        return z


class PreEnc(nn.Module):
    """预训练骨干 → 返回 stride 8/16/32 三档特征（可选带 stride 4）。

    为什么不用整套 YOLO-seg：检测头/锚点分配/实例解码对"单类单 mask"是纯负担；
    真正有迁移价值的是 backbone 的通用视觉特征。所以只借骨干，解码器自己写（很薄）。
    """

    def __init__(self, kind='scratch', in_ch=3):
        super().__init__()
        self.kind = kind
        self.in_ch = in_ch
        self.taps = None
        if kind == 'scratch':
            raise ValueError('scratch 不走这条路径')
        if kind in ('resnet18', 'resnet34'):
            import torchvision
            w = getattr(torchvision.models, kind.title().replace('18', '18')
                        if False else kind).__name__
            m = getattr(torchvision.models, kind)(weights='IMAGENET1K_V1')
            if in_ch != 3:
                m.conv1 = widen_first_conv(m.conv1, in_ch)
            # resnet 的 layer2/3/4 正好是 stride 8/16/32
            self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1,
                                      m.layer2, m.layer3, m.layer4)
            self.layers = list(self.body)
            self.tap_idx = [5, 6, 7]          # layer2/3/4（stride 8/16/32；层0..7 共 8 个）
        elif kind == 'mobilenet_v3_small':
            import torchvision
            m = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
            if in_ch != 3:
                m.features[0][0] = widen_first_conv(m.features[0][0], in_ch)
            self.body = m.features
            self.layers = list(self.body)
            self.tap_idx = None               # 按分辨率自动找
        elif kind == 'swin_t':
            import torchvision
            m = torchvision.models.swin_t(weights='IMAGENET1K_V1')
            if in_ch != 3:
                m.features[0][0] = widen_first_conv(m.features[0][0], in_ch)
            self.body = m.features
            self.layers = list(m.features)
            self.tap_idx = None               # 分层特征，按分辨率自动找（输出是 NHWC，读取时转换）
            self.patch = 4
        elif kind == 'vit_b_16':
            import torchvision
            m = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
            if in_ch != 3:
                raise ValueError('vit_b_16 不支持多通道输入')
            self.body = _ViTSeq(m)             # 预处理 + 逐层 encoder block（token 序列）
            self.layers = list(m.encoder.layers)
            # 单尺度（stride 16）：取最后三层当三档"伪尺度"，解码器照样能上采样相加
            self.tap_idx = [9, 10, 11]
            self.patch = 16
        elif kind == 'yolo11n_seg':
            if in_ch != 3:
                raise ValueError('yolo11n_seg 骨干暂不支持多通道输入（第一层被 ultralytics 多处引用）')
            from ultralytics import YOLO
            import os
            ck = os.path.join(r'<你的 ComfyUI 安装目录>', 'models', 'yolo11n-seg.pt')
            y = YOLO(ck if os.path.exists(ck) else 'yolo11n-seg.pt')
            self.body = y.model                # BaseModel：负责 list 管道（逐层直接调会炸）
            # ultralytics 载入的权重默认 requires_grad=False —— 不显式打开的话
            # `--freeze-enc 0` 只是"看起来没冻结"（实测可训 0.22M 而非 3.1M，
            # 于是 unfrozen 与 frozen 两行数字一模一样）。
            for p in self.body.parameters():
                p.requires_grad_(True)
            self.layers = list(y.model.model)  # 真正的层列表，只用来挂 hook / 索引
            self.tap_idx = None
        else:
            raise ValueError('未知 encoder: %s' % kind)
        self._feats = {}


    def _to_nchw(self, t):
        """把不同主干的特征统一成 NCHW：Swin 是 NHWC；ViT 是 (B,N,C) token 序列。"""
        if not torch.is_tensor(t):
            return None
        if t.dim() == 3:                      # (B, N, C) → 去掉 CLS，reshape 回二维
            p = getattr(self, 'patch', 16)
            h, w = getattr(self, '_in_hw', (0, 0))
            hh, ww = h // p, w // p
            t = t[:, 1:] if t.shape[1] == hh * ww + 1 else t
            if hh * ww != t.shape[1]:
                return None
            return t.reshape(t.shape[0], hh, ww, t.shape[2]).permute(0, 3, 1, 2).contiguous()
        # Swin 的 features 恒为 NHWC：**必须按主干判断**，不能靠"最后一维更大"的启发式——
        # 512 瓦片时 stride-4 特征图是 128×128×96，空间尺寸比通道数还大，启发式会漏判（C1 第一次就栽在这）。
        if getattr(self, 'kind', '') == 'swin_t' and t.dim() == 4:
            return t.permute(0, 3, 1, 2).contiguous()
        if t.dim() == 4 and t.shape[-1] > t.shape[1]:      # 其它主干的兜底启发式
            return t.permute(0, 3, 1, 2).contiguous()
        return t

    def _find_taps(self, h, w):
        """跑一次 dummy 前向，按输出分辨率挑出 H/8,H/16,H/32 三档（取通道数最多的那层）。"""
        self._feats = {}
        self._feats_size = {}
        hs = []
        for i, layer in enumerate(self.layers):
            hs.append(layer.register_forward_hook(
                lambda mod, inp, out, idx=i: self._feats.__setitem__(idx, out)))
        self._in_hw = (h, w)
        with torch.no_grad():
            dev = next(self.parameters()).device
            _ = self.body(torch.zeros(1, self.in_ch, h, w, device=dev))
        for hk in hs:
            hk.remove()
        # 不看 stride，只看"分辨率最大的三档"（同尺寸取通道最多的层），从深到浅排序
        cand = {}
        for i, t in self._feats.items():
            t = self._to_nchw(t)
            if isinstance(t, (list, tuple)):
                t = next((z for z in t if torch.is_tensor(z) and z.dim() == 4), None)
            if t is None or not torch.is_tensor(t) or t.dim() != 4:
                continue
            key = int(t.shape[-2])
            if key <= 4:                                  # 太深的 1/32 以下不用
                continue
            if key not in cand or t.shape[1] > self._feats_size.get(key, -1):
                cand[key] = i
                self._feats_size[key] = int(t.shape[1])
        sizes = sorted(cand.keys(), reverse=True)[:3]     # 分辨率从大到小
        if len(sizes) < 2:
            raise RuntimeError('可用特征层不足：%s' % sizes)
        return [cand[k] for k in sizes]                   # [最浅, ..., 最深]（尺寸降序）

    def forward(self, x):
        self._in_hw = (x.shape[-2], x.shape[-1])
        if self.tap_idx is None:
            h, w = max(32, x.shape[-2] // 32 * 32), max(32, x.shape[-1] // 32 * 32)
            self.tap_idx = self._find_taps(h, w)
        if self.kind in ('yolo11n_seg', 'vit_b_16'):
            # ultralytics 的 layer 是"整段 list 进 list 出"，逐层喂单个张量会炸（cat 需要 tuple）
            feats, hs = {}, []
            for i in self.tap_idx:
                hs.append(self.layers[i].register_forward_hook(
                    lambda mod, inp, out, idx=i: feats.__setitem__(idx, out)))
            _ = self.body(x)
            for hk in hs:
                hk.remove()
            out = []
            for i in self.tap_idx:
                t = feats[i]
                if isinstance(t, (list, tuple)):
                    t = next(z for z in t if torch.is_tensor(z))
                t = self._to_nchw(t)          # ViT 的 token 序列也在这里转成 NCHW
                out.append(t)
            return out
        feats = {}
        for i, layer in enumerate(self.body):
            x = layer(x)
            if i in self.tap_idx:
                feats[i] = x if torch.is_tensor(x) else x[0]
        return [self._to_nchw(feats[i]) for i in self.tap_idx]


class UNetPre(nn.Module):
    """预训练骨干 + 超薄解码器（1x1 对齐通道 → 逐级上采样相加 → 8x 上采样出 logits）。"""

    def __init__(self, base=24, kind='resnet18', mid=128, in_ch=3):
        super().__init__()
        self.enc = PreEnc(kind, in_ch=in_ch)
        with torch.no_grad():                 # 探一次通道数
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.enc.to(dev)
            ps = 224 if kind == 'vit_b_16' else 256   # ViT-B/16 的位置编码固定 224，探针也用 224
            cs = [f.shape[1] for f in self.enc(torch.zeros(1, in_ch, ps, ps, device=dev))]
            self.enc.to('cpu')
        self.n_tap = len(cs)
        self.proj = nn.ModuleList([nn.Conv2d(c, mid, 1) for c in cs])
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                         nn.ReLU())
        # 各档都被 proj 投到 mid 通道，逐级"上采样相加"后仍是 mid，所以 d2 是 mid→mid
        self.d2 = blk(mid, mid)
        self.d1 = blk(mid, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        f = self.enc(x)                                   # 尺寸降序：[浅, ..., 深]
        p = [self.proj[i](f[i]) for i in range(len(f))]
        y = p[-1]
        for j in range(len(p) - 2, -1, -1):               # 由深到浅逐级上采样相加
            y = F.interpolate(y, size=p[j].shape[-2:], mode='nearest') + p[j]
            if j > 0:
                y = self.d2(y)
        y = self.d1(y)
        # 统一插值回输入尺寸（不假设 8 倍，因为不同骨干/neck 的分辨率不一样）
        y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(y)


def augment_tile(a, b, rng, blur=(0.0, 2.0), p_blur=0.3, color=True, c=None, rot=0.0):
    """瓦片级增强：90° 旋转 / 翻转 / 任意角旋转 / 高斯模糊 / 有界颜色抖动。
    a 是 (s,s,3) HSV，b 是 (s,s) 标签，c 是可选的同形状权重图（cost map）——**几何变换必须一起做**，
    否则权重会贴到错误的像素上（旧代码就是这样：翻转/旋转后 cost map 一直是错位的）。

    任意角旋转在 **BGR** 里做再转回 HSV：直接插值 H 通道会在红色 0/1 环绕处串色（红↔青）。
    标签的边框用 0（未标注）填充，于是旋转带进来的反射内容不参与损失；图像用 BORDER_REFLECT 避免黑角。
    始终返回 (a, b, c)，c 可能为 None。
    """
    k = rng.randint(4)
    if k:
        a, b = np.rot90(a, k), np.rot90(b, k)
        if c is not None:
            c = np.rot90(c, k)
    if rng.rand() < 0.5:
        a, b = a[:, ::-1], b[:, ::-1]
        if c is not None:
            c = c[:, ::-1]
    if rng.rand() < 0.5:
        a, b = a[::-1], b[::-1]
        if c is not None:
            c = c[::-1]
    if rot and rot > 0:
        ang = rng.uniform(-rot, rot)
        h, w = a.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        bgr = cv2.cvtColor((np.clip(a, 0, 1) * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        bgr = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        a = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        b = cv2.warpAffine(b, M, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if c is not None:
            c = cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if blur and rng.rand() < p_blur:
        sig = rng.uniform(*blur)
        if sig > 0.05:
            bgr = cv2.cvtColor((np.clip(a, 0, 1) * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
            bgr = cv2.GaussianBlur(bgr, (0, 0), sig)
            a = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    if color:
        a = a.copy()
        if rng.rand() < 0.4:                                  # 亮度：gamma，恒 ≤1
            a[..., 2] = a[..., 2] ** (1.0 / rng.uniform(0.85, 1.18))
        if rng.rand() < 0.4:                                  # 饱和度：有界插值
            if rng.rand() < 0.5:
                a[..., 1] = a[..., 1] + (1.0 - a[..., 1]) * rng.uniform(0.0, 0.15)
            else:
                a[..., 1] = a[..., 1] * (1.0 - rng.uniform(0.0, 0.2))
        if rng.rand() < 0.25:                                 # 色相：环形 ±1.4（OpenCV 量程）
            a[..., 0] = (a[..., 0] + rng.uniform(-0.008, 0.008)) % 1.0
    if c is None:
        return np.ascontiguousarray(a), np.ascontiguousarray(b), None
    return np.ascontiguousarray(a), np.ascontiguousarray(b), np.ascontiguousarray(c)


def paste_patch(a, b, donors, rng, frac=(0.30, 0.55), ring=6):
    """Copy-Paste：从别的图里剪一块**含绳**的真实图块，羽化后贴进当前瓦片（图与标签一起贴）。

    目的正是补稀缺外观（白绳/棕绳/暗场景）——它把"真实绳的纹理"搬到别的背景上，
    比颜色抖动更接近真实分布。边界用羽化 alpha（在 BGR 里混，避免 H 通道串色）削弱接缝，
    标签在羽化环内按"给体标签"整块贴入（环外保持原标签），所以监督信号自洽。
    """
    if not donors:
        return a, b
    cand = [d for d in donors if (d['lab'] == 1).sum() > 400]
    if not cand:
        return a, b
    d = cand[rng.randint(len(cand))]
    yy, xx = np.where(d['lab'] == 1)
    i = rng.randint(len(yy))
    ph = int(a.shape[0] * rng.uniform(*frac))
    pw = int(a.shape[1] * rng.uniform(*frac))
    sy, sx = int(yy[i] - ph // 2 + rng.randint(-40, 41)), int(xx[i] - pw // 2 + rng.randint(-40, 41))
    pa = np.zeros((ph, pw, 3), np.float32)
    pb = np.zeros((ph, pw), np.uint8)
    y0, x0 = max(0, sy), max(0, sx)
    y1, x1 = min(d['lab'].shape[0], sy + ph), min(d['lab'].shape[1], sx + pw)
    if y1 - y0 < 32 or x1 - x0 < 32:
        return a, b
    pa[y0 - sy:y0 - sy + (y1 - y0), x0 - sx:x0 - sx + (x1 - x0)] = d['hsv'][y0:y1, x0:x1]
    pb[y0 - sy:y0 - sy + (y1 - y0), x0 - sx:x0 - sx + (x1 - x0)] = d['lab'][y0:y1, x0:x1]
    if (pb == 1).sum() < 200:
        return a, b
    dy = rng.randint(0, max(1, a.shape[0] - ph + 1))
    dx = rng.randint(0, max(1, a.shape[1] - pw + 1))
    al = np.zeros((ph, pw), np.float32)
    r = min(ring, ph // 4, pw // 4)
    al[r:ph - r, r:pw - r] = 1.0
    al = cv2.GaussianBlur(al, (0, 0), max(1.0, r / 2.0))
    A = cv2.cvtColor((np.clip(a, 0, 1) * 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    P = cv2.cvtColor((np.clip(pa, 0, 1) * 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    reg = A[dy:dy + ph, dx:dx + pw]
    A[dy:dy + ph, dx:dx + pw] = reg * (1 - al[..., None]) + P * al[..., None]
    a = cv2.cvtColor(A.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    core = al > 0.5
    regb = b[dy:dy + ph, dx:dx + pw]
    regb[pb == 1] = 1
    regb[(pb == 2) & core] = 2
    b[dy:dy + ph, dx:dx + pw] = regb
    return a, b


def to_input(a, in_mode):
    """HSV(0~1，H 已按 /255 压到 0~0.70) → 喂网络的表示。

    rgb   ：先回到 8bit HSV 再转 BGR→RGB。骨干第一层是 RGB 预训练的，
            这条支路用来验证"喂 HSV 是否浪费了预训练先验"。
    hsvrgb：6 通道 = cat(HSV, RGB)，第一层已按两半各半的权重重初始化（见 widen_first_conv）。
    **增强一律留在 HSV 空间做**（色相环形平移、亮度 gamma、饱和度有界插值都是 HSV 语义），
    只在最后一步换表示——这样三路对照不会混进增强强度的差异。
    """
    if in_mode == 'hsv':
        return a
    u8 = cv2.cvtColor((np.clip(a, 0, 1) * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    rgb = u8[:, :, ::-1].astype(np.float32) / 255.0
    if in_mode == 'rgb':
        return rgb
    if in_mode == 'hsvrgb':
        return np.concatenate([a, rgb], axis=2)
    raise ValueError('未知 in_mode: %s' % in_mode)


def sample_batch(items, s, bs, rng, aug=False, zoom=(0.75, 1.33), blur=(0.0, 2.0), p_blur=0.3, hard=None, hard_frac=0.3, weights=False, in_mode='hsv', rot=0.0, cp=0.0, donors=None):
    """一半瓦片以绳像素为中心（保证正样本比例），一半随机。

    aug=True：翻转 + 尺度变焦（多尺度）+ 90° 旋转（+ 可选 --rot 任意角）+ 高斯模糊 + 有界颜色抖动。
    颜色抖动的边界问题（用户提过"上溢下溢会毁数据"）：亮度用幂函数 gamma（a∈[0,1] 恒不越界）、
    饱和度用向灰/向纯的**有界插值**、色相 ±0.008 环形平移——数学上没有出界可能。
    weights=True 时返回的权重图会**跟着做同样的几何变换**（之前没做，cost map 一直错位）。
    cp>0 时按概率做 Copy-Paste（从 donors 里剪含绳的真实图块贴进来）。
    """
    xs, ys, ws = [], [], []
    for _ in range(bs):
        if hard and len(hard) and rng.rand() < hard_frac:
            # 难负样本挖掘：从当前模型最自信的误检位置里取裁剪，直接给决策边界喂反例
            it, hy, hx = hard[rng.randint(len(hard))]
            h, w = it['lab'].shape
            s0 = min(s, h, w)
            a, b = crop(it['hsv'], it['lab'], hy - s0 // 2, hx - s0 // 2, s0)
            c = crop(it['hsv'], it.get('w', it['lab']), hy - s0 // 2, hx - s0 // 2, s0)[1]
            if s0 != s:
                a = cv2.resize(a, (s, s), interpolation=cv2.INTER_LINEAR)
                b = cv2.resize(b, (s, s), interpolation=cv2.INTER_NEAREST)
                c = cv2.resize(c, (s, s), interpolation=cv2.INTER_LINEAR)
            if aug:
                a, b, c = augment_tile(a, b, rng, blur=blur, p_blur=p_blur, c=c, rot=rot)
                if cp and rng.rand() < cp:
                    a, b = paste_patch(a, b, donors, rng)
            xs.append(to_input(a, in_mode)); ys.append(b)
            ws.append(c if c is not None else np.ones((s, s), np.float32))
            continue
        it = items[rng.randint(len(items))]
        h, w = it['lab'].shape
        z = rng.uniform(*zoom) if aug else 1.0                # 尺度：先取 s*z 大小再缩放回 s
        s0 = int(min(max(24, round(s * z)), h, w))
        if rng.rand() < 0.5:
            yy, xx = np.where(it['lab'] == 1)
            if len(yy):
                i = rng.randint(len(yy))
                y0, x0 = yy[i] - s0 // 2, xx[i] - s0 // 2
            else:
                y0, x0 = rng.randint(h), rng.randint(w)
        else:
            y0, x0 = rng.randint(h), rng.randint(w)
        a, b = crop(it['hsv'], it['lab'], y0, x0, s0)
        c = crop(it['hsv'], it.get('w', it['lab']), y0, x0, s0)[1]
        if s0 != s:
            a = cv2.resize(a, (s, s), interpolation=cv2.INTER_LINEAR)
            b = cv2.resize(b, (s, s), interpolation=cv2.INTER_NEAREST)
            c = cv2.resize(c, (s, s), interpolation=cv2.INTER_LINEAR)
        if aug:
            a, b, c = augment_tile(a, b, rng, blur=blur, p_blur=p_blur, c=c, rot=rot)
            if cp and rng.rand() < cp:
                a, b = paste_patch(a, b, donors, rng)
        xs.append(to_input(a, in_mode))
        ys.append(b)
        ws.append(c if c is not None else np.ones((s, s), np.float32))
    # 注意要保持 (N,H,W,3) 到最后一步才转成 (N,3,H,W)：之前 crop 里就转置过，
    # 结果 `a[:, ::-1]` 翻的是**通道轴**（H/S/V 被搅乱），增强把数据毁了。
    X = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous().float()
    Y = torch.from_numpy(np.stack(ys)).long()
    if weights:
        W = torch.from_numpy(np.stack(ws)).float()
        return X, Y, W
    return X, Y


def soft_erode(img):
    """各向异性腐蚀：3×1 与 1×3 各腐蚀一次再取 min。

    这是 clDice 的关键细节——用各向同性 3×3 腐蚀时，"开运算"会把 1~2px 细线原样留下，
    于是 skel=relu(img-open(img))≡0，loss 恒为 1（等于这条损失白加）。分开方向后细线才留脊。
    """
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, 3, 1, 1)


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iters=8):
    """可微骨架（clDice 原版）：反复"腐蚀 → 开运算 → 取差"累计中心线。"""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def opening_loss(prob, weight=None):
    """连贯性损失：L = 被 3x3 开运算抹掉的概率质量。

    对应用户说的"要估分布、别只看单个像素"：孤立像素/毛刺在开运算下会消失，于是被惩罚；
    而绳子有 10~40px 宽，开运算后仍在，不受影响。比 TV/边界长度更直接——它不是泛泛压边界，
    而是明说"别留下连不成条的碎片"。
    """
    rem = F.relu(prob - soft_open(prob))
    if weight is not None:
        rem = rem * weight
    return rem.mean()


def cldice_loss(prob, target, iters=16, eps=1e-6):
    """clDice：中心线 Dice（Shit et al. 2021）。细管状结构（绳/血管）**拓扑保持**的专用损失，
    直接惩罚"预测断成好几段/中心线不重合"——正对我们最头疼的断续问题。
    """
    # 先做一次均值池化当软化：硬二值 GT 直接做形态学开运算时，细线会被整条消掉
    # （skel 全 0 → clDice 恒为 0，loss 恒为 1，这条损失就白加了）。软化后细结构也有脊。
    prob_s = F.avg_pool2d(prob, 3, 1, 1)
    target_s = F.avg_pool2d(target, 3, 1, 1)
    sp = soft_skel(prob_s, iters)
    st = soft_skel(target_s, iters)
    # 骨架提不出来（GT 太粗/太空，或 rope 宽超过 2*iters）时这条损失不适用：
    # 返回可微的 0，而不是恒 1（恒 1 会给出与目标无关的乱梯度）。
    if float(st.sum()) < 1e-3 or float(sp.sum()) < 1e-3:
        return prob.sum() * 0
    tprec = (sp * target).sum() / (sp.sum() + eps)
    tsens = (st * prob).sum() / (st.sum() + eps)
    return 1 - 2 * tprec * tsens / (tprec + tsens + eps)


class EMA:
    """教师模型用指数滑动平均：半监督里比"直接用自己"稳得多（避免自证偏差滚雪球）。"""

    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def pseudo_batch(ul_items, s, bs, rng, strong=True, in_mode='hsv'):
    """无标注瓦片：弱视图＝翻转+轻微缩放；强视图＝augment_tile（旋转/模糊/颜色）。

    这是 FixMatch/UniMatch 的核心机制：弱视图的高置信预测当伪标签，去监督强视图，
    于是"没标注的图"也能贡献梯度，而且强扰动逼模型学不变性而不是记住颜色。
    """
    Xw, Xs, Mk = [], [], []
    for _ in range(bs):
        it = ul_items[rng.randint(len(ul_items))]
        h, w = it['hsv'].shape[:2]
        s0 = int(min(s, h, w))
        y0, x0 = rng.randint(max(1, h - s0 + 1)), rng.randint(max(1, w - s0 + 1))
        a = it['hsv'][y0:y0 + s0, x0:x0 + s0]
        lab = np.zeros((s0, s0), np.uint8)
        a = cv2.resize(a, (s, s), interpolation=cv2.INTER_LINEAR)
        weak = a
        if rng.rand() < 0.5:
            weak = weak[:, ::-1]
        strong, _, _ = augment_tile(weak.copy(), lab, rng, blur=(0.0, 2.0), p_blur=0.35)
        Xw.append(np.ascontiguousarray(to_input(weak, in_mode)))
        Xs.append(np.ascontiguousarray(to_input(strong, in_mode)))
        Mk.append(np.ones((s, s), np.float32))
    tw = lambda L: torch.from_numpy(np.stack(L)).permute(0, 3, 1, 2).contiguous().float()
    return tw(Xw), tw(Xs), torch.from_numpy(np.stack(Mk)).unsqueeze(1)


def train_fold(tr_items, s, steps, lr, base, bs, pos_w, rng, aug=True, log_every=200,
               ul_items=None, ul_w=0.0, tau=0.9, loss_kind='bce+cldice', ema_decay=0.99,
               ul_bs=None, hard_neg=0, coh_w=0.0, costmap=False, edge_w0=0.2, d_ref=6.0,
               enc_kind='scratch', freeze_enc=True, enc_lr_scale=0.1, mid=128, in_mode='hsv',
               rot=0.0, cp=0.0, donors=None):
    """一个 fold 的训练。半监督：有 ul_items 且 ul_w>0 时，每步额外拉一批无标注瓦片，
    用 EMA 教师在弱视图上的高置信预测当伪标签去监督强视图（FixMatch/UniMatch 简化版）。"""
    in_ch = 6 if in_mode == 'hsvrgb' else 3
    if enc_kind and enc_kind != 'scratch':
        model = UNetPre(base, enc_kind, mid=mid, in_ch=in_ch).to('cuda')
        enc_params = [p for n, p in model.named_parameters() if n.startswith('enc.')]
        if freeze_enc:
            for p in enc_params:
                p.requires_grad_(False)
            groups = [{'params': [p for p in model.parameters() if p.requires_grad]}]
        else:
            groups = [{'params': enc_params, 'lr': lr * enc_lr_scale},
                      {'params': [p for n, p in model.named_parameters()
                                  if not n.startswith('enc.')]}]
        model = model.to('cuda')
        opt = torch.optim.Adam(groups, lr=lr)
    else:
        model = UNet(base).to('cuda')
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    # EMA 教师必须和主模型**同结构**（换预训练骨干时用 UNet 建会 load_state_dict 失败）
    teacher = (UNetPre(base, enc_kind, mid=mid, in_ch=in_ch).to('cuda')
               if enc_kind and enc_kind != 'scratch' else UNet(base).to('cuda'))
    teacher.load_state_dict(model.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    ema = EMA(model, ema_decay)
    t0 = time.time()
    ul_bs = ul_bs or bs
    hard = []
    for st in range(1, steps + 1):
        if hard_neg and st % 200 == 0:
            # 挖难负：在训练图的负样本区随机取裁剪，把模型判成绳(p>0.6)的像素中心收进池子
            model.eval()
            with torch.no_grad():
                for _ in range(4):
                    Xn, Yn = sample_batch(tr_items, s, 2, rng, aug=False, in_mode=in_mode)
                    pn = torch.sigmoid(model(Xn.cuda())).cpu().numpy()[:, 0]
                    yn = Yn.numpy()          # (B,H,W)，别切 [:,0]（那是高度维）
                    hits = (pn > 0.6) & (yn == 2)
                    if hits.any():
                        bi, yi, xi = np.where(hits)
                        for kk in rng.choice(len(yi), min(20, len(yi)), replace=False):
                            it_h = tr_items[rng.randint(len(tr_items))]
                            hard.append((it_h, int(yi[kk]), int(xi[kk])))
            hard = hard[-400:]
            model.train()
        if costmap:
            X, Y, Wt = sample_batch(tr_items, s, bs, rng, aug=aug, hard=hard,
                                    hard_frac=0.35 if hard_neg else 0.0, weights=True,
                                    in_mode=in_mode, rot=rot, cp=cp, donors=donors)
            Wt = Wt.cuda()
        else:
            X, Y = sample_batch(tr_items, s, bs, rng, aug=aug, hard=hard,
                                hard_frac=0.35 if hard_neg else 0.0, in_mode=in_mode,
                                rot=rot, cp=cp, donors=donors)
            Wt = None
        X, Y = X.cuda(), Y.cuda()
        logit = model(X).squeeze(1)               # (B,1,H,W) → (B,H,W)，与 target 对齐
        tgt = (Y == 1).float()
        m = (Y > 0).float()                       # 未标注像素不参与
        w = torch.where(tgt > 0.5, torch.full_like(tgt, pos_w), torch.ones_like(tgt))
        if Wt is not None:
            # cost map：边缘权重低、中心高（正样本），边界附近负样本也软化（容忍标注错位）
            w = w * Wt
        bce = (F.binary_cross_entropy_with_logits(logit, tgt, reduction='none') * m * w).sum() \
            / (m * w).sum().clamp(min=1)
        loss = bce
        if coh_w > 0:
            loss = loss + coh_w * opening_loss(torch.sigmoid(logit).unsqueeze(1))
        if loss_kind == 'bce+cldice':
            pr = torch.sigmoid(logit) * m
            loss = bce + 0.5 * cldice_loss(pr.unsqueeze(1), (tgt * m).unsqueeze(1), iters=16)
        if ul_items and ul_w > 0:
            Xw, Xs, _ = pseudo_batch(ul_items, s, ul_bs, rng, in_mode=in_mode)
            Xw, Xs = Xw.cuda(), Xs.cuda()
            with torch.no_grad():
                pw = torch.sigmoid(teacher(Xw).squeeze(1))
                conf = (pw > tau) | (pw < 1 - tau)
                pl = (pw > 0.5).float()
            ps = torch.sigmoid(model(Xs).squeeze(1))
            unsup = (F.binary_cross_entropy(ps, pl, reduction='none') * conf).sum() \
                / conf.sum().clamp(min=1)
            ramp = min(1.0, st / max(1, steps * 0.3))       # 预热，别一上来就被伪标签带偏
            loss = loss + ramp * ul_w * unsup
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
        teacher.load_state_dict(ema.shadow)
        if st % log_every == 0:
            print('    step %4d/%d  loss %.4f  (%.0fs)' % (st, steps, loss.item(), time.time() - t0),
                  flush=True)
    return model


@torch.no_grad()
def predict_full(model, hsv, tile=512, overlap=64, tta=False, in_mode='hsv'):
    """分块推理 + 重叠区平均，避免 4M 像素一次过图。

    tta=True 时对 4 个翻转/旋转视图各推一次再平均：细结构的分割对朝向敏感，
    测试时增强通常能把边界和断续同时改善一点，代价是 4 倍推理时间。
    """
    model.eval()
    h, w = hsv.shape[:2]
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    step = tile - overlap
    views = [(False, 0), (True, 0), (False, 2), (True, 2)] if tta else [(False, 0)]
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
            y0b, x0b = max(0, y1 - tile), max(0, x1 - tile)
            base = hsv[y0b:y1, x0b:x1]
            p = np.zeros(base.shape[:2], np.float32)
            for flip, rot in views:
                v = base
                if rot:
                    v = np.rot90(v, rot)
                if flip:
                    v = v[:, ::-1]
                v = to_input(v, in_mode)
                t = torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1)))[None].float().cuda()
                q = torch.sigmoid(model(t))[0, 0].cpu().numpy()
                if flip:
                    q = q[:, ::-1]
                if rot:
                    q = np.rot90(q, -rot)
                p += q
            p /= len(views)
            acc[y0b:y1, x0b:x1] += p
            cnt[y0b:y1, x0b:x1] += 1
    return acc / np.maximum(cnt, 1)


def load_unlabeled(pool_dirs, n, exclude, max_side=1024, seed=0, exts=('.png', '.jpg', '.jpeg', '.webp')):
    """无标注池：从给定目录随机抽 n 张（排除已标注的），缩到 max_side 内当 uint8 HSV 放着。

    半监督能不能用起来，取决于无标注池的**多样性**——这里直接抽整个 _PIC* 目录，
    所以伪标签覆盖的是"同一批画风但没标过的图"，正好补我们最缺的跨图泛化。
    """
    import random
    rnd = random.Random(seed)
    cand = []
    for d in pool_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if os.path.splitext(f)[1].lower() in exts and f not in exclude:
                cand.append(os.path.join(d, f))
    rnd.shuffle(cand)
    out = []
    for p in cand[:n]:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        k = min(1.0, max_side / max(h, w))
        if k < 1.0:
            im = cv2.resize(im, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        out.append({'name': os.path.basename(p), 'path': p,
                    'hsv': cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0})
    return out


def main():
    ap = argparse.ArgumentParser(description='U-Net 版绳子 kernel 训练/推理（LOO）')
    ap.add_argument('--gt', default=os.path.join(ROOT, 'data', 'masks', 'rope', 'raw_ps'))
    ap.add_argument('--out', default=os.path.join(ROOT, '.runtime', 'rope_cv', 'unet_loo'))
    ap.add_argument('--weights', default=os.path.join(ROOT, 'data', 'masks', 'rope', 'weights'))
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--steps', type=int, default=1600)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--base', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-aug', action='store_true', help='关掉增强（做消融）')
    ap.add_argument('--loss', default='bce+cldice', choices=['bce','bce+cldice'],
                    help='clDice 是细管状结构的拓扑保持损失，专治断续')
    ap.add_argument('--unsup-w', type=float, default=0.0, help='半监督一致性损失权重（0=关）')
    ap.add_argument('--ul-n', type=int, default=0, help='无标注池张数（0=不用半监督）')
    ap.add_argument('--ul-bs', type=int, default=0, help='无标注批大小（0=同 batch）')
    ap.add_argument('--tau', type=float, default=0.9, help='伪标签置信阈值（FixMatch 用 0.95）')
    ap.add_argument('--ema', type=float, default=0.99)
    ap.add_argument('--full-train', action='store_true',
                    help='用全部标注训一个模型并导出到 --weights（交付用，不做 LOO）')
    ap.add_argument('--mid', type=int, default=128, help='decoder mid channels')
    ap.add_argument('--encoder', default='scratch',
                    choices=['scratch', 'resnet18', 'mobilenet_v3_small', 'yolo11n_seg',
                             'swin_t', 'vit_b_16'],
                    help='预训练骨干做输入头（用户思路：单类单 mask 不需要整套 YOLO-seg）')
    ap.add_argument('--freeze-enc', type=int, default=1, help='1=冻结骨干只训解码器')
    ap.add_argument('--enc-lr-scale', type=float, default=0.1, help='不冻结时骨干的学习率倍率')
    ap.add_argument('--costmap', type=int, default=0,
                    help='1=按到绳子边界的距离给 loss 加权（中心重要、边缘弱化）')
    ap.add_argument('--edge-w0', type=float, default=0.2, help='边界像素保留的权重')
    ap.add_argument('--d-ref', type=float, default=6.0, help='多远算中心（px）')
    ap.add_argument('--pos-w', type=float, default=0.0,
                    help='>0 时覆盖正样本权重（默认按负/正比，上限 20）')
    ap.add_argument('--coh-w', type=float, default=0.0,
                    help='连贯性损失权重（惩罚细碎碎片）')
    ap.add_argument('--hard-neg', type=int, default=0, help='1=难负样本挖掘（治标得越多越敢误检）')
    ap.add_argument('--tta', type=int, default=0, help='1=测试时 4 视图增强平均')
    ap.add_argument('--rot', type=float, default=0.0,
                    help='任意角旋转幅度（度，双向）；0=只用 90 度整数倍')
    ap.add_argument('--copy-paste', dest='copy_paste', type=float, default=0.0,
                    help='Copy-Paste 概率：从训练集里剪含绳的真实图块贴进当前瓦片')
    ap.add_argument('--in', dest='in_mode', default='rgb', choices=['hsv', 'rgb', 'hsvrgb'],
                    help='网络输入表示：**rgb=默认**（实测优于 HSV，见 docs/rope_input_color.md）；'
                         'hsv=旧方案（仅为复现历史结果保留）；hsvrgb=6 通道拼接（居中，不推荐）')
    ap.add_argument('--only', default='', help='只用这些图训练（逗号分隔）')
    ap.add_argument('--test-only', default='', help='只在这些图上推理（逗号分隔）')
    ap.add_argument('--pool', default='', help='无标注图目录，逗号分隔（默认全部 _PIC0.._PIC4）')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.weights, exist_ok=True)
    items = load_items(a.gt)
    print('图 %d 张: %s' % (len(items), [it['name'] for it in items]), flush=True)
    test_items = None
    if a.only:
        keep = set(x.strip() for x in a.only.split(','))
        items = [it for it in items if it['name'] in keep]
        print('  --only 后训练/评估图: %s' % [it['name'] for it in items], flush=True)
    if a.test_only:
        want = [x.strip() for x in a.test_only.split(',')]
        test_items = [it for it in load_items(a.gt) if it['name'] in want]
        print('  --test-only 测试图: %s' % [it['name'] for it in test_items], flush=True)
    rng = np.random.RandomState(a.seed)
    # 固定 torch 的随机源：否则同一份配置每次重跑都是不同的初始化，
    # 对照实验（比如 HSV vs RGB）就分不清是"输入表示"还是"这次抽到的初始化"。
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    if a.full_train:
        # 交付：全部标注训练一次，导出权重（不做 LOO）
        npos = sum(int((x['lab'] == 1).sum()) for x in items)
        nneg = sum(int((x['lab'] == 2).sum()) for x in items)
        pos_w = float(a.pos_w) if a.pos_w > 0 else float(np.clip(nneg / max(npos, 1), 1.0, 20.0))
        print('[full-train] %d 张图  pos_w %.1f  encoder %s  mid %d'
              % (len(items), pos_w, a.encoder, a.mid), flush=True)
        model = train_fold(items, a.tile, a.steps, a.lr, a.base, a.batch, pos_w, rng,
                           aug=not a.no_aug, loss_kind=a.loss, hard_neg=a.hard_neg,
                           coh_w=a.coh_w, costmap=bool(a.costmap), edge_w0=a.edge_w0,
                           d_ref=a.d_ref, enc_kind=a.encoder, freeze_enc=bool(a.freeze_enc),
                           enc_lr_scale=a.enc_lr_scale, mid=a.mid, in_mode=a.in_mode,
                           rot=a.rot, cp=a.copy_paste, donors=items)
        # 阈值：在训练图的随机验证块上选（与 fold 内一致，不碰任何测试图）
        vb = []
        with torch.no_grad():
            for _ in range(8):
                Xv, Yv = sample_batch(items, a.tile, 4, rng, in_mode=a.in_mode)
                pv = torch.sigmoid(model(Xv.cuda())).cpu().numpy()[:, 0]
                vb.append((pv.ravel(), (Yv.numpy() == 1).ravel().astype(np.float32),
                           (Yv.numpy() > 0).ravel().astype(np.float32)))
        pv = np.concatenate([v[0] for v in vb]); yv = np.concatenate([v[1] for v in vb])
        mv = np.concatenate([v[2] for v in vb]) > 0
        best = (0, 0.9)
        for th in np.arange(0.2, 0.96, 0.02):
            pr = (pv >= th) & mv
            tp = float((pr & (yv > 0.5) & mv).sum()); fp = float((pr & (yv < 0.5) & mv).sum())
            fn = float((~pr & (yv > 0.5) & mv).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best[0]:
                best = (f1, float(th))
        os.makedirs(a.weights, exist_ok=True)
        pt = os.path.join(a.weights, 'rope_unet_%s_full.pt' % a.encoder)
        torch.save({'sd': model.state_dict(), 'base': a.base, 'mid': a.mid,
                    'encoder': a.encoder, 'thr': best[1], 'tile': 512, 'loss': a.loss,
                    'pos_w': pos_w, 'n_images': len(items), 'in_mode': a.in_mode,
                    'in_ch': 6 if a.in_mode == 'hsvrgb' else 3,
                    'val_f1': best[0]}, pt)
        print('[full-train] 已保存 %s （验证块 F1 %.3f, thr %.2f）' % (pt, best[0], best[1]))
        return

    pools = (a.pool.split(',') if a.pool else
             [os.path.join(r'<你的图片目录>', d) for d in ('_PIC0', '_PIC1', '_PIC2', '_PIC3', '_PIC4')])
    if a.ul_n > 0:
        print('半监督：无标注池 %d 张来自 %s' % (a.ul_n, pools), flush=True)
    summary = []
    folds = list(enumerate(test_items)) if test_items else list(enumerate(items))
    # --only/--test-only 分离时，每折的训练集**完全相同** → 只训一次。
    # 否则等于把同一份数据重训 5 次：报告值里混进 5 个随机初始化的噪声，而且每张测试图
    # 恰好由一个随机模型来评（图 × 初始化 混淆）。此前所有 --test-only 的对照表都有这个问题。
    disjoint = bool(test_items) and not ({x['name'] for x in test_items} &
                                         {x['name'] for x in items})
    cached = None
    for fold, te in folds:
        tr = items if disjoint else [x for x in items if x['name'] != te['name']]
        if disjoint and cached is not None:
            model, thr = cached
            print('[fold %d] 测试图 %s → 复用同一模型（训练集相同，不重复训练）'
                  % (fold, te['name']), flush=True)
        else:
            npos = sum(int((x['lab'] == 1).sum()) for x in tr)
            nneg = sum(int((x['lab'] == 2).sum()) for x in tr)
            pos_w = float(np.clip(nneg / max(npos, 1), 1.0, 20.0))
            if a.pos_w > 0:
                pos_w = float(a.pos_w)
            print('[fold %d] 测试图 %s  训练图 %s  pos_weight %.1f'
                  % (fold, te['name'], [x['name'] for x in tr], pos_w), flush=True)
            ul = None
            if a.ul_n > 0:
                excl = {os.path.basename(te.get('path', '')) or te['name'] + '.png'}
                ul = load_unlabeled([d for d in pools if os.path.isdir(d)], a.ul_n, exclude=set(),
                                    seed=a.seed + fold)
                ul = [u for u in ul if os.path.basename(u['path']) != os.path.basename(te.get('path', 'x'))]
            model = train_fold(tr, a.tile, a.steps, a.lr, a.base, a.batch, pos_w, rng,
                               aug=not a.no_aug, ul_items=ul, ul_w=a.unsup_w, tau=a.tau,
                               loss_kind=a.loss, ul_bs=(a.ul_bs or a.batch), hard_neg=a.hard_neg,
                               coh_w=a.coh_w, costmap=a.costmap, edge_w0=a.edge_w0,
                               d_ref=a.d_ref, enc_kind=a.encoder, freeze_enc=bool(a.freeze_enc),
                               enc_lr_scale=a.enc_lr_scale, mid=a.mid, in_mode=a.in_mode,
                           rot=a.rot, cp=a.copy_paste, donors=items)
            # 阈值只在**训练图**的随机验证块上选（不碰测试图）
            vb = []
            with torch.no_grad():
                for _ in range(6):
                    X, Y = sample_batch(tr, a.tile, 4, rng, in_mode=a.in_mode)
                    p = torch.sigmoid(model(X.cuda())).cpu().numpy()[:, 0]
                    vb.append((p.ravel(), (Y.numpy() == 1).ravel().astype(np.float32),
                               (Y.numpy() > 0).ravel().astype(np.float32)))
            pv = np.concatenate([v[0] for v in vb])
            yv = np.concatenate([v[1] for v in vb])
            mv = np.concatenate([v[2] for v in vb]) > 0
            best = (0, 0.5)
            for thr in np.arange(0.2, 0.91, 0.02):
                pred = (pv >= thr) & mv
                tp = float((pred & (yv > 0.5) & mv).sum())
                fp = float((pred & (yv < 0.5) & mv).sum())
                fn = float((~pred & (yv > 0.5) & mv).sum())
                f1 = 2 * tp / max(2 * tp + fp + fn, 1)
                if f1 > best[0]:
                    best = (f1, float(thr))
            thr = best[1]
            if disjoint:
                cached = (model, thr)
        t0 = time.time()
        prob = predict_full(model, te['hsv'], tta=bool(a.tta), in_mode=a.in_mode)
        dt = time.time() - t0
        px = te['hsv'].shape[0] * te['hsv'].shape[1]
        mask = (prob >= thr).astype(np.uint8) * 255
        n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        drop = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < 32]
        if drop:
            mask[np.isin(cc, drop)] = 0
        cv2.imwrite(os.path.join(a.out, te['name'] + '_mask.png'), mask)
        cv2.imwrite(os.path.join(a.out, te['name'] + '_conf.png'), (np.clip(prob, 0, 1) * 255).astype(np.uint8))
        # 单模型模式（训练集与测试集不相交）下 5 折就是同一个模型，**只存一次**——
        # 之前每个测试图存一份完全相同的 46MB 权重，一个 3-seed 组白占 690MB（G 盘被塞满过一次）。
        if (not disjoint) or fold == folds[0][0]:
            torch.save({'sd': model.state_dict(), 'base': a.base, 'thr': thr, 'tile': a.tile,
                        'loss': a.loss, 'unsup_w': a.unsup_w, 'ul_n': a.ul_n,
                        'encoder': a.encoder, 'mid': a.mid,      # 交付/复现都需要，别再靠反推
                        'in_mode': a.in_mode, 'in_ch': 6 if a.in_mode == 'hsvrgb' else 3},
                       os.path.join(a.weights,
                                    'rope_unet_%s_%s.pt' % (os.path.basename(a.out.rstrip(os.sep)),
                                                            te['name'])))
        rec = {'image': te['name'], 'thr': round(thr, 3), 'thr_val_F1': round(best[0], 4),
               'coverage': round(float((mask > 0).mean()), 4), 'pos_weight': round(pos_w, 2),
               'ms_per_Mpx': round(1000 * dt / (px / 1e6), 1)}
        summary.append(rec)
        print('    阈值 %.2f(验证F1 %.3f) 覆盖率 %.3f%%  稠密推理 %.1f ms/Mpx'
              % (thr, best[0], 100 * (mask > 0).mean(), rec['ms_per_Mpx']), flush=True)
    with io.open(os.path.join(a.out, 'unet_loo.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print('完成 → %s（再用 scripts/rope_eval.py --pred 该目录 算 KPI）' % a.out)


if __name__ == '__main__':
    main()
