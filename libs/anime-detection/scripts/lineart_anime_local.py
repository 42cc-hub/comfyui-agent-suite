"""把 ControlNet-aux 的 **AnimeLineart** 生成器（netG.pth）单独接出来用。

为什么：手搓的 blackhat/ridge 线检测在这个尺度与低对比度场景上不行（实测把整块背景当成了面块）。
这个模型是专门为动画线稿训的 pix2pix U-Net（3→1 通道），本机权重在
`<你的 ComfyUI 安装目录>\\custom_nodes\\comfyui_controlnet_aux\\ckpts\\lllyasviel\\Annotators\\netG.pth`（207.6 MB），
结构照抄 `custom_controlnet_aux/lineart_anime/__init__.py`（不 import controlnet_aux，避免 ComfyUI 依赖）。

对外只暴露 `line_response(bgr, net, ...) -> float32[H,W] ∈ [0,1]`（1=越像线），以及 `load_netG()`。
"""
import functools
import os

import numpy as np
import cv2
import torch
import torch.nn as nn

CKPT = r'<你的 ComfyUI 安装目录>\custom_nodes\comfyui_controlnet_aux\ckpts\lllyasviel\Annotators\netG.pth'


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        use_bias = (norm_layer.func == nn.InstanceNorm2d) if isinstance(norm_layer, functools.partial) \
            else (norm_layer == nn.InstanceNorm2d)
        input_nc = outer_nc if input_nc is None else input_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu, downnorm = nn.LeakyReLU(0.2, True), norm_layer(inner_nc)
        uprelu, upnorm = nn.ReLU(True), norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if use_dropout:
                model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x) if self.outermost else torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                        norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):
            block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=block,
                                            norm_layer=norm_layer, use_dropout=use_dropout)
        block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=block,
                                             outermost=True, norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)


def load_netG(path=None, device=None):
    path = path or CKPT
    if not os.path.exists(path):
        raise SystemExit('缺 netG.pth：%s' % path)
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    net = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
    ckpt = torch.load(path, map_location='cpu')
    for k in list(ckpt.keys()):
        if 'module.' in k:
            ckpt[k.replace('module.', '')] = ckpt.pop(k)
    net.load_state_dict(ckpt)
    net.eval()
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(dev)
    return net, dev


def line_response(bgr, net, dev, max_side=1024):
    """返回 [0,1] 的"线强度"图（1=线）。自动判极性：与"暗细线"响应相关性更高的那一边当线。"""
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) \
        if s < 1.0 else bgr
    H, W = small.shape[:2]
    Hn, Wn = 256 * int(np.ceil(H / 256.0)), 256 * int(np.ceil(W / 256.0))
    pad = cv2.copyMakeBorder(small, 0, Hn - H, 0, Wn - W, cv2.BORDER_REPLICATE).astype(np.float32)
    x = torch.from_numpy(pad[:, :, ::-1].copy()).permute(2, 0, 1)[None].to(dev) / 127.5 - 1.0
    with torch.no_grad():
        out = net(x)[0, 0].clamp(-1, 1).cpu().numpy()
    out = (out + 1.0) * 127.5                      # 0..255，模型输出的极性未知
    out = out[:H, :W]
    # 极性判定：与 blackhat（薄暗线）响应的相关性
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    a = out.ravel()
    b = bh.ravel()
    r_pos = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    r_neg = float(np.corrcoef(255.0 - a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    resp = (out / 255.0) if r_pos >= r_neg else (1.0 - out / 255.0)
    resp = cv2.resize(resp.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = np.percentile(resp, 1), np.percentile(resp, 99.5)
    resp = np.clip((resp - lo) / max(1e-6, hi - lo), 0, 1)
    return resp, {'polarity': '+' if r_pos >= r_neg else '-', 'r_pos': r_pos, 'r_neg': r_neg}


if __name__ == '__main__':
    import sys
    img = sys.argv[1]
    bgr = cv2.imread(img, cv2.IMREAD_COLOR)
    net, dev = load_netG()
    resp, info = line_response(bgr, net, dev)
    print('  %s  dev=%s  极性=%s (r+=%.3f r-=%.3f)' % (os.path.basename(img), dev, info['polarity'],
                                                     info['r_pos'], info['r_neg']))
    print('  线强度分位：p50 %.3f p90 %.3f p97 %.3f p99 %.3f；>0.5 占 %.2f%%'
          % (np.percentile(resp, 50), np.percentile(resp, 90), np.percentile(resp, 97),
             np.percentile(resp, 99), 100 * (resp > 0.5).mean()))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'out', 'lineart_check')
    os.makedirs(out, exist_ok=True)
    fp = os.path.join(out, os.path.splitext(os.path.basename(img))[0] + '_lineart.png')
    cv2.imwrite(fp, (resp * 255).astype(np.uint8))
    print('  写出 %s' % fp)
