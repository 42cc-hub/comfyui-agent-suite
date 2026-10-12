#!/usr/bin/env python
"""挑"没有绳子"的负样本候选（用户思路：看一眼就能整图标成非绳，标注成本极低）。

为什么要按"模型烧得厉害"排：我们的误检是同色异物（红衣物/红裙/红发饰）与暗场景，
确认一批"模型点亮了但其实没有绳子"的图当负样本，能直接惩罚这类错误，性价比最高。

两段式：
  ① pick    对无标注池抽样跑当前交付权重，算 cov(命中率) / red(红色占比) / V(亮度)，
            分桶做成缩略图接触表（图里有编号，人一眼扫完就能报"这些没绳"）。
  ② confirm 把人确认过的候选一键写成整图非绳标签（labels 全 2 / mask 全 0 / meta 记 reviewed），
            并把原图复制进 _orig，命名成新批号。**不用开 Photoshop**。

    python scripts/rope_neg_pick.py --n 600                      # ①
    python scripts/rope_neg_pick.py --confirm 12,37,41 --batch w # ②
"""
import argparse
import glob
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, '.runtime', 'rope_cv')
ORIG = os.path.join(W, 'ps_work', '_orig')
RAW = os.path.join(ROOT, 'data', 'masks', 'rope', 'raw_ps')
POOLS = [os.path.join(r'H:\FILE\temple', d) for d in ('_PIC0', '_PIC1', '_PIC2', '_PIC3', '_PIC4')]
EXTS = ('.png', '.jpg', '.jpeg', '.webp')


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def phash(gray):
    g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    v = cv2.dct(g)[:8, :8].ravel()[1:]
    return (v > np.median(v)).astype(np.uint8)


def known_set():
    """已标/已知的原图：md5 + pHash，用来避免重复标同一张（已经抓到过一例 v03==s02）。"""
    known = []
    for f in sorted(os.listdir(ORIG)):
        p = os.path.join(ORIG, f)
        if not os.path.isfile(p):
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        known.append({'file': f, 'md5': md5(p), 'p': phash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))})
    return known


def pool_files(limit_dirs=None):
    out = []
    for d in (limit_dirs or POOLS):
        for e in EXTS:
            out += glob.glob(os.path.join(d, '*' + e))
    return out


def score(img, prob, thr):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = float((((h <= 10) | (h >= 170)) & (s >= 80) & (v >= 60)).mean())
    m = prob >= thr
    n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    blk = [int(st[i, cv2.CC_STAT_AREA]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 64]
    return {'cov': round(float(m.mean()) * 100, 4),
            'cov50': round(float((prob >= 0.5).mean()) * 100, 4),
            'blocks': len(blk), 'max_blk': (max(blk) if blk else 0),
            'red': round(red * 100, 2), 'V': round(float(v.mean()) / 255, 3),
            'px': int(img.shape[0] * img.shape[1])}


def sheet(items, path, cols=4, rows=5, cell=320):
    """缩略图接触表：编号 + 关键统计，方便人一眼扫完报编号。"""
    if not items:
        return
    per = cols * rows
    for k in range(0, len(items), per):
        chunk = items[k:k + per]
        ch = cell + 34
        canvas = np.full((rows * ch, cols * cell, 3), 245, np.uint8)
        for i, it in enumerate(chunk):
            r, c = divmod(i, cols)
            th = cv2.imread(it['thumb'])
            if th is None:
                continue
            sc = min(cell / th.shape[1], cell / th.shape[0])
            th = cv2.resize(th, (max(1, int(th.shape[1] * sc)), max(1, int(th.shape[0] * sc))))
            y0, x0 = r * ch + 34, c * cell
            canvas[y0:y0 + th.shape[0], x0:x0 + th.shape[1]] = th
            txt = '#%d cov %.2f%% blk %d red %.1f%% V %.2f' % (
                it['idx'], it['cov'], it['blocks'], it['red'], it['V'])
            cv2.putText(canvas, txt, (x0 + 4, r * ch + 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 0, 0), 1, cv2.LINE_AA)
        n = k // per + 1
        cv2.imwrite(path % n, canvas)


def do_pick(a):
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, 'thumbs'), exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import rope_predict as RP
    wp = a.weights or RP.DEFAULT_W
    m = RP.load_model(wp)
    thr = float(a.conf) if a.conf is not None else m['thr']
    print('权重 %s  thr %.2f' % (os.path.basename(wp), thr), flush=True)

    known = known_set()
    khashes = np.stack([k['p'] for k in known]) if known else None
    kmd5 = {k['md5']: k['file'] for k in known}
    files = pool_files([d for d in a.pool.split(',') if d] if a.pool else None)
    random.Random(a.seed).shuffle(files)
    print('池 %d 张，目标抽样 %d 张' % (len(files), a.n), flush=True)

    rows = []
    t0 = time.time()
    for fp in files:
        if len(rows) >= a.n:
            break
        try:
            if md5(fp) in kmd5:
                print('  跳过（与已标图同文件）:', os.path.basename(fp), flush=True)
                continue
        except Exception:
            continue
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 200:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        prob = RP.dense_prob_unet(m, hsv, tile=m.get('tile', 512))
        rec = score(img, prob, thr)
        idx = len(rows)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hp = int(np.count_nonzero(phash(g) != khashes).min()) if khashes is not None else 999
        sc = min(360 / img.shape[1], 360 / img.shape[0], 1.0)
        thumb = os.path.join(a.out, 'thumbs', '%04d.jpg' % idx)
        cv2.imwrite(thumb, cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc))),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        rec.update({'idx': idx, 'src': fp, 'thumb': thumb, 'phash_known': hp})
        rows.append(rec)
        if len(rows) % 50 == 0:
            print('  %d/%d  %.1f min  最近一张 cov %.2f%%' %
                  (len(rows), a.n, (time.time() - t0) / 60, rec['cov']), flush=True)

    def near_dup(r):
        return r['phash_known'] <= 12

    p1 = sorted([r for r in rows if r['cov'] >= 1.0], key=lambda r: -r['cov'])
    p2 = sorted([r for r in rows if r['red'] >= 8 and r['cov'] < 1.0], key=lambda r: -r['red'])
    p3 = sorted([r for r in rows if r['cov'] < 0.05 and r['red'] < 3], key=lambda r: r['idx'])
    for nm, arr in (('P1_model_fires', p1), ('P2_red_no_fire', p2), ('P3_clean', p3)):
        sheet(arr, os.path.join(a.out, nm + '_%d.png'), cols=4, rows=5)
    with io.open(os.path.join(a.out, 'candidates.json'), 'w', encoding='utf-8') as fh:
        json.dump({'weights': os.path.basename(wp), 'thr': thr, 'n': len(rows),
                   'P1': [r['idx'] for r in p1], 'P2': [r['idx'] for r in p2],
                   'P3': [r['idx'] for r in p3], 'rows': rows}, fh, ensure_ascii=False, indent=1)

    lines = ['# 无绳负样本候选（抽样 %d 张，thr %.2f）' % (len(rows), thr), '',
             'P1 = 模型烧起来了（cov ≥ 1%%，确认无绳就是最值钱的误检负样本；如果其实有绳，那是正样本）',
             'P2 = 红得多但没怎么响应（red ≥ 8%%、cov < 1%%）', 'P3 = 干净背景（cov < 0.05%%）', '',
             '## P1（%d 张，接触表 P1_model_fires_*.png）' % len(p1), '',
             '| # | cov% | 块数 | 最大块px | red% | V | 近似重复 | 文件 |', '|---|---|---|---|---|---|---|---|']
    for r in p1[:40]:
        lines.append('| %d | %.3f | %d | %d | %.1f | %.2f | %s | %s |'
                     % (r['idx'], r['cov'], r['blocks'], r['max_blk'], r['red'], r['V'],
                        '⚠️' if near_dup(r) else '', os.path.basename(r['src'])))
    lines += ['', '## P2 前 20', '', '| # | red% | cov% | V | 文件 |', '|---|---|---|---|---|']
    for r in p2[:20]:
        lines.append('| %d | %.1f | %.3f | %.2f | %s |' % (r['idx'], r['red'], r['cov'], r['V'],
                                                          os.path.basename(r['src'])))
    lines += ['', '## P3 前 20', '', '| # | cov% | red% | V | 文件 |', '|---|---|---|---|---|']
    for r in p3[:20]:
        lines.append('| %d | %.3f | %.1f | %.2f | %s |' % (r['idx'], r['cov'], r['red'], r['V'],
                                                          os.path.basename(r['src'])))
    with io.open(os.path.join(a.out, 'candidates.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print('\nP1 %d / P2 %d / P3 %d（共抽样 %d）' % (len(p1), len(p2), len(p3), len(rows)))
    print('接触表与清单：', a.out)


def do_confirm(a):
    man = json.load(io.open(os.path.join(a.out, 'candidates.json'), encoding='utf-8'))
    by_idx = {r['idx']: r for r in man['rows']}
    by_stem = {}
    for r in man['rows']:
        by_stem[os.path.splitext(os.path.basename(r['src']))[0]] = r
    done = []
    for j, tok in enumerate(a.confirm.split(',')):
        tok = tok.strip().lstrip('#')
        rec = by_idx.get(int(tok)) if tok.isdigit() else by_stem.get(tok)
        if rec is None:
            print('!! 找不到候选:', tok)
            continue
        name = '%s%02d' % (a.batch, j + 1)
        src = rec['src']
        dst = os.path.join(ORIG, name + '_' + os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        img = cv2.imread(dst, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        cv2.imwrite(os.path.join(RAW, name + '_labels.png'), np.full((h, w), 2, np.uint8))
        cv2.imwrite(os.path.join(RAW, name + '_mask.png'), np.zeros((h, w), np.uint8))
        meta = {'image': os.path.basename(dst), 'size': [w, h], 'rope_px': 0,
                'nonrope_px': int(h * w), 'unlabeled_px': 0,
                'source': '人确认无绳（接触表核对）→ 整图非绳', 'reviewed': True,
                'picked_from': src, 'pick_stats': {k: rec[k] for k in
                                                   ('cov', 'blocks', 'red', 'V', 'phash_known')}}
        with io.open(os.path.join(RAW, name + '_meta.json'), 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        done.append((name, os.path.basename(src), rec['cov'], rec['red']))
        print('  %s ← %s (cov %.2f%% red %.1f%%)' % (name, os.path.basename(src),
                                                     rec['cov'], rec['red']))
    print('\n已写入 %d 张整图非绳标签到 %s' % (len(done), RAW))


def do_sheets(a):
    """只重排接触表（不重跑推理）。

    A/B/C/D 是给"绳子题材池"用的分档（看了一遍 600 张：cov ≥ 5% 的基本都真有红绳，
    所以"模型烧得厉害"不等于误检；真正值钱的是"红得多 / 模型还是点了 / 其实没绳"）。
    E1/E2/E3 是通用三分（模型点着的 / 点了一点的 / 完全没反应），批量为 48 张一张表。
    """
    man = json.load(io.open(os.path.join(a.out, 'candidates.json'), encoding='utf-8'))
    rows = man['rows']

    def pick(cond, key, n=16):
        arr = [r for r in rows if cond(r)]
        arr.sort(key=key)
        return arr[:n]

    A = pick(lambda r: r['cov'] < 1.0 and r['red'] >= 5.0, key=lambda r: -r['red'])
    B = pick(lambda r: 1.0 <= r['cov'] < 3.5 and r['red'] >= 8.0, key=lambda r: -r['red'])
    C = pick(lambda r: 3.5 <= r['cov'] < 6.0 and r['red'] >= 10.0, key=lambda r: -r['red'])
    D = pick(lambda r: r['cov'] < 0.05 and r['red'] < 5.0, key=lambda r: r['idx'])
    # 通用三分（任何池子都适用）：模型点着的 / 点了一点的 / 完全没反应的。
    # 批量为 48 张一张表（用户要的"批量确认模式"），每格小一点，靠人眼扫。
    bm = bool(getattr(a, 'batch_mode', False))
    cols, rows_n, cell = (8, 6, 250) if bm else (4, 4, 360)
    n_per = 960 if bm else 16
    E1 = pick(lambda r: r['cov'] >= 1.0, key=lambda r: -r['cov'], n=n_per)
    E2 = pick(lambda r: 0.05 <= r['cov'] < 1.0, key=lambda r: -r['cov'], n=n_per)
    E3 = pick(lambda r: r['cov'] < 0.05, key=lambda r: r['idx'], n=n_per)
    for nm, arr in (('E1_model_fires', E1), ('E2_mid', E2), ('E3_zero_resp', E3)):
        sheet(arr, os.path.join(a.out, nm + '_%d.png'), cols=cols, rows=rows_n, cell=cell)
    for nm, arr in (('A_red_nofire', A), ('B_mid_red', B), ('C_fires_red', C), ('D_clean', D)):
        sheet(arr, os.path.join(a.out, nm + '_%d.png'), cols=4, rows=4, cell=360)
    with io.open(os.path.join(a.out, 'sheets.md'), 'w', encoding='utf-8') as fh:
        for nm, arr, desc in (('E1_model_fires', E1, '模型点着了（cov≥1%）——要么是误检（最值钱的负样本），要么是真的绳（正样本）'),
                              ('E2_mid', E2, '模型点了一点（0.05%~1%）——多半没问题，扫一眼'),
                              ('E3_zero_resp', E3, '模型完全没反应（cov<0.05%）——批量负样本主力'),
                              ('A_red_nofire', A, '红得多但模型几乎不响应（cov<1%）'),
                              ('B_mid_red', B, '中等响应 + 红（1%~3.5%）'),
                              ('C_fires_red', C, '模型烧了不少 + 红（3.5%~6%）'),
                              ('D_clean', D, '零响应且不红')):
            fh.write('## %s — %s\n\n| # | cov%% | 块数 | 最大块px | red%% | V | 文件 |\n|---|---|---|---|---|---|---|\n'
                     % (nm, desc))
            for r in arr:
                fh.write('| %d | %.3f | %d | %d | %.1f | %.2f | %s |\n'
                         % (r['idx'], r['cov'], r['blocks'], r['max_blk'], r['red'], r['V'],
                            os.path.basename(r['src'])))
            fh.write('\n')
    print('A %d / B %d / C %d / D %d / E1 %d / E2 %d / E3 %d'
          % (len(A), len(B), len(C), len(D), len(E1), len(E2), len(E3)))
    print('\n'.join('  %s: %s' % (nm, [r['idx'] for r in arr])
                    for nm, arr in (('A', A), ('B', B), ('C', C), ('D', D))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(W, 'neg_pick'))
    ap.add_argument('--weights', default='')
    ap.add_argument('--conf', type=float, default=None)
    ap.add_argument('--n', type=int, default=600, help='抽样张数')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--confirm', default='', help='人确认无绳的候选编号，逗号分隔（如 12,37）')
    ap.add_argument('--batch', default='w', help='新批号前缀（默认 w）')
    ap.add_argument('--sheets-only', action='store_true', help='用已有 candidates.json 重排接触表')
    ap.add_argument('--pool', default='', help='要筛的目录，逗号分隔（默认 H:\\FILE\\temple\\_PIC0..4）')
    ap.add_argument('--batch-mode', action='store_true', help='批量确认模式：48 张/表、格子小一点')
    a = ap.parse_args()
    if a.confirm:
        do_confirm(a)
    elif a.sheets_only:
        do_sheets(a)
    else:
        do_pick(a)


if __name__ == '__main__':
    main()
