#!/usr/bin/env python
"""绳子 kernel 训练：从 PS 标注（data/masks/rope/raw_ps/*_labels.png）训一个 k×k×HSV 的小 MLP。

标签约定（raw_ps/<name>_labels.png，uint8）：0=未标注（忽略）1=绳子 2=非绳子。
输入是像素周围 k×k 邻域的 HSV 摊平（k*k*3 维）——即一个 k×k×3 的卷积核 + 1×1 MLP。

划分纪律（对外接口要求）：**按整图留出**，绝不按像素随机切。
  * 标注图 ≥2 张时：leave-one-image-out（每张轮流当测试图，报每张 + 汇总）；
  * 只有 1 张时：退化为图内 256px 块棋盘留出，并在 metrics 里显式标 in_image_split=true。
难负类假阳率：把"基线 HSV 规则会判成绳子的负样本像素"单列出来算 FP —— 潮红/勒痕/红衣物
这些真正难的东西就落在这里，比总 FP 更能反映人工成本。

产物：data/masks/rope/weights/<tag>.npz（纯 numpy 权重，推理端不需要 torch）
      data/masks/rope/weights/<tag>.json（指标，供模块文档引用）

用法（anime_det 环境，sklearn 够用）：
  python scripts/rope_train.py --src data/masks/rope/raw_ps --tag rope_mlp_k9 --kernel 9
"""
import argparse
import glob
import io
import json
import os
import sys

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIRS = [os.path.join(ROOT, '.runtime', 'rope_cv', 'ps_work'), ROOT]


def find_image(name, label_path=None):
    """按 <name>_meta.json 的 image 字段 / <name>.* / <name>_* 三种方式找原图。"""
    cands = []
    mp = os.path.join(os.path.dirname(label_path or ''), name + '_meta.json')
    if label_path and os.path.exists(mp):
        try:
            cands.append(json.load(io.open(mp, encoding='utf-8'))['image'])
        except Exception:
            pass
    for d in IMG_DIRS:
        for c in cands:
            p = os.path.join(d, c)
            if os.path.exists(p):
                return p
        for ext in ('.png', '.jpg', '.jpeg', '.webp'):
            if os.path.exists(os.path.join(d, name + ext)):
                return os.path.join(d, name + ext)
        g = sorted(glob.glob(os.path.join(d, name + '_*')))
        g = [x for x in g if not x.endswith(('_mask.png', '_labels.png', '_meta.json'))]
        if g:
            return g[0]
    return None


def gather(hsv, r, cy, cx):
    return np.concatenate([hsv[cy + dy, cx + dx, :]
                           for dy in range(-r, r + 1)
                           for dx in range(-r, r + 1)], axis=1)


def pick(mask, cap, seed):
    ys, xs = np.where(mask)
    if len(ys) > cap:
        k = np.random.RandomState(seed).choice(len(ys), cap, replace=False)
        ys, xs = ys[k], xs[k]
    return ys, xs


def block_split(shape, r, tile=256):
    h, w = shape
    gi, gj = (np.arange(h)[:, None] // tile) // 2, (np.arange(w)[None, :] // tile) // 2
    return ((gi + gj) % 2 == 0)


def predict_pixels(hsv, r, ys, xs, mu, sd, mlp, chunk=200000):
    """对任意像素列表分批预测（测试集用全量像素、保持真实先验，不采样成 1:1）。"""
    out = np.empty(len(ys), np.float32)
    for i in range(0, len(ys), chunk):
        X = gather(hsv, r, ys[i:i + chunk], xs[i:i + chunk])
        out[i:i + chunk] = mlp.predict_proba((X - mu) / sd)[:, 1]
    return out


def load_items(src):
    items = []
    for lp in sorted(glob.glob(os.path.join(src, '*_labels.png'))):
        name = os.path.basename(lp)[:-len('_labels.png')]
        ip = find_image(name, lp)
        if ip is None:
            print('跳过（找不到原图）:', name)
            continue
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        lab = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        if lab.shape != img.shape[:2]:
            lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        items.append({'name': name, 'img': img,
                      'hsv': cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0,
                      'rope': lab == 1, 'other': lab == 2})
    return items


def metrics(scores, y):
    if y.sum() == 0 or y.sum() == len(y):
        # 全负/全正的 fold（例如这张图确认没有绳子）：AUC/F1 都无定义，别让 sklearn 抛异常
        return {'auc': None, 'ap': None, 'bestF1': None, 'bestThr': None,
                'bestP': None, 'bestR': None, 'degenerate': True}
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    k = int(np.nanargmax(f1))
    return {'auc': round(float(roc_auc_score(y, scores)), 4),
            'ap': round(float(average_precision_score(y, scores)), 4),
            'bestF1': round(float(np.nanmax(f1)), 4), 'bestThr': round(float(thr[min(k, len(thr) - 1)]), 3),
            'bestP': round(float(prec[k]), 4), 'bestR': round(float(rec[k]), 4)}


def main():
    ap = argparse.ArgumentParser(description='训练绳子像素 kernel（k×k×HSV 小 MLP）')
    ap.add_argument('--src', default=os.path.join(ROOT, 'data', 'masks', 'rope', 'raw_ps'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'masks', 'rope', 'weights'))
    ap.add_argument('--tag', default='rope_mlp_k9')
    ap.add_argument('--kernel', type=int, default=9)
    ap.add_argument('--hidden', default='48,24')
    ap.add_argument('--per-class', type=int, default=40000)
    ap.add_argument('--test-per-class', type=int, default=40000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-masks', default='', help='用 fold 模型给留出图存 mask（供 rope_eval.py）')
    ap.add_argument('--backend', default='numpy', choices=['numpy','torch'])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = a.kernel // 2
    hid = tuple(int(x) for x in a.hidden.split(','))
    items = load_items(a.src)
    if not items:
        raise SystemExit('没有可用的 *_labels.png')
    print('标注图 %d 张: %s' % (len(items), [it['name'] for it in items]))
    per_image = len(items) > 1

    # 预计算每张图的"可用中心"（5x5 或 kxk 全落在标注内）；分块划分只在单图时用
    for it in items:
        lab0 = it['rope'] | it['other']
        pad0 = cv2.copyMakeBorder(lab0.astype(np.uint8), r, r, r, r, cv2.BORDER_CONSTANT, value=0)
        it['inside'] = cv2.erode(pad0, np.ones((2 * r + 1,) * 2, np.uint8))[r:r + lab0.shape[0],
                                                                             r:r + lab0.shape[1]] > 0
        it['blocks'] = block_split(lab0.shape, r)

    results = []
    for fold, it in enumerate(items):
        # 训练样本来源：多图＝除本张外的所有图（按整图留出）；单图＝本图训练块
        src_items = [x for j, x in enumerate(items) if j != fold] if per_image else [it]
        tr_masks = []
        for x in src_items:
            tr_masks.append(x['inside'] if per_image else (x['inside'] & x['blocks']))
        tys_list, txs_list, ytr_list = [], [], []
        for x, tm in zip(src_items, tr_masks):
            cap = a.per_class // max(1, len(src_items))
            cy, cx = pick(tm & x['rope'], cap, a.seed + fold)
            oy, ox = pick(tm & x['other'], cap, a.seed + 100 + fold)
            if len(cy):
                tys_list.append(gather(x['hsv'], r, cy, cx))
                ytr_list.append(np.ones(len(cy)))
            if len(oy):
                tys_list.append(gather(x['hsv'], r, oy, ox))
                ytr_list.append(np.zeros(len(oy)))
        Xt = np.vstack(tys_list)
        yt = np.concatenate(ytr_list)
        te = it['inside'] if per_image else (it['inside'] & ~it['blocks'])
        # 测试集＝留出图/留出块里**全部**已标注像素（真实先验），不采样成 1:1
        tys, txs = np.where(te)
        te_lab = np.where(it['rope'][tys, txs], 1.0, 0.0)
        results.append({'name': it['name'], 'Xtr': Xt, 'ytr': yt, 'tys': tys, 'txs': txs,
                        'yte': te_lab, 'item': it, 'te': te})

    X = np.vstack([r0['Xtr'] for r0 in results])
    y = np.concatenate([r0['ytr'] for r0 in results])
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6

    # 跨图指标必须用**每个 fold 自己的模型**（只用其余图训练）去评留出图。
    # 第一版是"一个模型吃全部图、再逐图评"，那是拿测试图训练过的自欺，已改。
    rep, hard, all_p, all_y, fold_models = [], [], [], [], []
    for r0 in results:
        mf = MLPClassifier(hidden_layer_sizes=hid, alpha=1e-4, max_iter=300,
                           early_stopping=True, n_iter_no_change=10, random_state=a.seed)
        mf.fit((r0['Xtr'] - mu) / sd, r0['ytr'])
        fold_models.append(mf)
        p = predict_pixels(r0['item']['hsv'], r, r0['tys'], r0['txs'], mu, sd, mf)
        m = metrics(p, r0['yte'])
        if m.get('degenerate'):
            fp_only = int((p >= 0.5).sum())
            m.update({'image': r0['name'], 'n_test': int(len(p)), 'FP_only_px': fp_only})
            rep.append(m)
            print('  留出图 %s 无正样本：仅统计误检 %d px' % (r0['name'], fp_only))
            continue
        m.update({'image': r0['name'], 'n_test': int(len(p)), 'n_pos': int(r0['yte'].sum()),
                  'n_neg': int((r0['yte'] == 0).sum())})
        tp = int(((p >= 0.5) & (r0['yte'] == 1)).sum())
        fp = int(((p >= 0.5) & (r0['yte'] == 0)).sum())
        m['FP/TP@0.5'] = round(fp / max(tp, 1), 3)
        rep.append(m)
        all_p.append(p)
        all_y.append(r0['yte'])
        H = r0['item']['hsv'][r0['tys'], r0['txs'], 0] * 179
        S = r0['item']['hsv'][r0['tys'], r0['txs'], 1] * 255
        V = r0['item']['hsv'][r0['tys'], r0['txs'], 2] * 255
        hh = np.minimum(np.abs(H), 180 - np.abs(H))
        base_hit = (hh <= 14) & (S >= 150) & (V >= 60)
        neg = r0['yte'] == 0
        hard.append({'image': r0['name'], 'hard_neg_n': int((base_hit & neg).sum()),
                     'hard_neg_FP': int((base_hit & neg & (p >= 0.5)).sum()),
                     'easy_neg_FP': int((~base_hit & neg & (p >= 0.5)).sum())})
    ag_p = np.concatenate([p for p, yv in zip(all_p, all_y) if yv.sum() > 0]) if all_p else np.zeros(0)
    ag_y = np.concatenate([yv for yv in all_y if yv.sum() > 0]) if all_y else np.zeros(0)
    agg = metrics(ag_p, ag_y) if len(ag_y) and ag_y.sum() > 0 else {'bestThr': 0.5}
    thr = agg['bestThr'] if agg.get('bestThr') else 0.5

    # 交付模型：用**所有**标注图训练（部署时不存在留出问题）；上面的指标才是它的泛化估计
    mlp = MLPClassifier(hidden_layer_sizes=hid, alpha=1e-4, max_iter=300, early_stopping=True,
                        n_iter_no_change=10, random_state=a.seed)
    mlp.fit((X - mu) / sd, y)
    print('交付模型：%d 图 / %d 样本训练，迭代 %d' % (len(items), len(y), mlp.n_iter_))

    # 可选：用**fold 模型**给每张留出图存 mask，供 rope_eval.py 统一算误检块/漏检段/预计笔数
    if a.save_masks:
        os.makedirs(a.save_masks, exist_ok=True)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import rope_predict as RP
        for r0, mf in zip(results, fold_models):
            mm = {'w1': mf.coefs_[0], 'b1': mf.intercepts_[0], 'w2': mf.coefs_[1],
                  'b2': mf.intercepts_[1], 'w3': mf.coefs_[2], 'b3': mf.intercepts_[2],
                  'mean': mu, 'std': sd, 'k': a.kernel, 'thr': thr}
            prob = RP.dense_prob(r0['item']['hsv'], mm, backend=a.backend)
            mk = (prob >= thr).astype(np.uint8) * 255
            n2, cc2, st2, _ = cv2.connectedComponentsWithStats(mk, 8)
            drop = [i for i in range(1, n2) if st2[i, cv2.CC_STAT_AREA] < 32]
            if drop:
                mk[np.isin(cc2, drop)] = 0
            cv2.imwrite(os.path.join(a.save_masks, r0['name'] + '_mask.png'), mk)
            cv2.imwrite(os.path.join(a.save_masks, r0['name'] + '_conf.png'),
                        (np.clip(prob, 0, 1) * 255).astype(np.uint8))
            print('  留出图 %s（fold 模型）mask 已存：覆盖率 %.3f%%' % (r0['name'], 100 * (mk > 0).mean()))

    npz = os.path.join(a.out, a.tag + '.npz')
    # sklearn 的 coefs_[i] 已经是 (n_in, n_out)，前向就是 X @ w + b，不要再转置
    w = [mlp.coefs_[0], mlp.intercepts_[0], mlp.coefs_[1], mlp.intercepts_[1],
         mlp.coefs_[2], mlp.intercepts_[2]]
    np.savez(npz, **{'w1': w[0], 'b1': w[1], 'w2': w[2], 'b2': w[3], 'w3': w[4], 'b3': w[5],
                     'mean': mu.astype(np.float32), 'std': sd.astype(np.float32),
                     'k': np.array([a.kernel]), 'thr': np.array([thr])})
    meta = {'tag': a.tag, 'kernel': a.kernel, 'hidden': list(hid), 'params': int(sum(x.size for x in w)),
            'in_image_split': not per_image, 'images': [r0['name'] for r0 in results],
            'agg': agg, 'per_image': rep, 'hard_neg': hard,
            'n_train': int(len(y)), 'label_source': a.src}
    with io.open(os.path.join(a.out, a.tag + '.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print('汇总: bestF1 %.4f (P%.3f/R%.3f) thr %.3f' % (agg['bestF1'], agg['bestP'], agg['bestR'], thr))
    for h in hard:
        print('  难负类 %-40s n=%6d  其中误检 %5d   易负类误检 %5d' %
              (h['image'], h['hard_neg_n'], h['hard_neg_FP'], h['easy_neg_FP']))
    print('已写', npz)


if __name__ == '__main__':
    main()
