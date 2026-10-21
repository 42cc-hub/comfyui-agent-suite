"""--det-person-weights opt-in 的最小验收：①默认路径与改动前逐字段一致 ②带参时 person 只来自人腿、hand/foot 不变。"""
import glob, json, os, subprocess, sys, tempfile

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LOC = os.path.join(SUITE, 'scripts', 'locate.py')
IMG = os.path.join(SUITE, 'data', 'hand_gt_multi', 'images')
BASE = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'variants', 'preds_multi60_v1_now')
PERSON_W = os.path.join(SUITE, 'runs', 'm2a_det_person_hybrid', 'weights', 'best.pt')
OUT = os.path.join(SUITE, '..', '..', 'team', 'reports', 'm2a-verifier', 'person_leg')


def run(src, extra, outdir):
    os.makedirs(outdir, exist_ok=True)
    cmd = [PY, LOC, '--src', src, '--out', outdir, '--conf', '0.3', '--imgsz', '1280', '--no-vitpose'] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if p.returncode != 0:
        print('RC', p.returncode, (p.stderr or '')[-600:])
    return p.returncode


def cls_boxes(rec, field='boxes_all'):
    d = {'person': [], 'hand': [], 'foot': []}
    for b in rec.get(field) or []:
        c = b.get('cls')
        if c in d:
            d[c].append(tuple(round(float(v), 1) for v in b['box']))
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    stems = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(IMG, '*.png')))]
    stems = [s for s in stems if os.path.exists(os.path.join(BASE, s + '.json'))][:6]
    src = os.path.join(OUT, 'src6')
    os.makedirs(src, exist_ok=True)
    for s in stems:
        shutil_src = os.path.join(IMG, s + '.png')
        dst = os.path.join(src, s + '.png')
        if not os.path.exists(dst):
            os.link(shutil_src, dst)
    print('images', len(stems))
    rc1 = run(src, [], os.path.join(OUT, 'default'))
    rc2 = run(src, ['--det-person-weights', PERSON_W], os.path.join(OUT, 'with_leg'))
    same_bytes = diff = 0
    hand_foot_diff = []
    person_from_leg = 0
    for s in stems:
        a = json.load(open(os.path.join(OUT, 'default', s + '.json'), encoding='utf-8'))
        b = json.load(open(os.path.join(OUT, 'with_leg', s + '.json'), encoding='utf-8'))
        old = json.load(open(os.path.join(BASE, s + '.json'), encoding='utf-8'))
        va = {k: v for k, v in a.items() if k not in ('chain',)}
        vo = {k: v for k, v in old.items() if k not in ('chain',)}
        if json.dumps(va, sort_keys=True, ensure_ascii=False) == json.dumps(vo, sort_keys=True, ensure_ascii=False):
            same_bytes += 1
        else:
            diff += 1
        ca, cb = cls_boxes(a), cls_boxes(b)
        if ca['hand'] != cb['hand'] or ca['foot'] != cb['foot']:
            hand_foot_diff.append(s)
        if cb['person'] and ('M2a.hybrid_person' in (b.get('chain') or [])):
            person_from_leg += 1
        print('%-46s person %d->%d hand %d->%d foot %d->%d chain_leg=%s' % (
            s[:46], len(ca['person']), len(cb['person']), len(ca['hand']), len(cb['hand']),
            len(ca['foot']), len(cb['foot']), 'M2a.hybrid_person' in (b.get('chain') or [])))
    print('A 默认 vs 改动前 一致 %d / 不一致 %d (rc %s) | B hand/foot 变化图 %d | 带参且有 person 的图 %d (rc %s)' % (
        same_bytes, diff, rc1, len(hand_foot_diff), person_from_leg, rc2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
