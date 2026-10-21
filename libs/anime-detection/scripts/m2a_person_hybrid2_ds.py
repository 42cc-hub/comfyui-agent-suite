"""ds_person_hybrid2：ds_det + ds_det_fix + 伪标 person 图（person 真值 + hand/foot/head 伪标）×3 份。

输入 data/ds_person_pl（m2a_person_pl.py 产出，已丢弃无 hand/foot 伪标的图）。
产物 data/ds_person_pl/w3 + data/det_person_hybrid2.yaml
"""
import io, json, os, shutil, sys

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(SUITE, 'data')
PL = os.path.join(D, 'ds_person_pl')
OUT = os.path.join(PL, 'w3')
YAML = os.path.join(D, 'det_person_hybrid2.yaml')
BASE = [os.path.join(D, 'ds_det', 'images', 'train'),
        os.path.join(D, 'ds_det_fix', 'images'),
        os.path.join(PL, 'images', 'train')]


def main():
    src_i, src_l = os.path.join(PL, 'images', 'train'), os.path.join(PL, 'labels', 'train')
    files = sorted(f for f in os.listdir(src_i) if f.lower().endswith(('.jpg', '.png')))
    if not files:
        print('NO PSEUDO LABELS'); return 2
    want = len(files) * 3
    have = len(os.listdir(os.path.join(OUT, 'images'))) if os.path.isdir(os.path.join(OUT, 'images')) else 0
    if have != want:
        if os.path.isdir(OUT):
            shutil.rmtree(OUT)
        os.makedirs(os.path.join(OUT, 'images'))
        os.makedirs(os.path.join(OUT, 'labels'))
        for f in files:
            stem, ext = os.path.splitext(f)
            lab = os.path.join(src_l, stem + '.txt')
            for k in (1, 2, 3):
                dst = os.path.join(OUT, 'images', '%s__p%d%s' % (stem, k, ext))
                if not os.path.exists(dst):
                    os.link(os.path.join(src_i, f), dst)
                if os.path.exists(lab):
                    shutil.copyfile(lab, os.path.join(OUT, 'labels', '%s__p%d.txt' % (stem, k)))
    train_lines = BASE + [os.path.join(OUT, 'images')]
    yml = ('# ds_det_person_hybrid2: 基础集原标签 + 伪标 person 图 x3\npath: %s\ntrain:\n%sval: %s\nnc: 4\nnames: [person, hand, foot, head]\n'
           % (D, ''.join('  - %s\n' % t for t in train_lines), os.path.join(D, 'ds_det', 'images', 'val')))
    io.open(YAML, 'w', encoding='utf-8').write(yml)
    stats = {'pseudo_images': len(files), 'w3_images': want,
             'base': {'ds_det_train': 792, 'ds_det_fix': 537}, 'total': 792 + 537 + len(files) + want}
    json.dump(stats, io.open(os.path.join(PL, 'ds_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
