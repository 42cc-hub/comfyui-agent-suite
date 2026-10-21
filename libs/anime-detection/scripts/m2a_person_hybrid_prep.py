"""ds_det_person_hybrid：ds_det(train) + ds_det_fix + ds_person/train + ds_person 每张复制 3 份。

权重等效 = 复制 3 份（不改 loss 权重）。person 图已有 4 个角度（a0/a30/a60/a90），
这里对 184 张各复制 3 份（__b1/__b2/__b3），所以 person 类相对基础集是 3 倍。
"""
import io, json, os, shutil, sys

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(SUITE, 'data')
OUT = os.path.join(D, 'ds_person', 'w3')
YAML = os.path.join(D, 'det_person_hybrid.yaml')

SRC_IMG = os.path.join(D, 'ds_person', 'images', 'train')
SRC_LBL = os.path.join(D, 'ds_person', 'labels', 'train')
BASE = [
    os.path.join(D, 'ds_det', 'images', 'train'),
    os.path.join(D, 'ds_det_fix', 'images'),
    SRC_IMG,
]


def count(d):
    n = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    l = len([f for f in os.listdir(d.replace(os.sep + 'images', os.sep + 'labels')) if f.endswith('.txt')])
    return n, l


def main():
    n_img = len([f for f in os.listdir(SRC_IMG) if f.lower().endswith(('.jpg', '.png'))])
    if os.path.isdir(OUT):
        existing = len([f for f in os.listdir(os.path.join(OUT, 'images'))])
    else:
        existing = 0
    if existing != n_img * 3:
        if os.path.isdir(OUT):
            shutil.rmtree(OUT)
        os.makedirs(os.path.join(OUT, 'images'))
        os.makedirs(os.path.join(OUT, 'labels'))
        for f in sorted(os.listdir(SRC_IMG)):
            if not f.lower().endswith(('.jpg', '.png')):
                continue
            stem, ext = os.path.splitext(f)
            lab = os.path.join(SRC_LBL, stem + '.txt')
            for k in (1, 2, 3):
                dst = os.path.join(OUT, 'images', '%s__b%d%s' % (stem, k, ext))
                if not os.path.exists(dst):
                    os.link(os.path.join(SRC_IMG, f), dst)
                if os.path.exists(lab):
                    shutil.copyfile(lab, os.path.join(OUT, 'labels', '%s__b%d.txt' % (stem, k)))
    stats = {}
    for d in BASE + [os.path.join(OUT, 'images')]:
        key = os.path.relpath(d, D)
        lab = d.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)
        nn, nl = count(lab) if os.path.isdir(lab) else (0, 0)
        stats[key] = [len([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]), nl]
    train_lines = [os.path.join(d) for d in BASE] + [os.path.join(OUT, 'images')]
    yml = (
        '# ds_det_person_hybrid: 基础集 hand/foot/head/person + person 专项 3x\n'
        'path: %s\n'
        'train:\n%s\n'
        'val: %s\n'
        'nc: 4\n'
        'names: [person, hand, foot, head]\n'
        % (D, ''.join('  - %s\n' % t for t in train_lines), os.path.join(D, 'ds_det', 'images', 'val'))
    )
    io.open(YAML, 'w', encoding='utf-8').write(yml)
    print('yaml', YAML)
    print(json.dumps(stats, ensure_ascii=False))
    total = sum(v[0] for v in stats.values())
    print('total train images', total)
    json.dump({'stats': stats, 'total': total}, io.open(os.path.join(D, '..', 'runs', '_hybrid_ds.json') if False else os.path.join(D, 'ds_person', 'w3_build.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    sys.exit(main())
