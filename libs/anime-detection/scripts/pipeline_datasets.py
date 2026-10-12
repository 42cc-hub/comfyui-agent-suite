"""一条命令跑完"从预标注到数据集"的链路（供长跑/夜间串联用）。

顺序（每一步都幂等，可重复跑）：
  1. `export_labels.py`   —— det/pose/pose17/face 标签导出 + cls 切分（含过采样；cls 需人工标签）
  2. `build_dataset.py`   —— 拷图 + 按图划分 + 写 yaml（det / pose / pose17 / face）
  3. （可选 `--train m2a`）`train_detached.py` 启动 M2a 训练（脱离会话，防回收）

为什么单独有这个脚本：夜里我是"上一个跑完再手动起下一个"，中间容易断；
串成一条链路后可以整段无人值守，且每步的产物都在 `data/` 下可核对。

用法：python scripts/pipeline_datasets.py [--train m2a|none]
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
PY_A = r'python'
PY_M = r'python'


def run(cmd, tag):
    print('\n=== %s ===' % tag, flush=True)
    r = subprocess.run(cmd, cwd=PROJ, capture_output=True, text=True, encoding='utf-8', errors='replace')
    for line in (r.stdout or '').splitlines():
        if any(k in line for k in ('数据集', 'posed', 'pose/', 'face', 'det/', 'cls', '划分', 'error', 'Error')):
            print('  ' + line.strip())
    if r.returncode != 0:
        print('  !! 退出码 %d\n%s' % (r.returncode, (r.stderr or '')[-500:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='none', choices=['none', 'm2a'])
    ap.add_argument('--det-epochs', type=int, default=150)
    ap.add_argument('--det-imgsz', type=int, default=960)
    a = ap.parse_args()
    run([PY_A, '-X', 'utf8', os.path.join(HERE, 'export_labels.py')], '导出标签')
    run([PY_A, '-X', 'utf8', os.path.join(HERE, 'build_dataset.py')], '构建数据集')
    if a.train == 'm2a':
        # 名字带上数据规模，便于在 runs/ 里区分（夜里会跑好几版）
        n_img = 0
        try:
            n_img = len([f for f in os.listdir(os.path.join(PROJ, 'data', 'prelabel')) if f.endswith('.json')])
        except Exception:
            pass
        run([PY_M, '-X', 'utf8', os.path.join(HERE, 'train_detached.py'),
             '--task', 'det', '--data', os.path.join(PROJ, 'data', 'ds_det', 'det_body.yaml'),
             '--model', 'yolo11n.pt', '--epochs', str(a.det_epochs), '--imgsz', str(a.det_imgsz),
             '--name', 'm2a_det_%dimg_%d' % (n_img, a.det_imgsz)], '启动 M2a 训练')
    print('\n链路完成。')


if __name__ == '__main__':
    main()
