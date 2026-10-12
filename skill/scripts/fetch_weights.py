"""fetch_weights.py —— 下载自训权重（仓库里不带权重），并校验 sha256。

权重清单在 `weights/manifest.json`（名字、任务、配置、指标、大小、sha256）。
三种来源，按顺序尝试：
  1. `--from-dir <目录>`：从本地目录里按 `<name>.best.pt` 找（自己训练完的人用这个）；
  2. `--from-release <owner/repo@tag>`：从 GitHub Release 附件下（默认读环境变量 `SUITE_RELEASE`）；
  3. 都不可用时打印清单并退出，让人自己下（脚本不静默失败）。

落到 `<repo>/runs/<name>/weights/best.pt` —— 与套件里 `locate.py` 等脚本的默认路径一致。

用法：
    python fetch_weights.py --list
    python fetch_weights.py --from-dir D:/my_runs
    python fetch_weights.py --from-release 42cc-hub/comfyui-agent-suite@weights-v1
"""
import argparse
import hashlib
import io
import json
import os
import sys

# Windows 中文控制台默认 GBK：输出被重定向（写日志 / CI / 传管道）时，打印 ✓/✗/⚠ 会
# UnicodeEncodeError 直接崩。把不可编码字符降级为 '?'，别让编码问题吃掉一次成功的下载。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors='replace')
    except Exception:                                       # noqa: BLE001
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # <repo>/skill/scripts → <repo>
MANIFEST = os.path.join(ROOT, 'weights', 'manifest.json')
# 权重落到**套件自己的项目根**下（locate.py 等按 <suite>/runs/<name>/weights/best.pt 找）
LIB = os.path.join(ROOT, 'libs', 'anime-detection')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def target(name):
    return os.path.join(LIB, 'runs', name, 'weights', 'best.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--from-dir')
    ap.add_argument('--from-release', default=os.environ.get('SUITE_RELEASE'))
    ap.add_argument('--only', nargs='+', help='只处理这些权重名')
    a = ap.parse_args()

    if not os.path.exists(MANIFEST):
        raise SystemExit('找不到 %s' % MANIFEST)
    man = json.load(io.open(MANIFEST, encoding='utf-8'))
    items = [w for w in man['weights'] if not a.only or w['name'] in a.only]

    if a.list:
        print('%-34s %-12s %8s  %s' % ('name', 'task', 'size', 'desc'))
        for w in items:
            print('%-34s %-12s %7.1fM  %s' % (w['name'], w['task'], w['size'] / 1e6, w['desc']))
        print('\n共 %d 个，%.1f MB' % (len(items), sum(w['size'] for w in items) / 1e6))
        return 0

    ok = miss = 0
    for w in items:
        dst = target(w['name'])
        if os.path.exists(dst) and sha256(dst) == w['sha256']:
            print('✓ 已存在且校验一致：%s' % w['name'])
            ok += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        got = False
        if a.from_dir:
            src = os.path.join(a.from_dir, w['file'])
            if not os.path.exists(src):
                src = os.path.join(a.from_dir, w['name'], 'weights', 'best.pt')
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)
                got = True
        if not got and a.from_release:
            import urllib.request
            url = 'https://github.com/%s/releases/download/%s/%s' % (
                a.from_release.split('@')[0], a.from_release.split('@')[-1], w['file'])
            print('下载 %s' % url)
            try:
                urllib.request.urlretrieve(url, dst)
                got = True
            except Exception as e:                              # noqa: BLE001
                print('  失败：%s' % e)
        if not got:
            miss += 1
            print('✗ 没拿到 %s（用 --from-dir 指向你的 runs 目录，或 --from-release owner/repo@tag）' % w['name'])
            continue
        h = sha256(dst)
        if h != w['sha256']:
            print('⚠ %s 校验不一致（文件已留在 %s，请核对来源）' % (w['name'], dst))
        else:
            print('✓ %s' % w['name'])
            ok += 1
    print('\n完成：%d 个就位，%d 个缺失' % (ok, miss))
    return 0 if miss == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
