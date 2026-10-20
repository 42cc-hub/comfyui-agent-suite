"""fetch_weights.py —— 下载自训权重（仓库里不带权重），并校验 sha256。

权重清单在 `weights/manifest.json`（名字、任务、配置、指标、大小、sha256）。
三种来源，按顺序尝试：
  1. `--from-dir <目录>`：从本地目录里按 `<name>.best.pt` 找（自己训练完的人用这个）；
  2. `--from-release <owner/repo@tag[,tag2,...]>`：从 GitHub Release 附件下，tag 可给多个（逗号分隔）
     并按顺序尝试 —— 因为 Release 是**增量式**的（v1 11 个、v2 4 个、v3 2 个），只写一个 tag 会让
     其余权重的 URL 404；全失败才报缺失（默认读环境变量 `SUITE_RELEASE`）；
  3. 都不可用时打印清单并退出，让人自己下（脚本不静默失败）。

两种条目形态：
  * 单文件（默认）：落到 `<libs/anime-detection>/runs/<name>/weights/best.pt`，与 `locate.py` 等脚本的默认路径一致；
  * **目录型**（`"files": [...]`，例如微调的 ViTPose，是一整个 HF 目录）：逐文件落到
    `<libs/anime-detection>/<dest>/<as>`；`as` 是写盘文件名（Release 附件名是 `file`，两者刻意分开，
    否则 `config.json` 这种通用名在同一个 Release 里会撞名）。

用法：
    python fetch_weights.py --list
    python fetch_weights.py --from-dir D:/my_runs
    python fetch_weights.py --from-release 42cc-hub/comfyui-agent-suite@weights-v1,weights-v2,weights-v3
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


def target(name, w=None):
    """写盘路径：目录型条目看 `dest`/`as`，单文件条目固定落在 runs/<name>/weights/best.pt。"""
    if w and w.get('dest'):
        return os.path.join(LIB, w['dest'].replace('/', os.sep), w.get('as') or w['file'])
    return os.path.join(LIB, 'runs', name, 'weights', 'best.pt')


def entries(w):
    """把一条清单统一成 [(附件名, 写盘路径, size, sha256)]，屏蔽"单文件 / 目录型"的差别。"""
    if w.get('files'):
        return [(f['file'], target(w['name'], dict(w, **{'as': f.get('as') or f['file']})),
                 f.get('size'), f['sha256']) for f in w['files']]
    return [(w['file'], target(w['name'], w), w.get('size'), w['sha256'])]


def local_cands(from_dir, w, asset, dst):
    """`--from-dir` 的查找顺序：**先按仓库里的相对位置**（from_dir = 套件根目录即可），
    再退回"平坦的 runs 目录"（自己训练完的人多半是这种）。"""
    base = os.path.basename(dst)
    if w.get('dest'):
        return [os.path.join(from_dir, w['dest'].replace('/', os.sep), base),
                os.path.join(from_dir, base),
                os.path.join(from_dir, w['name'], base)]
    return [os.path.join(from_dir, 'runs', w['name'], 'weights', 'best.pt'),
            os.path.join(from_dir, asset),
            os.path.join(from_dir, base),
            os.path.join(from_dir, w['name'], 'weights', 'best.pt')]


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
        print('%-34s %-16s %8s  %s' % ('name', 'task', 'size', 'desc'))
        for w in items:
            print('%-34s %-16s %7.1fM  %s' % (w['name'], w['task'], w['size'] / 1e6, w['desc']))
        print('\n共 %d 个，%.1f MB' % (len(items), sum(w['size'] for w in items) / 1e6))
        return 0

    ok = miss = 0
    for w in items:
        todo = entries(w)
        if all(os.path.exists(p) and sha256(p) == h for _f, p, _s, h in todo):
            print('✓ 已存在且校验一致：%s' % w['name'])
            ok += 1
            continue
        got_all = True
        for asset, dst, _size, want in todo:
            if os.path.exists(dst) and sha256(dst) == want:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            got = False
            if a.from_dir:
                for cand in local_cands(a.from_dir, w, asset, dst):
                    if os.path.exists(cand):
                        import shutil
                        shutil.copy2(cand, dst)
                        got = True
                        break
            if not got and a.from_release:
                import urllib.request
                repo, _at, tag_part = a.from_release.partition('@')
                tags = [t.strip() for t in tag_part.split(',') if t.strip()]
                for tag in tags:                    # 增量式 Release：按顺序试，第一个拿到的为准
                    url = 'https://github.com/%s/releases/download/%s/%s' % (repo, tag, asset)
                    print('下载 %s' % url)
                    try:
                        urllib.request.urlretrieve(url, dst)
                        got = True
                        break
                    except Exception as e:                              # noqa: BLE001
                        print('  失败：%s' % e)
            if not got:
                got_all = False
                print('✗ 没拿到 %s（用 --from-dir 指向你的 runs 目录，或 --from-release owner/repo@tag）' % asset)
                continue
            h = sha256(dst)
            if h != want:
                print('⚠ %s 校验不一致（文件已留在 %s，请核对来源）' % (os.path.basename(dst), dst))
                got_all = False
        if got_all:
            ok += 1
        else:
            miss += 1
    print('\n完成：%d 个就位，%d 个缺失' % (ok, miss))
    return 0 if miss == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
