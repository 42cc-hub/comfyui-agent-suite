"""install.py —— 把本仓库部署成"能被 agent 用的 skill"。

做四件事（都可单独关掉）：
  1. 把 `skill/` 装到 DSH skills 目录（默认 `~/.dsh/skills/comfyui-image-generation`，`--dest` 可改）；
  2. 生成 `skill/config/paths.json`：把 `libs/` 下的子库路径、ComfyUI 地址写进去（密钥不写，走 keys.json）；
  3. 可选下载模型（`--with-models`：SAM ViT-B + WD14）；
  4. 打印自检命令（安装成功探针）。

设计原则：**不覆盖已有东西除非 `--force`**；路径一律用绝对路径写进 paths.json（skill 被拷到别处也能找到 libs）。

用法：
    python install.py                      # 装 skill + 写 paths.json
    python install.py --with-models        # 顺便下 SAM 与 WD14
    python install.py --dest D:/some/skills/comfyui-image-generation --force
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEST = os.path.join(os.path.expanduser('~'), '.dsh', 'skills', 'comfyui-image-generation')


def load_libs():
    p = os.path.join(HERE, 'libs.json')
    return json.load(io.open(p, encoding='utf-8'))['libs'] if os.path.exists(p) else []


def write_paths(dest, comfyui):
    """把子库绝对路径写进 <dest>/config/paths.json（保留已有的用户键）。"""
    libs = load_libs()
    paths = {}
    for lib in libs:
        p = os.path.join(HERE, lib['path'].replace('/', os.sep))
        entry = os.path.join(HERE, lib.get('entry', '').replace('/', os.sep)) if lib.get('entry') else ''
        # 写"能被 import 的入口目录"（多数子库是 <lib>/scripts），没有就用库根
        paths[lib['env']] = entry if entry and os.path.isdir(entry) else p
    paths['ANIME_DET_SUITE'] = os.path.join(HERE, 'libs', 'anime-detection')
    paths['WD14_MODEL_DIR'] = os.path.join(HERE, 'libs', 'wd14-gate', 'models')
    paths['SAM_CKPT'] = os.path.join(HERE, 'skill', 'models', 'sams', 'sam_vit_b_01ec64.pth')
    paths['COMFYUI_BASE'] = comfyui
    cfgdir = os.path.join(dest, 'config')
    os.makedirs(cfgdir, exist_ok=True)
    out = os.path.join(cfgdir, 'paths.json')
    if os.path.exists(out):                                  # 合并，别把用户手写的键冲掉
        try:
            paths = dict(json.load(io.open(out, encoding='utf-8')), **paths)
        except Exception:                                    # noqa: BLE001
            pass
    io.open(out, 'w', encoding='utf-8', newline='\n').write(json.dumps(paths, ensure_ascii=False, indent=2) + '\n')
    keys = os.path.join(cfgdir, 'keys.example.json')
    if os.path.exists(keys) and not os.path.exists(os.path.join(cfgdir, 'keys.json')):
        shutil.copy2(keys, os.path.join(cfgdir, 'keys.json'))
        print('  · 已生成 config/keys.json（空模板；它在 .gitignore 里，可安全填你自己的 key）')
    print('  · paths.json → %s（%d 个键）' % (out, len(paths)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=DEFAULT_DEST)
    ap.add_argument('--comfyui', default=os.environ.get('COMFYUI_BASE', 'http://127.0.0.1:8188'))
    ap.add_argument('--mode', choices=['copy', 'link'], default=None, help='默认 Windows 拷贝、其它平台软链')
    ap.add_argument('--force', action='store_true', help='目标已存在时覆盖（默认拒绝，不动已有东西）')
    ap.add_argument('--with-models', action='store_true', help='顺便下载 SAM 与 WD14 模型')
    ap.add_argument('--skip-skill', action='store_true', help='只写 paths.json，不装 skill')
    a = ap.parse_args()

    dest = os.path.abspath(a.dest)
    src = os.path.join(HERE, 'skill')
    mode = a.mode or ('copy' if os.name == 'nt' else 'link')

    print('== 安装 %s' % ('（跳过）' if a.skip_skill else 'skill → %s（%s）' % (dest, mode)))
    if not a.skip_skill:
        if os.path.exists(dest) and not a.force:
            print('  · 目标已存在，只更新 config/paths.json（要重装加 --force）')
        else:
            if os.path.exists(dest):
                shutil.rmtree(dest)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if mode == 'link':
                os.symlink(src, dest, target_is_directory=True)
            else:
                shutil.copytree(src, dest)
            print('  · 已写入 %s' % dest)

    print('== 生成 paths.json')
    write_paths(dest, a.comfyui)

    if a.with_models:
        print('== 下载模型')
        for cmd in ([sys.executable, '-X', 'utf8', os.path.join(HERE, 'libs', 'sam-matting', 'scripts', 'fetch_sam.py')],
                    [sys.executable, '-X', 'utf8', os.path.join(HERE, 'libs', 'wd14-gate', 'scripts', 'fetch_wd14_model.py')]):
            if os.path.exists(cmd[3] if len(cmd) > 3 else cmd[-1]):
                print('  $ %s' % ' '.join(cmd[-1:]))
                subprocess.call(cmd)
            else:
                print('  · 跳过（脚本不存在）：%s' % cmd[-1])

    print('\n== 下一步 ==')
    print('  1) 校验子库路径：python "%s"   （应逐行打印出每个子库的绝对路径）'
          % os.path.join(HERE, 'libs', 'anime-detection', 'scripts', '_libs.py'))
    print('  2) 拿自训权重：python "%s" --list' % os.path.join(HERE, 'skill', 'scripts', 'fetch_weights.py'))
    print('  3) 跑安装成功探针：见 %s' % os.path.join(HERE, 'skill', 'examples', 'hutao_furina_two_girls.md'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
