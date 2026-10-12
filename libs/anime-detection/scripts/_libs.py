"""子库路径解析 —— 让仓库里的脚本找到 `libs/` 下的兄弟组件，**不写死任何绝对路径**。

解析顺序（先命中先用）：
  1. 环境变量（下表第二列）
  2. `skill/config/paths.json`（部署时由 `install.py` 生成；也可手写）
  3. 源码树里的相对位置（`<repo>/libs/<lib>`）
  4. 常见安装位置（`~/.dsh/skills/comfyui-image-generation/../..` 等）

用法（脚本开头两行搞定）：

    from _libs import add_lib
    add_lib("wd14-gate")          # 把该子库的 scripts/ 加进 sys.path
    from wd14_gate import Tagger

新增子库时只要在 `LIBS` 里加一行。
"""
import json
import os
import sys

# name -> (环境变量名, 相对仓库根的路径)
LIBS = {
    'wd14-gate': ('WD14_GATE', 'libs/wd14-gate/scripts'),
    'sam-matting': ('SAM_MATTING', 'libs/sam-matting/scripts'),
    'blender-controlnet': ('BLENDER_CONTROLNET', 'libs/blender-controlnet/scripts'),
    'anime-detection': ('ANIME_DET_SUITE', 'libs/anime-detection'),
    'human-canvas': ('HUMAN_CANVAS', 'libs/human-canvas'),
    'comic-layout': ('COMIC_LAYOUT', 'libs/comic-layout/scripts'),
}


def repo_root(start=None):
    """从 start（默认本文件）向上找同时含 `skill/` 与 `libs/` 的目录（源码树根）。"""
    d = os.path.dirname(os.path.abspath(start or __file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(d, 'libs')) and os.path.isdir(os.path.join(d, 'skill')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _config_dirs():
    """可能放着 paths.json 的目录（本脚本同目录、skill/config、仓库根的 config）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    out = [here, os.path.join(here, '..'), os.path.join(here, '..', '..'),
           os.path.join(here, 'config'), os.path.join(here, '..', 'config'), os.path.join(here, '..', '..', 'config')]
    root = repo_root()
    if root:
        out += [os.path.join(root, 'skill', 'config'), os.path.join(root, 'config')]
    return [os.path.normpath(p) for p in out]


def load_paths():
    """读 paths.json（第一个找到的）；返回 dict（失败给空 dict）。"""
    for d in _config_dirs():
        p = os.path.join(d, 'paths.json')
        if os.path.exists(p):
            try:
                return json.load(open(p, encoding='utf-8'))
            except Exception:                                     # noqa: BLE001
                return {}
    return {}


def find_lib(name):
    """返回该子库的可用路径；找不到给 None（调用方自己报错，别静默）。"""
    if name not in LIBS:
        raise KeyError('未登记的子库：%s（可用：%s）' % (name, ', '.join(sorted(LIBS))))
    env_name, rel = LIBS[name]
    cands = [os.environ.get(env_name), (load_paths() or {}).get(env_name)]
    root = repo_root()
    if root:
        cands.append(os.path.join(root, rel))
    cands += [
        os.path.join(os.path.expanduser('~'), '.dsh', 'skills', 'comfyui-image-generation', '..', '..', rel),
        os.path.join(os.path.expanduser('~'), 'comfyui_agent_suite', rel),
    ]
    for c in cands:
        if c and os.path.exists(os.path.abspath(c)):
            return os.path.abspath(c)
    return None


def add_lib(name):
    """把子库路径插到 sys.path[0]；找不到时**大声报错**并给出可操作提示。"""
    p = find_lib(name)
    if not p:
        env_name, rel = LIBS[name]
        raise SystemExit(
            '找不到子库 %s。三种修法：① 设环境变量 %s=<路径>；'
            '② 在 skill/config/paths.json 里写 {"%s": "<路径>"}；'
            '③ 把仓库 clone 完整（它应在 %s）。' % (name, env_name, env_name, rel))
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


if __name__ == '__main__':
    for k in LIBS:
        print('%-20s %s' % (k, find_lib(k) or '（未找到）'))
