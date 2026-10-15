"""检查这个 skill 自身的一致性 —— 改完入口必须跑。

为什么需要：SKILL.md 是"路由器"，它靠 `references/xxx.md`、`scripts/xxx.py` 这些指针把 agent 送到
该读的地方。指针一断，agent 要么撞空、要么自己即兴发挥 —— 这类事故在 2026-10-13 实测到 3 个：
`references/localization_and_masks.md`、`references/underpaint_and_stencil.md`、
`scripts/tools/pattern_fill.py` 三个指针只在发布版（release_src overlay 进仓库）存在，本机 skill 里没有。

检查项：
  1) 所有 md/py 里出现的 references/ scripts/ libs/ models/ 路径是否真实存在（硬失败）
  2) references/ 与 scripts/ 下的文件有没有从没被任何文档提到（提示，不失败）
  3) 入口 frontmatter 必需字段
  4) 入口行数不超过预算（棘轮：只许降）
  5) 入口的一级标题恰好 1 个（围栏代码块里的 # 不算）

用法：python scripts/skill_lint.py [--budget 700]
退出码 0 = 全过；1 = 有硬失败。
"""
import argparse
import io
import os
import re
import sys

# Windows 控制台常是 GBK：收尾那行 ✅ 会让 print 抛 UnicodeEncodeError —— 检查其实已经通过了，
# 却以非 0 退出、看起来像失败。统一转 UTF-8。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
ENTRY = os.path.join(SKILL, 'SKILL.md')
# 少数引用指向 skill 之外：本机共享工具在 `agent_work\pixiv_publishing\pipeline\`，
# 文档里一律写成 `tools/xxx.py`（工作区里 agent_work 那层）。发布版由
# release_build.py 把它们改写成"自带工具"的说明。这两类都按工作区解析，不存在才算断链。
WORKSPACE = r'<你的 ComfyUI 安装目录>'
WORKSPACE_AGENT = os.path.join(WORKSPACE, 'agent_work')
PATH_RE = re.compile(r'\b(?:references|scripts|libs|models|examples)/[A-Za-z0-9_][A-Za-z0-9_./\-]*'
                     r'\.(?:md|py|mjs|json|txt|pth|safetensors)')
EXT_RE = re.compile(r'\b(?:agent_work|pixiv_publishing)/[A-Za-z0-9_][A-Za-z0-9_./\-]*\.(?:py|md)')
REQUIRED_FM = ('name', 'description', 'allowed-tools', 'license')


def read(p):
    return io.open(p, encoding='utf-8').read().replace('\r\n', '\n')


def scan_docs():
    out = []
    for root, dirs, files in os.walk(SKILL):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'models')]
        for fn in files:
            if fn.endswith(('.md', '.py', '.mjs', '.json')):
                p = os.path.join(root, fn)
                if os.path.abspath(p) == os.path.abspath(__file__):
                    continue                    # 检查器自己的说明里有示例路径，别当引用
                out.append(p)
    return out


def strip_fences(text):
    """去掉围栏代码块内容，用于标题/结构检查。"""
    keep, inside = [], False
    for line in text.split('\n'):
        if line.lstrip().startswith('```'):
            inside = not inside
            continue
        if not inside:
            keep.append(line)
    return '\n'.join(keep)


def check_paths(docs):
    missing = {}
    for p in docs:
        rel_owner = os.path.relpath(p, SKILL).replace(os.sep, '/')
        text = read(p)
        cands = [(m, os.path.join(SKILL, m.replace('/', os.sep))) for m in PATH_RE.findall(text)]
        cands += [(m, os.path.join(WORKSPACE_AGENT, m.replace('/', os.sep))) for m in EXT_RE.findall(text)]
        for m, full in cands:
            if os.path.exists(full):
                continue
            missing.setdefault(m, set()).add(rel_owner)
    return missing


def check_orphans(docs):
    body = '\n'.join(read(p) for p in docs)
    orphans = []
    for sub in ('references', 'scripts'):
        for root, dirs, files in os.walk(os.path.join(SKILL, sub)):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for fn in files:
                if not fn.endswith(('.md', '.py', '.mjs', '.json')):
                    continue
                rel = os.path.relpath(os.path.join(root, fn), SKILL).replace(os.sep, '/')
                if rel not in body:
                    orphans.append(rel)
    return sorted(orphans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=580,
                    help='入口行数上限（棘轮：2026-10-13 瘦身后 561 行，只许降不许升）')
    a = ap.parse_args()
    docs = scan_docs()
    entry = read(ENTRY)
    fails, notes = [], []

    missing = check_paths(docs)
    if missing:
        for m in sorted(missing):
            fails.append('指针指向不存在的文件：%s（%s 里引用）' % (m, '、'.join(sorted(missing[m]))))
    n_refs = len(PATH_RE.findall(entry))
    notes.append('入口里有 %d 处路径指针，%d 个去重目标' % (n_refs, len(set(PATH_RE.findall(entry)))))

    fm = entry.split('---')[1] if entry.startswith('---') else ''
    for k in REQUIRED_FM:
        if not re.search(r'(?m)^%s:' % re.escape(k), fm):
            fails.append('frontmatter 缺字段：%s' % k)

    n_lines = len(entry.split('\n'))
    if n_lines > a.budget:
        fails.append('入口 %d 行 > 预算 %d 行（棘轮只许降）' % (n_lines, a.budget))

    body = strip_fences(entry)
    h1 = [l for l in body.split('\n') if re.match(r'^#\s', l)]
    if len(h1) != 1:
        fails.append('入口一级标题 %d 个（应为 1）：%s' % (len(h1), ' | '.join(x[:40] for x in h1[:4])))

    orphans = check_orphans(docs)
    print('skill: %s' % SKILL)
    print('入口 %d 行；文档 %d 个' % (n_lines, len(docs)))
    for n in notes:
        print('  · %s' % n)
    if orphans:
        print('  · 从没被任何文件提到的 %d 个（提示，不算失败）：' % len(orphans))
        for o in orphans:
            print('      ? %s' % o)
    if fails:
        print('\n❌ %d 项硬失败：' % len(fails))
        for f in fails:
            print('   - %s' % f)
        return 1
    print('\n✅ 全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
