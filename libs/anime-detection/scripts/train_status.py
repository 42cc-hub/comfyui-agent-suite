"""看脱离会话跑的训练进度：读日志尾部 + 抓最后一行指标。"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
LOGS = os.path.join(PROJ, 'runs', '_logs')


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if not name:
        for f in sorted(os.listdir(LOGS)) if os.path.isdir(LOGS) else []:
            p = os.path.join(LOGS, f)
            print('%-28s %8.1f KB  %s' % (f, os.path.getsize(p) / 1024.0,
                                          '跑完' if 'epochs completed' in open(p, errors='ignore').read() else '进行中'))
        return
    p = os.path.join(LOGS, name if name.endswith('.log') else name + '.log')
    if not os.path.exists(p):
        print('没有日志：%s' % p); return
    txt = open(p, errors='ignore').read()
    tail = [l for l in txt.splitlines() if l.strip()][-12:]
    print('\n'.join(tail))
    m = re.findall(r'^\s+all\s+.*$', txt, re.M)
    if m:
        print('\n最后一次验证：%s' % m[-1].strip())
    print('\n完成' if 'epochs completed' in txt else '\n仍在跑')


if __name__ == '__main__':
    main()
