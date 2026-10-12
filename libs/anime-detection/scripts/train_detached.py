"""**独立进程**启动训练（`Start-Process`），日志写文件。

为什么不用 harness 的后台作业：长训练在会话被回收时可能一起没掉（CV 子代理那边实测踩过）。
所以长任务一律：`Start-Process` 脱离会话 + stdout/stderr 重定向到 `runs/_logs/<name>.log`，
之后用 `python scripts/train_status.py <name>` 或直接读日志看进度。

用法：python scripts/train_detached.py --task pose --data <abs yaml> --epochs 100 --name m1_pose_v4
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
LOGS = os.path.join(PROJ, 'runs', '_logs')
PY = r'python'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['det', 'pose', 'cls'])
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', default=None)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', default='-1')
    ap.add_argument('--name', required=True)
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, a.name + '.log')
    cmd = [PY, '-X', 'utf8', os.path.join(HERE, 'train.py'), '--task', a.task, '--data', a.data,
           '--epochs', str(a.epochs), '--imgsz', str(a.imgsz), '--batch', str(a.batch), '--name', a.name]
    if a.model:
        cmd += ['--model', a.model]
    with open(log, 'wb') as fh:
        # DETACHED_PROCESS：脱离当前会话，避免被回收
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=PROJ,
                             creationflags=0x00000008 | 0x00000200)   # DETACHED_PROCESS | NEW_PROCESS_GROUP
    print('已脱离会话启动：pid=%d' % p.pid)
    print('日志：%s' % log)
    print('查看：python scripts/train_status.py %s' % a.name)


if __name__ == '__main__':
    main()
