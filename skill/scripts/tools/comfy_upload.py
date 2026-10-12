"""comfy_upload.py —— 把本地图片上传到 ComfyUI 的 input 目录（工作流里 `LoadImage` 要用）。

    python comfy_upload.py IMG.png [IMG2.png ...] [--sub dsh_ctrl] [--base http://127.0.0.1:8188]

为什么需要它：ComfyUI 的 `LoadImage` 只认**它自己 input 目录里**的文件名。要让"局部重绘"这类
工作流跑起来（它们必须读一张底图和一张掩版），就得先把文件放进去。本脚本只做这一件事，
返回可直接写进工作流的名字（`<sub>/<filename>`）。

注意：`--sub` 下的文件会留在 ComfyUI 的 input 里（不自动清）；不需要了就自己删。
"""
import argparse
import os
import sys
import urllib.request
import uuid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--sub', default='dsh_ctrl', help='放到 input 下的哪个子目录')
    ap.add_argument('--base', default=os.environ.get('COMFY_BASE', 'http://127.0.0.1:8188'))
    ap.add_argument('--overwrite', action='store_true', default=True)
    a = ap.parse_args()

    names = []
    for p in a.images:
        if not os.path.exists(p):
            raise SystemExit('找不到文件：%s' % p)
        fn = os.path.basename(p)
        with open(p, 'rb') as fh:
            data = fh.read()
        boundary = '----dsh%d' % uuid.uuid4().int
        body = b''
        body += ('--%s\r\nContent-Disposition: form-data; name="image"; filename="%s"\r\n'
                 'Content-Type: application/octet-stream\r\n\r\n' % (boundary, fn)).encode('utf-8')
        body += data + b'\r\n'
        body += ('--%s\r\nContent-Disposition: form-data; name="subfolder"\r\n\r\n%s\r\n' % (boundary, a.sub)).encode('utf-8')
        body += ('--%s\r\nContent-Disposition: form-data; name="type"\r\n\r\ninput\r\n' % boundary).encode('utf-8')
        # 必须显式 overwrite=true：否则 ComfyUI 遇到同名文件会存成 "xxx (1).png"，
        # 而工作流里引用的是固定名字 ⇒ 会静默用到上一轮的旧图（探针实测踩到过）。
        if a.overwrite:
            body += ('--%s\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n' % boundary).encode('utf-8')
        body += ('--%s--\r\n' % boundary).encode('utf-8')
        req = urllib.request.Request(a.base.rstrip('/') + '/upload/image', data=body,
                                     headers={'Content-Type': 'multipart/form-data; boundary=%s' % boundary})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                import json
                resp = json.loads(r.read().decode('utf-8', 'ignore'))
            name = '%s/%s' % (resp.get('subfolder') or a.sub, resp.get('name') or fn)
        except Exception as e:                                    # noqa: BLE001
            raise SystemExit('上传失败：%s\n（ComfyUI 地址用 --base 或 COMFY_BASE 指定；确认服务在跑）' % e)
        names.append(name)
        print('↑ %s → %s' % (p, name))
    print('\n工作流里这样引用：')
    for n in names:
        print('  "image": "%s"' % n)
    return 0


if __name__ == '__main__':
    sys.exit(main())
