"""下载 SAM ViT-B 权重（Apache-2.0），仓库不带权重。

用法：python scripts/fetch_sam.py [--out <repo>/models/sams]
"""
import argparse
import os
import sys

URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
NAME = 'sam_vit_b_01ec64.pth'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  '..', '..', '..', 'skill', 'models', 'sams'))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, NAME)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print('已存在，跳过 %s' % dst)
        return 0
    import urllib.request
    print('下载 %s → %s（约 375 MB）' % (URL, dst))
    try:
        urllib.request.urlretrieve(URL, dst)
    except Exception as e:                                      # noqa: BLE001
        print('失败：%s\n（可手动下载后放到 %s）' % (e, out))
        return 1
    print('完成。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
