"""下载 WD14 tagger 模型（onnx + 标签 csv）到本地，仓库不带权重。

用法：
    python scripts/fetch_wd14_model.py [--out ../models] [--repo SmilingWolf/wd-v1-4-convnextv2-tagger-v2]

下完把 `--out` 目录写进 config/paths.json 的 WD14_MODEL_DIR，或设同名环境变量。
"""
import argparse
import os
import sys

FILES = ('model.onnx', 'selected_tags.csv')
URL = 'https://huggingface.co/{repo}/resolve/main/{fn}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
    ap.add_argument('--repo', default='SmilingWolf/wd-v1-4-convnextv2-tagger-v2')
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    import urllib.request
    for fn in FILES:
        dst = os.path.join(out, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print('已存在，跳过 %s' % dst)
            continue
        url = URL.format(repo=a.repo, fn=fn)
        print('下载 %s → %s' % (url, dst))
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:                                  # noqa: BLE001
            print('失败：%s\n（若本机走代理，请先设 HTTPS_PROXY，或手动下载后放到 %s）' % (e, out))
            return 1
    print('完成。把这一行写进 config/paths.json：\n  "WD14_MODEL_DIR": "%s"' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
