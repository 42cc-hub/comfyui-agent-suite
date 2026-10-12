"""wd14_gate.py —— 用**本地 WD14 tagger** 做语义闸门（替代不可靠的肤色占比）。

为什么换掉肤色统计：隔着布料的敏感部位，颜色会被布料改变 ⇒ 像素规则抓不到 ✗
（实测：一张"半透明白裙"只有 9% 肤色、被判 SAFE，但人眼一看就不合适 ✗）。
WD14 给的是**语义标签 + rating**，正好对应"有没有 bare crotch / nipples 这类元素" ✓。

模型与标签表直接复用 ComfyUI 那个 custom node 里已经装好的：
    <你的 ComfyUI 安装目录>\\custom_nodes\\comfyui-wd14-tagger\\models\\wd-v1-4-convnextv2-tagger-v2.{onnx,csv}
预处理与官方节点完全一致（缩放→白底方形→float32→RGB→BGR），保证结果可比。

判定（硬边界是**关键点**，不是 WD14 的整体 rating，也不是姿势）：
    rating = 前 general_index 个条目里概率最高者（wd14 的四个 rating 类别：
    general / sensitive / questionable / explicit）
    - 命中关键点标签表 BLOCK_TAGS               -> BLOCK    （不要用视觉工具看）
    - rating 为 explicit 且命中关键点/姿势表      -> BLOCK
    - rating 为 explicit 但两者都不命中           -> REVIEW   （标 escalated：题材过杀，人工过目）
    - 只命中姿势表 POSE_TAGS（衣着完整）          -> REVIEW   （标 pose-level）
    - rating 为 questionable                   -> REVIEW
    - rating 为 sensitive 或命中关注标签表        -> REVIEW   （交给人过目）
    - 否则                                     -> OK

    2026-10-07 变更依据（用户同意，两轮实测）：
      a) 连测 9 张纯面部特写（blush / shy / rolling eyes / angry+咬牙 / orgasm face(ahegao) /
         torogao / empty eyes / crying / naughty tongue），rating 最高只到 questionable，
         **无一到 explicit**；反例是"旗袍 + crotch rope + 绑在大腿的按摩棒"那张：rating=explicit
         却**连一个裸露标签都没有** ⇒ rating 拦的是题材，不是关键点 ⇒ explicit 不再一刀切 BLOCK。
      b) 姿势实测：横一字马那张**衣着完整、rating 只有 sensitive**，却因一个 `spread legs`
         标签被判 BLOCK（用户亲眼复核：没露、尺度比同批其它图更小）⇒ `spread legs` 降为
         **姿势级**（POSE_TAGS）：单独出现不再 BLOCK，只有与关键点同时出现、或 rating=explicit
         时才 BLOCK。同时把 `cameltoe / partially visible vulva / vulva / labia / visible clitoris`
         这些"精确关键点"从 rating 兜底提升为硬 BLOCK。

用法：
    python wd14_gate.py a.png [b.png ...] [--tags]      # --tags 打印全部标签
"""
import argparse
import csv
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

def _models_dir():
    """模型目录解析顺序：环境变量 WD14_MODEL_DIR → 本库 models/ → ComfyUI 自定义节点目录。

    仓库不带权重；先用 `python scripts/fetch_wd14_model.py` 下到 models/。
    """
    env = os.environ.get("WD14_MODEL_DIR")
    if env and os.path.isdir(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "models"), os.path.join(here, "models")):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return r"<你的 ComfyUI 安装目录>\custom_nodes\comfyui-wd14-tagger\models"


NODE_MODELS = _models_dir()
MODEL = "wd-v1-4-convnextv2-tagger-v2"

# 明确的敏感元素（命中即 BLOCK）
BLOCK_TAGS = {
    "nipples", "puffy nipples", "pussy", "penis", "anus", "bare crotch", "crotch",
    "completely nude", "nude", "topless", "bottomless", "presenting",
    "large areolae", "areolae", "clitoris", "testicles", "sex", "cum",
    # 2026-10-07：精确关键点，过去只因不在表里而靠 rating 兜底 ⇒ 提为硬 BLOCK
    "cameltoe", "partially visible vulva", "vulva", "labia", "visible clitoris",
}
# 姿势级标签：**不是**关键点。衣着完整时张开姿势不该被拦 ✗
# 只有与关键点同时出现、或 rating=explicit 时才升级为 BLOCK（见 verdict）。
POSE_TAGS = {
    "spread legs",
}
# 需要人过目的（REVIEW）
WATCH_TAGS = {
    "see-through", "transparent clothing", "sheer", "lingerie", "underwear", "panties",
    "bra", "bikini", "swimsuit", "cleavage", "underboob", "sideboob", "navel", "bare shoulders",
    "bare legs", "thighhighs", "garter belt", "wet clothes", "undressing", "skirt lift",
}
# explicit 场景的复核表：比 BLOCK_TAGS 多收一批"虽非硬关键点、但 explicit 时必须算数"的标签
NUDITY_TAGS = BLOCK_TAGS | {
    "bare butt", "exposed butt", "no panties", "exposed crotch", "exposed breasts",
    "large breasts", "huge breasts",
}
# 历史上 rating 的硬闸；现在只在 verdict() 里作为**辅助**条件使用
RATING_BLOCK = {"explicit"}


class Tagger:
    def __init__(self, models_dir=NODE_MODELS, model=MODEL):
        onnx_path = os.path.join(models_dir, model + ".onnx")
        csv_path = os.path.join(models_dir, model + ".csv")
        if not os.path.exists(onnx_path):
            raise SystemExit("找不到 tagger 模型: " + onnx_path)
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0]
        self.size = self.input.shape[1]
        self.tags, self.general_index, self.character_index = [], None, None
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if self.general_index is None and row[2] == "0":
                    self.general_index = reader.line_num - 2
                elif self.character_index is None and row[2] == "4":
                    self.character_index = reader.line_num - 2
                self.tags.append(row[1].replace("_", " "))

    def run(self, path, threshold=0.35):
        im = Image.open(path).convert("RGB")
        ratio = float(self.size) / max(im.size)
        new = tuple(int(x * ratio) for x in im.size)
        im = im.resize(new, Image.LANCZOS)
        square = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        square.paste(im, ((self.size - new[0]) // 2, (self.size - new[1]) // 2))
        a = np.array(square).astype(np.float32)[:, :, ::-1][None, ...]
        out = self.sess.run([self.sess.get_outputs()[0].name], {self.input.name: a})[0][0]
        res = list(zip(self.tags, out.tolist()))
        ratings = res[: self.general_index]
        rating = max(ratings, key=lambda x: x[1])[0]
        general = [t for t, p in res[self.general_index:self.character_index] if p > threshold]
        return rating, general


def verdict(rating, tags):
    # 2026-09-24 用户同意放宽：硬边界是**关键点**，不是 WD14 的整体 rating。
    # 分类器对 BDSM 语境偏保守（实测一圈"绳缚+强迫+居家薄衣"全判 questionable，
    # 关键点标签一个都没有、人眼看也正常）⇒ questionable 不再一刀切 BLOCK ✗。
    # 2026-10-07 延伸到 explicit 与姿势标签，见文件头"变更依据"。
    t = set(tags)
    if BLOCK_TAGS & t:
        return "BLOCK"
    if rating == "explicit":
        return "BLOCK" if ((NUDITY_TAGS | POSE_TAGS) & t) else "REVIEW"
    if rating == "questionable":
        return "REVIEW"
    if rating == "sensitive" or (WATCH_TAGS & t):
        return "REVIEW"
    return "OK"


def overkill(rating, tags):
    """rating 被顶到 explicit、却一个关键点/姿势标签都没有 ⇒ 题材过杀，降级交人过目。"""
    t = set(tags)
    return rating == "explicit" and not ((NUDITY_TAGS | POSE_TAGS) & t) and not (BLOCK_TAGS & t)


def pose_level(tags):
    """只凭姿势标签（无关键点、非 explicit）被判 REVIEW 的情形，用于输出留痕。"""
    t = set(tags)
    return bool(POSE_TAGS & t) and not (BLOCK_TAGS & t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--tags", action="store_true", help="打印全部标签")
    ap.add_argument("--threshold", type=float, default=0.35)
    a = ap.parse_args()
    t = Tagger()
    worst = "OK"
    for p in a.images:
        if not os.path.exists(p):
            print("MISSING %s" % p)
            continue
        rating, tags = t.run(p, a.threshold)
        v = verdict(rating, tags)
        hit_b = sorted(BLOCK_TAGS & set(tags))
        hit_w = sorted(WATCH_TAGS & set(tags))
        print("%-8s rating=%-12s %s" % (v, rating, os.path.basename(p)))
        if overkill(rating, tags):
            print("         escalated: rating=explicit but no key-point tags -> REVIEW, human check")
        if v != "BLOCK" and pose_level(tags):
            print("         pose-level: %s (clothing intact -> REVIEW, not BLOCK)"
                  % ", ".join(sorted(POSE_TAGS & set(tags))))
        if hit_b:
            print("         BLOCK-tags: %s" % ", ".join(hit_b))
        if hit_w:
            print("         watch-tags: %s" % ", ".join(hit_w))
        if a.tags:
            print("         all: %s" % ", ".join(tags))
        order = {"OK": 0, "REVIEW": 1, "BLOCK": 2}
        if order[v] > order[worst]:
            worst = v
    print("OVERALL=%s" % worst)
    return 0 if worst == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
