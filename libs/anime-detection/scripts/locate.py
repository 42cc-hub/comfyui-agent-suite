"""产线自检入口（M1~M3 串起来，输出统一契约 + flags + 可选蒙版）。

这是整套东西**最终被产线调用的那个口**：给一批图，回答"人在哪、手/脚在哪、指/趾数量可不可疑"，
并可选地吐出手/脚蒙版（直接喂 `SetLatentNoiseMask` 做局部重绘）。

分层与降级（每层都能单独替换）：
  M1 骨架：优先我们自己的权重（默认 `runs/m1_pose_coco17`，COCO-17 schema），缺省回落到 imgutils 的 dwpose
  M2a 框 ：优先 `runs/m2a_det_v2_960`（person/hand/foot/head），缺省回落 imgutils 的 person/hand/head
           **脚召回兜底**：一个脚框都没有但骨架里有踝点时，用踝点合成候选框（实测交付稿上 det 会漏脚）
  M3 判定：WD14 先出 `state`（covered/shod ⇒ 直接 skip，不劳烦分类器）→ 二元分类器判 ok/bad
  M4 绳子：**不在这里跑**（由 rope 模块的 `rope_predict.py` 负责）；若约定路径下已有产物则引用进来

用法：
  python scripts/locate.py --src <图|目录> --out .runtime/locate [--mask] [--imgsz 1280] [--conf 0.3]

⚠️ 产线权重不是"val 上最好的那个"：v5 在 val 上关键点 0.457 > v4 的 0.349，但在**真实交付图（2000px）**上
   v4@1280 能 7/8 检出人、v5@1280 只有 5/8 ⇒ **交付图上的表现才是选权重的依据**（2026-10-11 实测）。
"""
import argparse
import io
import json
import os
import sys

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, r"<你的用户目录>\.dsh\skills\comfyui-image-generation\scripts")

from export_labels import COCO17_NAMES                            # noqa: E402

# 点名字表内联在这里：locate.py 是产线入口，不该为了几个常量去 import 预标注脚本
# （`prelabel_batch` 顶层就 import imgutils ⇒ 在没装 imgutils 的环境会直接崩，2026-10-11 实测踩到）
BODY18 = ['nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist',
          'r_hip', 'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 'r_eye', 'r_ear', 'l_eye', 'l_ear']
HAND21 = ['wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
          'index_mcp', 'index_pip', 'index_dip', 'index_tip',
          'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
          'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
          'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip']
FOOT3 = ['ankle', 'big_toe', 'small_toe']


def vis_of(score):
    return 2 if score >= 0.5 else (1 if score >= 0.3 else 0)


DEF_POSE = [os.path.join(PROJ, 'runs', 'm1_pose_coco17', 'weights', 'best.pt'),
            os.path.join(PROJ, 'runs', 'm1_pose17_1129', 'weights', 'best.pt')]
# v4：val 关键点 0.349，交付图漏 test6；新版：val 0.835，交付图漏 test5 ⇒ **两个都保留、按"检出人多的"取优**
# ⚠️ 上面两个是 **回落**。M1 现在的首选是下面这个微调 ViTPose（2026-10-13 起）：
# 2026-10-22：rot 增广版，产线框口径 48 图四桶 PCK 0.506→0.588、p=0.044；回滚=改回 v6
#   （`models/vitpose_ft_v6_labelbox_honly`）。注意它是 **HF 目录**（config/preprocessor/safetensors），
#   不是 YOLO 的 `weights/best.pt`；单次覆盖仍用 `--vitpose-dir <目录>`。
# 2026-10-22 晚：权重已归档到 models，runs 清理不影响（`models/vitpose_rot_abs` 与
#   `runs/m1_pose_rot_abs` 逐文件同字节，model.safetensors sha256 8E98B16C…817C86）。
DEF_VITPOSE = os.path.join(PROJ, 'models', 'vitpose_rot_abs')
# 固定 15 张报告集（279 个关节，**从不参与训练/选点**）配对聚类 bootstrap 实测：
#   ViTPose-ft v6  coverage 88% / median 23.1px / common 23.7 / p90 97.8
#   YOLO-pose      coverage 71% / median 66.0px / common 65.6 / p90 341.6
#   dwpose         coverage 52% / median 26.2px / common 25.4 / p90 224.0
#   配对差 v6−dwpose：median −2.0px [−22.0, +14.1] ⇒ **无差别**；mean −31.8px [−53.3, −11.0] p=0.0004 ⇒ **显著**
#   （median 已到"人工重标同一张图"的噪声尺度：现 GT vs 旧 GT 中位 16.0px ⇒ 再压 median 收益极小，
#    换它是为了**覆盖率 + 尾部**：p90 从 224/342 收到 98）
# 口径：top-down，吃 M2a 的 `person` 框（与 prelabel_vitpose.py / 训练评测同口径）；坐标/名字与 YOLO 分支一致
# （COCO-17 全名 + `vis∈{0,1,2}`），所以 check_parts.py 的骨架掩版、脚召回兜底、M3 脚裁图都不用改。
DEF_DET = os.path.join(PROJ, 'runs', 'm2a_det_fix_v2', 'weights', 'best.pt')
# 混合的第二条腿（2026-10-21 用户批准"权重可以换"）：person/head 仍用上面这条（0° 上 person 召回 92.9%、
# 换掉会掉到 82.1%），hand/foot 改用旋转增强半监督版 —— 脚在两个独立口径都赚（真实躺卧 33.3%→83.3%、
# upright 独立集 32.1%→48.6%），手不亏（upright 41.8% vs 36.7%）。
# 只在**不传 `--det-weights`** 时启用（传了就只用那一个模型，行为与改动前完全一致）；实测数据见
# `team/reports/m2a-verifier/rot_semi_det.md` §0 第 8 条 / §10 / §12。
DEF_DET_LIMB = os.path.join(PROJ, 'runs', 'm2a_det_rot_semi', 'weights', 'best.pt')
# 996 张版（2.46h；**确认训练进程退出后**才敢验收）：person 0.984 / hand 0.914 / foot 0.694 / head 0.978
# 308 张版是 0.882 / 0.750 / 0.577 / 0.899 ⇒ 数据翻 3 倍的收益
DEF_M3 = [os.path.join(PROJ, 'runs', 'm3_cls_aug_v1', 'weights', 'best.pt'),
          os.path.join(PROJ, 'runs', 'm3_review_v1', 'weights', 'best.pt')]
# 两个 M3 **都跑、任一报警就算可疑**（recall 优先，符合"宁可多报交人看"）：
#  · `m3_cls_aug_v1`：只用 182 条"数量/形状"标签 + 增强扩充 —— 交付图上**报出 3/4 已知缺陷**
#  · `m3_review_v1` ：含用户新判的 200 张（多出 92 张"不是手脚"）—— val 更"大"，但**交付图反而只报 1 张**
#   （又一次 val≠交付图；两个一起用可以兼顾"数量错"和"不是手脚"两类）
DEF_FOOT = os.path.join(PROJ, 'runs', 'm2a_foot_v1', 'weights', 'best.pt')   # 专用脚（单类），与 4 类取并集
DEF_FACE = os.path.join(PROJ, 'runs', 'm2b_face_v3_3983', 'weights', 'best.pt')  # M2b：256 固定画布，val mAP50 0.993
DEF_GAG = [os.path.join(PROJ, 'runs', 'm2b_gag_v2', 'weights', 'best.pt'),   # 289 head：gag 交付图 4/6
           os.path.join(PROJ, 'runs', 'm2b_gag_v4', 'weights', 'best.pt')]   # 580 head：blindfold 变可信
# 两个都跑、任一报出就算（老办法：不同权重的错误分布不同 ⇒ 取并集补召回）
#   v2（289 head / 195 框）：交付图 gag 02/03/04/07；blindfold 弱（53 框）
#   v4（580 head / 415 框，含 pixiv 眼罩 176 框）：blindfold val mAP50 0.967、训练集外报出率翻倍；
#                                                    代价是 gag 略退（交付图掉 04）
SKIP_STATES = {'covered', 'shod'}          # 穿袜/穿鞋 ⇒ 数不了趾，直接跳过（标注规范 §2.1）


def imgs_from(src):
    if os.path.isdir(src):
        return [os.path.join(src, f) for f in sorted(os.listdir(src))
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    return [src]


def points_from_imgutils(kp, chain):
    out = []
    for part, names in (('body', BODY18), ('left_hand', HAND21), ('right_hand', HAND21),
                        ('left_foot', FOOT3), ('right_foot', FOOT3)):
        pts = getattr(kp, part, None)
        if pts is None:
            continue
        a = np.asarray(pts, dtype=float)
        if a.size == 0:
            continue
        if a.ndim == 1:
            a = a.reshape(1, -1)
        for i, row in enumerate(a):
            sc = float(row[-1])
            out.append({'layer': 'M1', 'chain': chain,
                        'name': '%s.%s' % (part, names[i] if i < len(names) else i),
                        'cls': part, 'x': round(float(row[0]), 1), 'y': round(float(row[1]), 1),
                        'prob': round(sc, 3), 'vis': vis_of(sc)})
    return out


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _dedupe_person_r5(items, img_area, giant=0.50, dup_iou=0.50, ratio=0.85):
    """R5 去重：两个框面积都 >50% 图、IoU>=0.50 且面积比 <0.85 ⇒ 删低 conf 那个。

    与 `scripts/m2a_multi_queue.py` 的 R5 同一条规则（只针对错位巨框对）。
    刻意不放宽到"尺寸相近也删"：贴身穿插场景真人对面积比落在 .86–1.00，与伪框对
    （.53–.99）区间重叠，几何上分不开（见 team/reports/m2a-verifier/dedup_rules/rules.txt）。
    """
    out = []
    for b in sorted(items, key=lambda x: -x['prob']):
        ba = (b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1])
        drop = False
        for k in out:
            ka = (k['box'][2] - k['box'][0]) * (k['box'][3] - k['box'][1])
            if ba / img_area <= giant or ka / img_area <= giant:
                continue
            if _iou(b['box'], k['box']) < dup_iou:
                continue
            if min(ba, ka) / max(ba, ka, 1e-6) < ratio:
                drop = True
                break
        if not drop:
            out.append(b)
    return out


def mask_from_boxes(shape, items, grow=12):
    m = Image.new('L', (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(m)
    for box in items:
        x0, y0, x1, y1 = [int(v) for v in box]
        d.rectangle([max(0, x0 - grow), max(0, y0 - grow), min(shape[1], x1 + grow), min(shape[0], y1 + grow)],
                    fill=255)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pose-weights', nargs='+', default=DEF_POSE,
                    help='可给多个 pose 权重：**取检出人多的那个**。实测 v4 漏 test6、新版漏 test5，取优后 8/8。'
                         '⚠️ 只在 ViTPose（默认）不可用或被 --no-vitpose 关掉时才走这里')
    ap.add_argument('--vitpose-dir', default=DEF_VITPOSE,
                    help='微调后的 ViTPose（HF 目录：config.json + preprocessor_config.json + 权重）。'
                         '**默认存在就优先用它当 M1**（见文件头 DEF_VITPOSE 处的实测数字）；'
                         '目录不存在或传 --no-vitpose 时回落到 --pose-weights → dwpose')
    ap.add_argument('--no-vitpose', dest='vitpose_dir', action='store_const', const=None,
                    help='关掉 ViTPose，M1 回到 YOLO-pose 双权重取优')
    ap.add_argument('--det-weights', default=None,
                    help='**只给一个 4 类检测模型**：传了就只用它，不混合（与改动前行为一致）。'
                         '不传（默认）= 双头混合：person/head 用 `runs/m2a_det_fix_v2`，'
                         'hand/foot 用 `runs/m2a_det_rot_semi`（见 --det-limb-weights）')
    ap.add_argument('--det-limb-weights', default=DEF_DET_LIMB,
                    help='混合模式下提供 hand/foot 的 4 类模型（默认 m2a_det_rot_semi，旋转增强+半监督）。'
                         '只有**不传 --det-weights** 时才用得上')
    ap.add_argument('--det-person-weights', default=None,
                    help='**opt-in** 人框专用腿：给定后用该模型的 person 框**替换**主检测的 person 框'
                         '（替换、不合并），替换后按 R5 去重；hand/foot/head 不受影响。默认 None = 完全不走这段。'
                         '实测候选 `runs/m2a_det_person_hybrid`（batch-2 R .922 / P .738 / 面积比 1.026）')
    ap.add_argument('--foot-weights', default=DEF_FOOT,
                    help='专用脚检测器（单类）。与 4 类模型**取并集**：实测它能把 4 类漏掉的 test1 脚找回来')
    ap.add_argument('--hand-imgsz', type=int, default=1536,
                    help='手第二遍的推理尺寸（**默认 1536 = 开**）。做法：主检测之后用同一个 4 类 det 权重在'
                         '这个尺寸再跑一遍，**只取 hand 类并替换主检测那遍的 hand 框**（其他类不动）。'
                         '与专用手模型（--hand-weights）互相独立。`--hand-imgsz 1280`（= --imgsz）时第二遍不跑，'
                         '逐字节回到旧行为，是逃生门。实测手 1280→1536：`data/hand_gt` R .405→.481 / P .582→.717')
    ap.add_argument('--hand-weights', default=None,
                    help='专用**手**检测器（单类，可选）。与 4 类模型取并集；默认不传 = 关闭'
                         '（`m2a_hand_v1` 未训完，不可用；`m2a_hand_v2` 训完再按交付图实测决定）')
    ap.add_argument('--face-weights', default=DEF_FACE,
                    help='M2b 面部组（输入是 head 裁剪的 256×256）')
    ap.add_argument('--gag-weights', nargs='+', default=DEF_GAG,
                    help='M2b-gag 口塞/眼罩检测器（可多个，取并集；空则关闭）')
    ap.add_argument('--m3-foot-tight', dest='m3_foot_tight', action='store_true', default=True,
                    help='脚再按"下 40%% + 宽 1.2×高"收紧判一次，作为并列旗标 foot_tight_check —— **默认开**：'
                         '交付图验收从 3/4 → 4/4（08 的"趾多"只被这条报出），代价是 02/03 各多一个待复核旗标')
    ap.add_argument('--no-m3-foot-tight', dest='m3_foot_tight', action='store_false',
                    help='关掉上面那条收紧旗标（回到只有主判定的 3/4）')
    ap.add_argument('--m3-foot-crop', choices=['det', 'pose'], default='det',
                    help='M3 脚裁图口径：det=检测器框+10%%（老行为）；pose=踝/膝锚点+35%%（复现训练口径）。'
                         '实测 pose **不能**把 08 的"趾多"救回主判定（仍只被收紧旗标报出），'
                         '但 02/03 的误报消失：已知缺陷 3/4→3/4、主判定∪收紧 4/4→4/4、'
                         '误报图 4/4→2/4（R42）。默认仍是 det —— 改产线默认属于契约变更，待用户拍板')
    ap.add_argument('--m3-conf', type=float, default=0.5,
                    help='M3 判 bad 的 p(bad) 阈值（默认 0.5 = 原来的 argmax 行为）。'
                         '阈值扫描：并集在 0.2~0.7 区间 val 召回恒为 1.00，而池外报出率从 0.42 降到 0.35'
                         '（`data/sweep_m3_thresh.csv`）⇒ 抬高只砍滥报、不动召回')
    ap.add_argument('--gag-conf', type=float, default=0.5,
                    help='M2b-gag 的置信门限。默认 0.5：交付图实测的两个误报（把束带当眼罩 0.41、'
                         '把颈圈当口塞 0.30）都在 0.5 以下，而三个正确的球塞框是 0.65~0.86')
    ap.add_argument('--m3-weights', nargs='+', default=DEF_M3,
                    help='一个或多个二元/多元分类器；**任一模型的判定不是 ok 就算可疑**（recall 优先）')
    ap.add_argument('--mask', action='store_true', help='额外导出手/脚蒙版（重绘用）')
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--imgsz', type=int, default=1280,
                    help='推理分辨率；交付图 2000px 时 1280 明显好于 640（实测 7/8 vs 5/8 检出人）')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mask:
        os.makedirs(os.path.join(a.out, 'mask'), exist_ok=True)

    # ---- 延迟加载：缺哪个模型就降级到 imgutils，绝不让整条链因为一个权重缺失而挂掉 ----
    try:
        from ultralytics import YOLO
    except ImportError:
        YOLO = None
        print('本环境没有 ultralytics ⇒ 全部走 imgutils 回落路径')
    pose_models = []
    if YOLO:
        for w in ([a.pose_weights] if isinstance(a.pose_weights, str) else a.pose_weights):
            if w and os.path.exists(w):
                pose_models.append(YOLO(w))
    pose_model = pose_models[0] if pose_models else None
    # ViTPose 微调权重：M1 的首选。加载失败/目录缺失一律**静默回落**，不让整条链挂掉
    vitpose = None
    if a.vitpose_dir and os.path.isdir(a.vitpose_dir):
        try:
            import torch
            from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
            _vp_proc = VitPoseImageProcessor.from_pretrained(a.vitpose_dir)
            _vp_model = VitPoseForPoseEstimation.from_pretrained(a.vitpose_dir).eval()
            _vp_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            _vp_model.to(_vp_dev)
            vitpose = (_vp_proc, _vp_model, _vp_dev)
        except Exception as e:
            print('ViTPose 不可用（%s）⇒ M1 回落 YOLO-pose/dwpose' % str(e)[:120])
    elif a.vitpose_dir:
        print('ViTPose 目录不存在（%s）⇒ M1 回落 YOLO-pose/dwpose' % a.vitpose_dir)
    # 双头混合：person/head ← 主模型（DEF_DET），hand/foot ← 第二条腿（DEF_DET_LIMB）。
    # 显式传 --det-weights 时 hybrid 关闭（det_limb_model=None），行为回到单模型。
    _det_w = a.det_weights or DEF_DET
    det_model = YOLO(_det_w) if (YOLO and os.path.exists(_det_w)) else None
    det_limb_model = None
    if det_model is not None and a.det_weights is None:
        _limb_w = a.det_limb_weights
        if _limb_w and os.path.exists(_limb_w) and os.path.abspath(_limb_w) != os.path.abspath(_det_w):
            det_limb_model = YOLO(_limb_w)
        elif _limb_w and os.path.abspath(_limb_w) == os.path.abspath(_det_w):
            print('注意：--det-limb-weights 与主检测权重相同，混合等价于单模型')
    # opt-in 人框专用腿：默认 None ⇒ 不加载、下面替换逻辑跳过，默认路径逐字段不变。
    det_person_model = None
    if a.det_person_weights and YOLO and os.path.exists(a.det_person_weights):
        det_person_model = YOLO(a.det_person_weights)
    m3_models = []
    if YOLO:
        for w in ([a.m3_weights] if isinstance(a.m3_weights, str) else a.m3_weights):
            if w and os.path.exists(w):
                m3_models.append(YOLO(w))
    m3_model = m3_models[0] if m3_models else None
    foot_model = YOLO(a.foot_weights) if (YOLO and a.foot_weights and os.path.exists(a.foot_weights)) else None
    hand_model = YOLO(a.hand_weights) if (YOLO and a.hand_weights and os.path.exists(a.hand_weights)) else None
    face_model = YOLO(a.face_weights) if (YOLO and a.face_weights and os.path.exists(a.face_weights)) else None
    gag_models = []
    for _w in (a.gag_weights if isinstance(a.gag_weights, (list, tuple)) else [a.gag_weights]):
        if YOLO and _w and os.path.exists(_w):
            gag_models.append(YOLO(_w))
    gag_model = gag_models[0] if gag_models else None
    # 坑：`train.py --task det` 会把 checkpoint 的 task 存成短名 'det'，而 predict 只认 'detect'
    for _m in (pose_model, det_model, m3_model, foot_model, face_model, hand_model, det_limb_model,
               det_person_model):
        if _m is not None and getattr(_m, 'task', None) == 'det':
            _m.task = 'detect'

    dwpose_estimate = None
    if vitpose is None and pose_model is None:
        try:
            from imgutils.pose import dwpose_estimate
        except ImportError:
            pass
    detect_hands = detect_heads = detect_person = None
    if det_model is None:
        try:
            from imgutils.detect import detect_hands, detect_heads, detect_person
        except ImportError:
            pass
    tagger = None
    state_of = None
    try:
        from _libs import add_lib; add_lib("wd14-gate")
        from wd14_gate import Tagger
        from wd14_autotag import state_of
        tagger = Tagger()
    except Exception as e:
        print('WD14 不可用（state 一律 unknown）：%s' % e)

    n = 0
    for src in a.src:
        for path in imgs_from(src):
            name = os.path.splitext(os.path.basename(path))[0]
            img = Image.open(path).convert('RGB')
            w, h = img.size
            rec = {'image': os.path.basename(path), 'size': '%dx%d' % (w, h), 'chain': [],
                   'points': [], 'boxes': [], 'flags': []}

            # ---- M2a det 提前：ViTPose 是 top-down，要吃 person 框，而 M2a 的框原本在 M1 之后才算 ----
            # 只跑一次 det，M1（ViTPose）与下面 M2a 共用同一份结果 ⇒ 不增加耗时
            det_boxes, det_r = [], None
            if det_model is not None:
                try:
                    det_r = det_model.predict(path, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
                    det_boxes = [{'layer': 'M2a', 'cls': det_r.names[int(b.cls[0])],
                                  'box': [round(v, 1) for v in b.xyxy[0].tolist()],
                                  'prob': round(float(b.conf[0]), 3)} for b in (det_r.boxes or [])]
                except Exception as e:
                    rec['flags'].append({'type': 'det_error', 'detail': str(e)[:80]})

            # ---- 双头混合（默认路径）：hand/foot 换成第二条腿的输出，person/head 保持主模型 ----
            # 刻意选择：**即使主模型一个 person 框都没报**，仍保留第二条腿的 hand/foot。
            # 理由（lead 2026-10-21 口径）：宁可多框，也不要因为"没框住人"就把手脚整批丢掉；
            # 手脚框不依赖 person 框存在（下游按各自 cls 消费），实测 g4_01_lumine_sickbed 在
            # person=0 时保留了 6 个 hand/foot 框。
            # 为什么不是整体替换：person 在 0° 上 rot_semi 会从 92.9% 掉到 82.1%（§0 第 3 条）；
            # 为什么不是并集：两模型的 hand/foot 并集在实测里只多真框 0 个、多 13/15 个误报（§7 的教训）。
            if det_model is not None and det_limb_model is not None:
                try:
                    rl = det_limb_model.predict(path, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
                    _limb = []
                    for b in (rl.boxes or []):
                        _cls = rl.names[int(b.cls[0])]
                        if 'hand' not in _cls and 'foot' not in _cls:
                            continue
                        _limb.append({'layer': 'M2a', 'cls': _cls,
                                      'box': [round(v, 1) for v in b.xyxy[0].tolist()],
                                      'prob': round(float(b.conf[0]), 3),
                                      'from': 'det_limb'})
                    det_boxes = [b for b in det_boxes if 'hand' not in b['cls'] and 'foot' not in b['cls']]
                    det_boxes += _limb
                    rec['chain'].append('M2a.hybrid_limb')
                except Exception as e:
                    rec['flags'].append({'type': 'det_limb_error', 'detail': str(e)[:80]})

            # ---- opt-in person 专用腿（--det-person-weights；默认 None ⇒ 整段跳过）----
            # 语义：person 框**只**来自该模型（替换，不与主检测合并），hand/foot/head 保持上面的现役组合。
            # 为什么不合并：两模型 person 并集实测只多真框也成倍多误框，还会破坏 1:1 链（一框罩两人）；
            # 为什么在 _assign 之前替换：person_id 与 M1 链都按 person 框序分配，必须让人框先定稿。
            # 替换后按 R5 去重，压掉"两个 55% 巨框罩同一副身体"这对错位重复框。
            if a.det_person_weights and det_person_model is not None:
                try:
                    rp = det_person_model.predict(path, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
                    _ppl = [{'layer': 'M2a', 'cls': 'person',
                             'box': [round(v, 1) for v in b.xyxy[0].tolist()],
                             'prob': round(float(b.conf[0]), 3),
                             'from': 'det_person'} for b in (rp.boxes or [])
                            if rp.names[int(b.cls[0])] == 'person']
                    _ih, _iw = float(rp.orig_shape[0]), float(rp.orig_shape[1])
                    det_boxes = ([b for b in det_boxes if b['cls'] != 'person']
                                 + _dedupe_person_r5(_ppl, _iw * _ih))
                    rec['chain'].append('M2a.hybrid_person')
                except Exception as e:
                    rec['flags'].append({'type': 'det_person_error', 'detail': str(e)[:80]})

            # ---- M1 骨架 ----
            vit_pts = []
            vpid = {}   # 'M1#i' -> i：ViTPose 分支的链就是第 i 个人框裁出来的（:278/:284/:293/:347 保序）
            if vitpose is not None:
                pboxes = [b['box'] for b in det_boxes if b['cls'] == 'person']
                if pboxes:
                    try:
                        _vproc, _vmodel, _vdev = vitpose
                        import torch
                        with torch.no_grad():
                            _vin = _vproc(images=img, boxes=[pboxes], return_tensors='pt')
                            _vin = {k: (v.to(_vdev) if hasattr(v, 'to') else v) for k, v in _vin.items()}
                            _vout = _vmodel(**_vin)
                        _vres = _vproc.post_process_pose_estimation(_vout, boxes=[pboxes])
                        for pi, inst in enumerate(_vres[0]):
                            vpid['M1#%d' % pi] = pi
                            xy = inst['keypoints'].cpu().numpy()
                            sc = inst['scores'].cpu().numpy()
                            for j, cname in enumerate(COCO17_NAMES):
                                s = float(sc[j])
                                vit_pts.append({'layer': 'M1', 'chain': 'M1#%d' % pi,
                                                'name': 'body.%s' % cname, 'cls': 'body',
                                                'x': round(float(xy[j][0]), 1), 'y': round(float(xy[j][1]), 1),
                                                'prob': round(s, 3), 'vis': vis_of(s),
                                                'src': 'ViTPose-ft'})
                        rec['chain'].append('M1.vitpose')
                        rec['vitpose'] = os.path.basename(os.path.normpath(a.vitpose_dir))
                        rec['n_people'] = len(_vres[0])
                    except Exception as e:
                        rec['flags'].append({'type': 'vitpose_error', 'detail': str(e)[:80]})
                        vit_pts = []

            if vit_pts:
                rec['points'] += vit_pts
            elif pose_models:
                # 多权重取优：跑全部，用"检出人最多"的那个（实测能把 7/8 补成 8/8）
                best = None
                for _pm in pose_models:
                    _r = _pm.predict(path, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
                    if best is None or len(_r.keypoints or []) > len(best.keypoints or []):
                        best, best_model = _r, _pm
                r = best
                nk = 17
                try:
                    nk = int(best_model.model.yaml.get('kpt_shape', [17, 3])[0])
                except Exception:
                    pass
                names_tbl = COCO17_NAMES if nk == 17 else BODY18
                for pi, k in enumerate(r.keypoints or []):
                    xy = k.xy[0].tolist()
                    cf = k.conf[0].tolist()
                    for i in range(min(len(names_tbl), len(xy))):
                        rec['points'].append({'layer': 'M1', 'chain': 'M1#%d' % pi,
                                              'name': 'body.%s' % names_tbl[i], 'cls': 'body',
                                              'x': round(xy[i][0], 1), 'y': round(xy[i][1], 1),
                                              'prob': round(float(cf[i]), 3), 'vis': vis_of(float(cf[i]))})
                rec['chain'].append('M1.ours')
                rec['n_people'] = len(r.keypoints or [])
            elif dwpose_estimate is None:
                rec['flags'].append({'type': 'pose_unavailable', 'detail': '既无我们的 pose 权重也无 imgutils'})
                rec['n_people'] = 0
            else:
                try:
                    kps = dwpose_estimate(img)
                    rec['chain'].append('M1.imgutils')
                    for pi, kp in enumerate(kps):
                        rec['points'] += points_from_imgutils(kp, 'M1#%d' % pi)
                    rec['n_people'] = len(kps)
                except Exception as e:
                    rec['flags'].append({'type': 'pose_error', 'detail': str(e)[:80]})
                    rec['n_people'] = 0

            # ---- M2a 身体组 ----
            if det_model is not None:
                rec['boxes'] += det_boxes
                rec['chain'].append('M2a.ours')
            elif detect_person is None:
                rec['flags'].append({'type': 'det_unavailable', 'detail': '既无我们的 det 权重也无 imgutils'})
            else:
                for fn, cls in ((detect_person, 'person'), (detect_hands, 'hand'), (detect_heads, 'head')):
                    try:
                        for box, _l, c in fn(img):
                            rec['boxes'].append({'layer': 'M2a', 'cls': cls,
                                                 'box': [int(v) for v in box], 'prob': round(float(c), 3)})
                    except Exception:
                        pass
                rec['chain'].append('M2a.imgutils')

            # ---- M2b 面部组：head 框裁 256×256 再检脸/眼（尺度固定的"标准画布"）----
            #      同一张 256 裁剪图上顺带跑 gag/blindfold 专用小检测器（M2b-gag，R36）。
            #      为什么单独一个模型而不是并进 M2b 四类：那两类只有"目标覆盖完整"的 289 张 head 有人工标签，
            #      并进 1462 张的 ds_face 会拿 442 张"很可能真有目标"的图当背景（R35）。
            if (face_model is not None or gag_model is not None) and (rec['points'] or rec['boxes']):
                heads = [b for b in rec['boxes'] if b['cls'] == 'head']
                for hi, hb in enumerate(heads):
                    x0, y0, x1, y1 = hb['box']
                    pad = int(0.15 * max(x1 - x0, y1 - y0))
                    hx0, hy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
                    hx1, hy1 = min(w, int(x1 + pad)), min(h, int(y1 + pad))
                    if hx1 - hx0 < 32 or hy1 - hy0 < 32:
                        continue
                    crop = img.crop((hx0, hy0, hx1, hy1)).resize((256, 256), Image.LANCZOS)
                    cp = os.path.join(a.out, '_tmp_head.png')
                    crop.save(cp)
                    sx, sy = (hx1 - hx0) / 256.0, (hy1 - hy0) / 256.0
                    for model, layer, tag, cf in tuple(
                            [(m, 'M2b', 'M2b.ours', 0.25) for m in ([face_model] if face_model else [])] +
                            [(m, 'M2b-gag', 'M2b.gag', a.gag_conf) for m in gag_models]):
                        if model is None:
                            continue
                        rf = model.predict(cp, conf=cf, verbose=False)[0]
                        for b in (rf.boxes or []):
                            cls = rf.names[int(b.cls[0])]
                            bx = b.xyxy[0].tolist()
                            nb = [round(hx0 + bx[0] * sx, 1), round(hy0 + bx[1] * sy, 1),
                                  round(hx0 + bx[2] * sx, 1), round(hy0 + bx[3] * sy, 1)]
                            if layer == 'M2b-gag' and any(
                                    _iou(nb, x['box']) > 0.6 for x in rec['boxes']
                                    if x.get('layer') == 'M2b-gag' and x.get('cls') == cls):
                                continue                       # 两个 gag 权重报了同一个目标 ⇒ 只留一个
                            rec['boxes'].append({'layer': layer, 'cls': cls, 'box': nb,
                                                 'prob': round(float(b.conf[0]), 3),
                                                 'from': 'head%d_256' % hi})
                        rec['chain'].append(tag)

            # ---- 脚召回兜底（两级）----
            hands = [b for b in rec['boxes'] if b['cls'] == 'hand']
            feet = [b for b in rec['boxes'] if b['cls'] in ('foot', 'left_foot', 'right_foot')]
            # ① 专用脚检测器（单类）与 4 类**取并集**：实测能补回 4 类漏掉的脚（test1）
            if foot_model is not None:
                try:
                    rf = foot_model.predict(path, conf=max(0.15, a.conf - 0.15), imgsz=a.imgsz, verbose=False)[0]
                    for b in (rf.boxes or []):
                        nb = [round(v, 1) for v in b.xyxy[0].tolist()]
                        if any(_iou(nb, f['box']) > 0.5 for f in feet):
                            continue
                        feet.append({'layer': 'M2a', 'cls': 'foot', 'box': nb,
                                     'prob': round(float(b.conf[0]), 3), 'from': 'foot_model'})
                        rec['chain'].append('M2a.foot_model')
                except Exception as e:
                    rec['flags'].append({'type': 'foot_model_error', 'detail': str(e)[:80]})
            # ①.5 手第二遍（`--hand-imgsz`，默认 1536）：同一个 4 类 det 权重换分辨率再跑一次，只取 hand 类，
            #      **替换**主检测那遍的 hand 框（不是并集——实测并集的真框与第二遍完全相同却多 13/15 个误报）。
            #      动机：手是小目标，1280 下手长边 <72px 的基本检不到（GT 上该档四口径召回全 0），1536 能补一部分。
            #      逃生门：`--hand-imgsz` 等于 `--imgsz` 时这段不跑 ⇒ 逐字节回到旧行为。
            #      混合模式下由第二条腿（转增强模型）负责手，否则仍是主模型换分辨率再跑一遍。
            _hand_src = det_limb_model if det_limb_model is not None else det_model
            if _hand_src is not None and a.hand_imgsz and a.hand_imgsz != a.imgsz:
                try:
                    r2 = _hand_src.predict(path, conf=a.conf, imgsz=a.hand_imgsz, verbose=False)[0]
                    p2 = [{'layer': 'M2a', 'cls': 'hand',
                           'box': [round(v, 1) for v in b.xyxy[0].tolist()],
                           'prob': round(float(b.conf[0]), 3),
                           'from': 'hand_pass2_%d' % a.hand_imgsz}
                          for b in (r2.boxes or []) if r2.names[int(b.cls[0])] == 'hand']
                    main_ids = {id(b) for b in hands}
                    rec['boxes'] = [b for b in rec['boxes'] if id(b) not in main_ids]
                    hands[:] = p2
                    rec['chain'].append('M2a.hand_pass2')
                except Exception as e:
                    rec['flags'].append({'type': 'hand_pass2_error', 'detail': str(e)[:80]})
            # ② 专用**手**检测器（单类）与 4 类取并集：与脚同理 —— 手是交付图上最弱的一环
            #    （1280 下只有 4/8 张检到手、1600 下 5/8）。默认**关**（不传 `--hand-weights`），
            #    因为 `runs/m2a_hand_v1` 只训到 27/150 epoch 就被中断、不可用；等 `m2a_hand_v2` 训完按实测决定。
            if hand_model is not None:
                try:
                    rf = hand_model.predict(path, conf=max(0.15, a.conf - 0.15),
                                            imgsz=a.imgsz, verbose=False)[0]
                    for b in (rf.boxes or []):
                        nb = [round(v, 1) for v in b.xyxy[0].tolist()]
                        if any(_iou(nb, f['box']) > 0.5 for f in hands):
                            continue
                        hands.append({'layer': 'M2a', 'cls': 'hand', 'box': nb,
                                      'prob': round(float(b.conf[0]), 3), 'from': 'hand_model'})
                        rec['chain'].append('M2a.hand_model')
                except Exception as e:
                    rec['flags'].append({'type': 'hand_model_error', 'detail': str(e)[:80]})
            if not feet:
                # 脚召回兜底：踝点优先；**踝点不可信时用"髋→膝"方向外推小腿**，把框中心落到脚趾附近
                # （test6 那种脚底朝镜头的图，踝点置信度很低，只靠踝点会完全拿不到框）
                hgt = max([b['box'][3] - b['box'][1] for b in rec['boxes'] if b['cls'] == 'person'],
                          default=0) or (h * 0.5)
                side_len = 0.28 * hgt
                by_name = {}
                for p in rec['points']:
                    if p['cls'] == 'body':
                        by_name.setdefault(p['chain'], {})[p['name'].split('.', 1)[-1]] = p
                for chain, pts in by_name.items():
                    for sd in ('left', 'right'):
                        ank, knee, hip = pts.get('%s_ankle' % sd), pts.get('%s_knee' % sd), pts.get('%s_hip' % sd)
                        cx = cy = None
                        src, lc = None, False
                        # 门槛放宽到"坐标存在即可"：test6 那种脚底朝镜头的图，踝/膝 prob 只有 0.05~0.25，
                        # 用 vis>=1 会一个框都拿不到。低置信的框在 flag 里标 low_conf，交给下游按可疑处理。
                        if ank and (ank['vis'] >= 1 or ank['prob'] >= 0.05):
                            cx, cy, src = ank['x'], ank['y'], 'ankle:' + sd
                            lc = ank['vis'] < 1
                        elif knee and hip and (knee['vis'] >= 1 or knee['prob'] >= 0.05) and hip['vis'] >= 1:
                            vx, vy = knee['x'] - hip['x'], knee['y'] - hip['y']
                            cx, cy, src = knee['x'] + 0.9 * vx, knee['y'] + 0.9 * vy, 'knee_extrapolate:' + sd
                            lc = knee['vis'] < 1
                        if cx is None:
                            continue
                        bx = [cx - side_len / 2, cy - side_len / 2, cx + side_len / 2, cy + side_len / 2]
                        bx = [max(0, bx[0]), max(0, bx[1]), min(w, bx[2]), min(h, bx[3])]
                        if min(bx[2] - bx[0], bx[3] - bx[1]) < 48:
                            continue
                        feet.append({'layer': 'M2a', 'cls': 'foot(pose)', 'box': [round(v, 1) for v in bx],
                                     'prob': (ank or knee)['prob'], 'from': src})
                        rec['flags'].append({'type': 'foot_recovered_from_pose', 'how': src,
                                             'prob': (ank or knee)['prob'], 'low_conf': lc,
                                             'box': [round(v, 1) for v in bx]})

            # ---- M3 判定：WD14 状态 → 分类器 ----
            # 坑（2026-10-11 实测，R39）：M3 的**训练裁图**是 `prelabel_batch` 用"姿态脚点包围盒 + 0.35 外扩"
            # 做的（紧贴脚），而这里一直用"检测器脚框 + 0.10 外扩"。同一张图同一只脚，前者的 p(bad)=1.00、
            # 后者 0.00 —— 交付图 08 的"趾多"三个模型全判 ok 就是这个口径差造成的。
            # `--m3-foot-crop pose` 改成用**踝/膝锚点**复现训练口径（默认 det = 老行为，先不动产线）。
            def foot_crop_pose(bx):
                pts_ = [p for p in rec['points'] if p.get('cls') == 'body' and p.get('vis', 0) >= 1]
                ank = [p for p in pts_ if 'ankle' in (p.get('name') or '')]
                kne = [p for p in pts_ if 'knee' in (p.get('name') or '')]
                if not ank:
                    return None
                ccx, ccy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
                a = min(ank, key=lambda p: (p['x'] - ccx) ** 2 + (p['y'] - ccy) ** 2)
                side = 'left' if 'left' in (a.get('name') or '') else 'right'
                k = min([p for p in kne if side in (p.get('name') or '')] or [None],
                        key=lambda p: ((p['x'] - a['x']) ** 2 + (p['y'] - a['y']) ** 2) if p else 1e9,
                        default=None)
                if k is not None:
                    vx, vy = a['x'] - k['x'], a['y'] - k['y']          # 膝→踝方向 = 脚尖方向
                    shin = max(8.0, (vx * vx + vy * vy) ** 0.5)
                    tx, ty = a['x'] + 0.9 * vx, a['y'] + 0.9 * vy     # 再往外推 0.9 个小腿长
                else:
                    shin, tx, ty = max(24.0, (bx[2] - bx[0]) * 0.5), a['x'], a['y']
                half = 0.75 * shin
                return (min(a['x'], tx) - half, min(a['y'], ty) - half,
                        max(a['x'], tx) + half, max(a['y'], ty) + half)

            def m3_vote(_mm, path):
                """→ (类名, top1 置信, p(bad))。判定用 p(bad) ≥ `--m3-conf`，默认 0.5 时与"看 top1 标签"等价。"""
                res = _mm.predict(path, verbose=False)[0]
                pr = res.probs.data.tolist()
                bi = [i for i, n in _mm.names.items() if n == 'bad']
                if not bi:      # 三类/多类权重（类别口径消融）：除 ok 以外都算可疑
                    bi = [i for i, n in _mm.names.items() if n != 'ok']
                pb = max(pr[j] for j in bi) if bi else 0.0
                return str(_mm.names[int(res.probs.top1)]), round(float(res.probs.top1conf), 3), pb

            for kind, arr in (('hand', hands), ('foot', feet)):
                for i, b in enumerate(arr):
                    x0, y0, x1, y1 = b['box']
                    pad_f = 0.10
                    if kind == 'foot' and a.m3_foot_crop == 'pose':
                        fb = foot_crop_pose([x0, y0, x1, y1])
                        if fb is not None:
                            x0, y0, x1, y1 = fb
                            pad_f = 0.35
                    pad = int(pad_f * max(x1 - x0, y1 - y0))
                    box = (max(0, int(x0 - pad)), max(0, int(y0 - pad)),
                           min(w, int(x1 + pad)), min(h, int(y1 + pad)))
                    if min(box[2] - box[0], box[3] - box[1]) < 48:
                        continue
                    cp = os.path.join(a.out, '_tmp_crop.png')
                    img.crop(box).save(cp)
                    st = 'unknown'
                    if tagger is not None:
                        try:
                            _r, tags = tagger.run(cp, threshold=0.35)
                            st = state_of('hand' if kind == 'hand' else 'left_foot', tags)
                        except Exception:
                            pass
                    entry = {'type': 'part_check', 'part': kind, 'index': i + 1,
                             'state': st, 'box': list(box), 'det_conf': b.get('prob')}
                    # 并列信号（默认关）：R39 发现 M3 的训练裁图是"紧贴远端脚"，产线给的是"整只脚+腿"。
                    # 收紧后再判一次，作为**额外**旗标（不动上面那条判定），代价（新增误报）显式可查。
                    if kind == 'foot' and a.m3_foot_tight and m3_models:
                        W0, H0 = box[2] - box[0], box[3] - box[1]
                        ty = box[1] + 0.60 * H0
                        w2 = 1.2 * (box[3] - ty)
                        cx = (box[0] + box[2]) / 2.0
                        tbox = (max(0, int(cx - w2 / 2)), max(0, int(ty)),
                                min(w, int(cx + w2 / 2)), min(h, box[3]))
                        if min(tbox[2] - tbox[0], tbox[3] - tbox[1]) >= 48:
                            tcp = os.path.join(a.out, '_tmp_crop_tight.png')
                            img.crop(tbox).save(tcp)
                            tv, tpb = {}, {}
                            for _mm in m3_models:
                                _t = os.path.basename(os.path.dirname(os.path.dirname(_mm.ckpt_path or 'x')))
                                _nm, _cf, tpb[_t] = m3_vote(_mm, tcp)
                                tv[_t] = (_nm, _cf)
                            tbad = {k: v for k, v in tv.items() if tpb[k] >= a.m3_conf}
                            rec.setdefault('flags', []).append(
                                {'type': 'foot_tight_check', 'box': list(tbox), 'votes': tv,
                                 'verdict': ('bad' if tbad else 'ok'),
                                 'prob': (round(tpb[list(tbad)[0]], 3) if tbad else
                                          list(tv.values())[0][1])})
                    if st in SKIP_STATES:
                        entry['verdict'] = 'skip'
                        entry['why'] = 'WD14 判 %s（数不了）' % st
                    elif m3_models:
                        votes, vpb = {}, {}
                        for _mm in m3_models:
                            _t = os.path.basename(os.path.dirname(os.path.dirname(_mm.ckpt_path or 'x')))
                            _nm, _cf, vpb[_t] = m3_vote(_mm, cp)
                            votes[_t] = (_nm, _cf)
                        # **任一模型 p(bad) ≥ --m3-conf 就算可疑**（recall 优先）；默认 0.5 即原行为
                        bad_votes = {k: v for k, v in votes.items() if vpb[k] >= a.m3_conf}
                        entry['votes'] = votes
                        if bad_votes:
                            entry['verdict'] = 'bad'
                            _k = max(bad_votes, key=lambda k: vpb[k])
                            entry['prob'] = round(vpb[_k], 3)
                        else:
                            entry['verdict'] = 'ok'
                            entry['prob'] = list(votes.values())[0][1]
                    else:
                        entry['verdict'] = 'unknown'
                    rec['flags'].append(entry)
            if not hands:
                rec['flags'].append({'type': 'no_hand_visible'})
            if rec.get('n_people', 0) == 0:
                rec['flags'].append({'type': 'no_person'})

            # ---- M4 绳子：只消费 rope 模块的产物（约定路径），不在这里跑它的模型 ----
            rope_mask = os.path.join(PROJ, 'data', 'masks', 'rope', '%s_mask.png' % name)
            if os.path.exists(rope_mask):
                rope_conf = os.path.join(PROJ, 'data', 'masks', 'rope', '%s_conf.png' % name)
                rec['ropes'] = {'mask': rope_mask.replace('\\', '/'),
                                'conf': rope_conf.replace('\\', '/') if os.path.exists(rope_conf) else None,
                                'source': 'M4(rope module)'}

            if a.mask and (hands or feet):
                mask_from_boxes((h, w), [b['box'] for b in hands + feet]).save(
                    os.path.join(a.out, 'mask', '%s_hands_feet.png' % name))

            # ---- 归属与"补出来的部位"（2026-10-13，下游做**逐人掩版重绘**提的需求）----
            # ① 每个点/框挂 `person_id`：M1 的每条骨架链按关键点包围盒归属，
            #    其余框按"中心落在哪个 person 框内、否则 IoU 最大"归属（-1 = 没归上）。
            # ② `n_persons`（M2a 的 person 框数）与 `n_skeletons`（M1 链数）分开报 —— 二者本来就会不一致，
            #    旧的 `n_people` 只是骨架数，保留不改（兼容）。
            # ③ `boxes_all` = `boxes` ∪ 兜底补的部位（专用脚/踝点外推/专用手），每个框带 `origin`：
            #    `det`（检测器直接给出）或 `fallback`（兜底路径补的）。`boxes` 本身**不变**（老行为）。
            persons = [b for b in rec['boxes'] if b.get('cls') == 'person']
            for i, p in enumerate(persons):
                p['person_id'] = i

            def _ctr(bx):
                return ((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0)

            def _assign(bx):
                cx, cy = _ctr(bx)
                for i, p in enumerate(persons):
                    x0, y0, x1, y1 = p['box']
                    if x0 <= cx <= x1 and y0 <= cy <= y1:
                        return i
                best, bi = 0.0, -1
                for i, p in enumerate(persons):
                    iou = _iou(list(bx), p['box'])
                    if iou > best:
                        best, bi = iou, i
                return bi

            chain_box = {}
            for p in rec['points']:
                c = p.get('chain')
                if not c:
                    continue
                b = chain_box.setdefault(c, [p['x'], p['y'], p['x'], p['y']])
                b[0], b[1] = min(b[0], p['x']), min(b[1], p['y'])
                b[2], b[3] = max(b[2], p['x']), max(b[3], p['y'])
            for c, b in chain_box.items():
                # ViTPose 分支的链本身就是人框裁出来的（:278/:284/:293 → 链号 = 人框号），直接用；
                # 其余分支（YOLO-pose 回退 :325、imgutils :339）没有这条对应，仍按几何归属。
                pid = vpid.get(c)
                if pid is None or pid >= len(persons):
                    pid = _assign(b)
                for p in rec['points']:
                    if p.get('chain') == c:
                        p['person_id'] = pid

            def _origin(b):
                if b.get('from') in ('foot_model', 'hand_model') or str(b.get('cls', '')).endswith('(pose)'):
                    return 'fallback'
                return 'det'

            for b in rec['boxes']:
                if b.get('cls') != 'person':
                    b.setdefault('person_id', _assign(b['box']))
                b['origin'] = _origin(b)
            extra = [b for b in (hands + feet) if id(b) not in {id(x) for x in rec['boxes']}]
            for b in extra:
                b.setdefault('person_id', _assign(b['box']))
                b['origin'] = _origin(b)
            rec['boxes_all'] = list(rec['boxes']) + extra
            rec['n_persons'] = len(persons)
            rec['n_skeletons'] = len(chain_box)

            with io.open(os.path.join(a.out, '%s.json' % name), 'w', encoding='utf-8') as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=1)
            nbad = sum(1 for f in rec['flags']
                       if f.get('type') == 'part_check' and f.get('verdict') == 'bad')
            print('[%-36s] 人=%s 手框=%d 脚框=%d 判 bad=%d  %s'
                  % (os.path.basename(path), rec.get('n_people'), len(hands), len(feet), nbad,
                     ','.join(sorted({f['type'] for f in rec['flags']}))), flush=True)
            n += 1
    tmp = os.path.join(a.out, '_tmp_crop.png')
    if os.path.exists(tmp):
        os.remove(tmp)
    print('\n完成 %d 张 → %s（JSON 契约 + flags%s）' % (n, a.out, ' + mask' if a.mask else ''))


if __name__ == '__main__':
    main()
