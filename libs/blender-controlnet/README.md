# blender-controlnet

用 **Blender** 摆场景 / 摆姿势，导出 **depth / OpenPose / 手部** 控制图，喂给 ComfyUI 的 ControlNet
（或 Anima 的 LLLite）。适合"位置、朝向、手型必须精确"的场合 —— 这些靠提示词是碰运气。

- `blender_build_scene.py` / `blender_stage_multi.py` —— 建场景、摆多角色。
- `blender_draw_openpose.py` / `blender_dwpose_to_kps.py` / `blender_extract_kps.py` —— 关键点与骨架图。
- `blender_fit_pose.py` —— 用关键点拟合姿势（把参考图姿势搬到 3D）。
- `blender_make_inpaint_mask.py` —— 从 3D 生成重绘掩版。
- `blender_compare_hands.py` / `blender_crop_hands.py` / `blender_pose_error.py` / `blender_overlay_check.py` —— **量化校验**（手型/姿势误差），别靠肉眼。
- 参考文档：`references/blender_control_pipeline.md`（21 KB，含参数与坑）；样例场景：`references/blender_scenes_example.json`。
