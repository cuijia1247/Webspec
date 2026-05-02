# -*- coding: utf-8 -*-
"""将 xyxy 像素框转为 YOLO 标注行（相对宽高归一化），供检测脚本写 txt。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def detections_to_yolo_txt(
    class_ids: np.ndarray,
    boxes_xyxy: np.ndarray,
    img_w: int,
    img_h: int,
) -> str:
    """
    YOLO 标签：每行 ``class_id x_center y_center width height``，均为相对整图宽高的 0~1 浮点数。
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"无效图像尺寸: {img_w}x{img_h}")
    if class_ids.size == 0:
        return ""
    lines: list[str] = []
    wf, hf = float(img_w), float(img_h)
    for i in range(class_ids.shape[0]):
        cid = int(class_ids[i])
        x1, y1, x2, y2 = (float(v) for v in boxes_xyxy[i].tolist())
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        lines.append(
            f"{cid} {xc / wf:.6f} {yc / hf:.6f} {bw / wf:.6f} {bh / hf:.6f}"
        )
    return "\n".join(lines) + "\n"


def write_yolo_labels_txt(
    txt_path: Path,
    class_ids: np.ndarray,
    boxes_xyxy: np.ndarray,
    img_w: int,
    img_h: int,
) -> None:
    """
    写入与图像 ``stem`` 对应的 ``.txt``（YOLO detection 格式，无置信度列）。
    """
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(
        detections_to_yolo_txt(class_ids, boxes_xyxy, img_w, img_h),
        encoding="utf-8",
    )
