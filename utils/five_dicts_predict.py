# -*- coding: utf-8 -*-
"""
对网页截图做五分区（header / footer / body / leftsider / rightsider）YOLO 检测，
并在原图上绘制框与标签。

默认配置与 `start_single_detect.py` 一致地使用 Ultralytics YOLO 推理接口，并支持
可选二次 NMS、按类别去重、边界裁剪等后处理。

默认将标注图保存到仓库 ``output/level1/``（可用 ``-o`` 覆盖文件名或给定绝对路径）。

用法（在仓库根目录 Webspec 下执行，需先启用 conda 环境 webspec）::

    conda activate webspec
    pip install ultralytics opencv-python
    python utils/five_dicts_predict.py --image path/to/screenshot.png
    python utils/five_dicts_predict.py --auto --input-dir data/images_origin
    python utils/five_dicts_predict.py --auto --input-dir data/images_origin --recursive
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors as ultra_colors

from yolo_label_export import write_yolo_labels_txt

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

# 仓库根目录：.../Webspec（本文件位于 utils/five_dicts_predict.py）
REPO_ROOT = Path(__file__).resolve().parent.parent
# 默认可视化结果目录（相对仓库根）
DEFAULT_VIS_OUTPUT_DIR = REPO_ROOT / "output" / "level1"
# 参与批量遍历的图片扩展名（小写，含点）
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


@dataclass
class FiveDictsPredictConfig:
    """与业务侧 YAML 对齐的推理与后处理参数（可在代码或 CLI 中覆盖）。"""

    model_path: str = "pretrainedModels/level1/best.ptt"
    confidence_threshold: float = 0.25
    # YOLO predict 内置 NMS 的 IoU（与 Ultralytics 默认 0.7 一致）
    predict_nms_iou_threshold: float = 0.7
    enable_nms: bool = True
    enable_duplicate_filtering: bool = True
    # 二次 NMS（按类别分组后分别做 NMS，避免同类框大量重叠）
    nms_iou_threshold: float = 0.001
    # 将框限制在图像范围内时的「迭代裁剪」次数上限（多次收敛于边界）
    max_clip_attempts: int = 4
    prefer_clipping: bool = True
    layout_labels: list[str] = field(
        default_factory=lambda: [
            "body",
            "footer",
            "header",
            "leftsider",
            "rightsider",
        ]
    )


def _resolve_model_path(repo_root: Path, model_path: str) -> Path:
    """
    解析权重路径：支持相对仓库根目录；若用户写成 .ptt 且文件不存在，则尝试 .pt。
    """
    p = Path(model_path)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    if p.is_file():
        return p
    if p.suffix.lower() == ".ptt":
        alt = p.with_suffix(".pt")
        if alt.is_file():
            return alt
    return p


def nms_xyxy_per_class(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """
    对每个类别分别做标准 NMS，返回保留的下标（一维 int 数组）。

    NMS 规则：得分降序遍历；与已保留框 IoU > threshold 的框删除。
    """
    if boxes_xyxy.size == 0:
        return np.zeros((0,), dtype=np.int64)

    keep_all: list[int] = []
    for cid in np.unique(class_ids):
        idxs = np.where(class_ids == cid)[0]
        if idxs.size == 0:
            continue
        sub_boxes = boxes_xyxy[idxs]
        sub_scores = scores[idxs]
        order = sub_scores.argsort()[::-1]
        picked_local: list[int] = []
        while order.size > 0:
            i = int(order[0])
            picked_local.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            # IoU(i, rest)
            bx = sub_boxes[i : i + 1]
            xx1 = np.maximum(bx[0, 0], sub_boxes[rest, 0])
            yy1 = np.maximum(bx[0, 1], sub_boxes[rest, 1])
            xx2 = np.minimum(bx[0, 2], sub_boxes[rest, 2])
            yy2 = np.minimum(bx[0, 3], sub_boxes[rest, 3])
            iw = np.clip(xx2 - xx1, a_min=0.0, a_max=None)
            ih = np.clip(yy2 - yy1, a_min=0.0, a_max=None)
            inter = iw * ih
            area_i = (bx[0, 2] - bx[0, 0]) * (bx[0, 3] - bx[0, 1])
            area_j = (
                (sub_boxes[rest, 2] - sub_boxes[rest, 0])
                * (sub_boxes[rest, 3] - sub_boxes[rest, 1])
            )
            union = area_i + area_j - inter
            union = np.maximum(union, 1e-9)
            iou = inter / union
            order = rest[iou <= iou_threshold]

        keep_all.extend(int(idxs[j]) for j in picked_local)

    return np.asarray(sorted(keep_all), dtype=np.int64)


def dedupe_one_per_class(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    """每个类别只保留置信度最高的一个框。"""
    if boxes_xyxy.size == 0:
        return np.zeros((0,), dtype=np.int64)
    keep: list[int] = []
    for cid in np.unique(class_ids):
        idxs = np.where(class_ids == cid)[0]
        best = idxs[int(scores[idxs].argmax())]
        keep.append(int(best))
    return np.asarray(sorted(keep), dtype=np.int64)


def clip_boxes_xyxy(
    boxes_xyxy: np.ndarray,
    width: int,
    height: int,
    *,
    max_attempts: int,
    enabled: bool,
) -> np.ndarray:
    """
    将 xyxy 裁剪到 [0,W]×[0,H]。多次迭代用于处理「先裁后面积变为 0」等边界情况：
    若某轮无变化则提前结束。
    """
    if not enabled or boxes_xyxy.size == 0:
        return boxes_xyxy.copy()
    b = boxes_xyxy.astype(np.float64).copy()
    w_f = float(width)
    h_f = float(height)
    prev = None
    for _ in range(max(1, max_attempts)):
        b[:, 0] = np.clip(b[:, 0], 0.0, w_f)
        b[:, 1] = np.clip(b[:, 1], 0.0, h_f)
        b[:, 2] = np.clip(b[:, 2], 0.0, w_f)
        b[:, 3] = np.clip(b[:, 3], 0.0, h_f)
        # 保证 x2>=x1, y2>=y1
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        b[:, 2] = np.maximum(x2, x1)
        b[:, 3] = np.maximum(y2, y1)
        if prev is not None and np.allclose(b, prev):
            break
        prev = b.copy()
    return b.astype(np.float32)


def draw_boxes_bgr(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    names: list[str],
) -> np.ndarray:
    """
    使用 Ultralytics Annotator 在 BGR 图像上绘制矩形与标签（与库内 predict 可视化风格一致）。
    """
    h, w = image_bgr.shape[:2]
    line_width = max(1, int(round(min(h, w) / 400)))
    im = image_bgr.copy()
    annotator = Annotator(im, line_width=line_width)
    for i in range(boxes_xyxy.shape[0]):
        cid = int(class_ids[i])
        name = names[cid] if 0 <= cid < len(names) else str(cid)
        label = f"{name} {float(scores[i]):.2f}"
        box = boxes_xyxy[i].tolist()
        color_bgr = ultra_colors(cid, bgr=True)
        annotator.box_label(box, label, color=color_bgr)
    # result() 在 cv2 模式下返回与输入一致的 BGR ndarray
    return np.asarray(annotator.result())


def yolo_result_to_arrays(result: "Results") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 ultralytics.engine.results.Results 提取 (xyxy, conf, cls_int)。
    若当前图无检测，返回空数组。
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = boxes.conf.cpu().numpy().astype(np.float32)
    cls = boxes.cls.cpu().numpy().astype(np.int64)
    return xyxy, conf, cls


def _result_names_ordered(result: "Results") -> list[str]:
    """Ultralytics names 为 dict(int->str)，按类别 id 排序组成列表。"""
    raw = getattr(result, "names", None)
    if not raw:
        return []
    if isinstance(raw, dict):
        return [raw[k] for k in sorted(raw)]
    return [str(x) for x in raw]


def _warn_if_layout_names_mismatch(names: list[str], layout_labels: list[str]) -> None:
    """训练/业务约定的五类名与当前权重 `names` 不一致时在 stderr 给出警告。"""
    if not names or not layout_labels:
        return
    sn, sl = set(names), set(layout_labels)
    if sn != sl:
        print(
            "[WARN] 模型 categories 与 layout_labels 集合不一致。\n"
            f"  模型: {sorted(sn)}\n"
            f"  配置: {sorted(sl)}",
            file=sys.stderr,
        )


def predict_five_dicts(
    image_path: str | Path,
    config: FiveDictsPredictConfig | None = None,
    *,
    repo_root: Path | None = None,
    model: YOLO | None = None,
    return_vis: bool = True,
) -> dict[str, Any]:
    """
    对单张截图推理并后处理，返回结构化结果（含可视化用 BGR 图与框）。

    Args:
        return_vis: 为 False 时不绘制检测框（``vis_bgr`` 为 ``None``），仅返回数组结果。

    Returns:
        dict 包含:
          - image_bgr: 原图 BGR
          - vis_bgr: 绘制后的 BGR；若 ``return_vis`` 为 False 则为 ``None``
          - boxes_xyxy, scores, class_ids
          - names: 模型类别名列表
          - result_raw: 原始 ultralytics Results（单张时的第一个元素）
    """
    cfg = config or FiveDictsPredictConfig()
    root = repo_root or REPO_ROOT
    img_path = Path(image_path).expanduser()
    if not img_path.is_file():
        raise FileNotFoundError(f"截图不存在: {img_path}")

    suffix = img_path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        print(
            f"[WARN] 扩展名 {suffix!r} 非常见截图格式，仍尝试按 OpenCV 读取。",
            file=sys.stderr,
        )

    if model is None:
        model_file = _resolve_model_path(root, cfg.model_path)
        if not model_file.is_file():
            raise FileNotFoundError(
                f"权重文件不存在: {model_file}（配置 model_path={cfg.model_path!r}）"
            )
        model = YOLO(str(model_file))

    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("需要 opencv-python 读取图像（可与 ultralytics 一并安装）。") from e

    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"无法读取图像: {img_path}")

    h, w = image_bgr.shape[:2]
    # 与 start_single_detect.py 一致：列表输入，得到 Results 列表
    results = model.predict(
        source=[str(img_path)],
        conf=cfg.confidence_threshold,
        iou=cfg.predict_nms_iou_threshold,
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO 未返回任何结果")
    result0 = results[0]

    names: list[str] = _result_names_ordered(result0)
    _warn_if_layout_names_mismatch(names, cfg.layout_labels)
    xyxy, conf, cls = yolo_result_to_arrays(result0)

    xyxy = clip_boxes_xyxy(
        xyxy,
        w,
        h,
        max_attempts=cfg.max_clip_attempts,
        enabled=cfg.prefer_clipping,
    )

    if cfg.enable_nms and xyxy.shape[0] > 0:
        keep = nms_xyxy_per_class(xyxy, conf, cls, cfg.nms_iou_threshold)
        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]

    if cfg.enable_duplicate_filtering and xyxy.shape[0] > 0:
        keep = dedupe_one_per_class(xyxy, conf, cls)
        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]

    if return_vis:
        vis_bgr = draw_boxes_bgr(image_bgr, xyxy, cls, conf, names)
    else:
        vis_bgr = None

    return {
        "image_bgr": image_bgr,
        "vis_bgr": vis_bgr,
        "boxes_xyxy": xyxy,
        "scores": conf,
        "class_ids": cls,
        "names": names,
        "result_raw": result0,
    }


def _list_images_in_folder(
    folder: Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    """
    列出文件夹下的图片文件；非递归时仅第一层，递归时使用 rglob。
    """
    if not folder.is_dir():
        raise NotADirectoryError(f"不是文件夹或不存在: {folder}")
    if recursive:
        candidates = folder.rglob("*")
    else:
        candidates = folder.iterdir()
    images = [
        p
        for p in candidates
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def auto_predict_five_dicts_folder(
    image_dir: str | Path,
    config: FiveDictsPredictConfig | None = None,
    *,
    out_dir: Path | None = None,
    repo_root: Path | None = None,
    recursive: bool = False,
    save_img: bool = True,
    save_txt: bool = False,
) -> tuple[int, int]:
    """
    遍历指定文件夹下的图片，逐张执行五分区检测，可视化结果写入 ``out_dir``。

    Args:
        image_dir: 图片所在目录。
        config: 推理配置；默认 ``FiveDictsPredictConfig()``。
        out_dir: 输出根目录；默认 ``DEFAULT_VIS_OUTPUT_DIR``（即 ``output/level1``）。
        repo_root: 解析相对权重路径用。
        recursive: 是否递归子目录。
        save_img: 是否保存 ``*_five_dicts_vis`` 图片。
        save_txt: 是否保存与源图同 ``stem`` 的 YOLO 格式 ``.txt``。

    Returns:
        (成功写入数, 失败数)。
    """
    import cv2

    cfg = config or FiveDictsPredictConfig()
    root = repo_root or REPO_ROOT
    dest = out_dir or DEFAULT_VIS_OUTPUT_DIR
    dest.mkdir(parents=True, exist_ok=True)

    folder = Path(image_dir).expanduser().resolve()
    paths = _list_images_in_folder(folder, recursive=recursive)
    if not paths:
        print(
            f"[WARN] 目录下未找到支持的图片 ({', '.join(sorted(IMAGE_EXTENSIONS))}): {folder}",
            file=sys.stderr,
        )
        return 0, 0

    model_file = _resolve_model_path(root, cfg.model_path)
    if not model_file.is_file():
        raise FileNotFoundError(
            f"权重文件不存在: {model_file}（配置 model_path={cfg.model_path!r}）"
        )
    model = YOLO(str(model_file))

    n_ok = 0
    n_fail = 0
    for img_path in paths:
        try:
            out = predict_five_dicts(
                img_path,
                cfg,
                repo_root=root,
                model=model,
                return_vis=save_img,
            )
            suf = img_path.suffix or ".png"
            if save_txt:
                ih, iw = out["image_bgr"].shape[:2]
                write_yolo_labels_txt(
                    dest / f"{img_path.stem}.txt",
                    out["class_ids"],
                    out["boxes_xyxy"],
                    iw,
                    ih,
                )
            if save_img:
                assert out["vis_bgr"] is not None
                out_path = dest / f"{img_path.stem}_five_dicts_vis{suf}"
                if not cv2.imwrite(str(out_path), out["vis_bgr"]):
                    raise RuntimeError(f"cv2.imwrite 失败: {out_path}")
            n_ok += 1
            parts = []
            if save_img:
                parts.append(f"{img_path.stem}_five_dicts_vis{suf}")
            if save_txt:
                parts.append(f"{img_path.stem}.txt")
            print(
                f"[OK] {img_path.name} -> {' + '.join(parts)} "
                f"(det={out['boxes_xyxy'].shape[0]})"
            )
        except Exception as e:  # noqa: BLE001 — 批量任务单张失败不中断
            n_fail += 1
            print(f"[FAIL] {img_path}: {e}", file=sys.stderr)

    return n_ok, n_fail


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="五分区 YOLO 预测并在截图上可视化。",
    )
    p.add_argument(
        "-i",
        "--image",
        default=str(
            REPO_ROOT
            / "data/images_origin/gardening_products_website_ui_design_492370171740540158.jpg"
        ),
        help="输入截图路径（.png / .jpg 等）；未指定时使用仓库内示例图",
    )
    p.add_argument(
        "-o",
        "--output",
        default="",
        help=(
            "可视化保存路径。留空则写入仓库 output/level1/<主名>_five_dicts_vis<后缀>；"
            "传入相对路径时相对于 output/level1；绝对路径则原样使用。"
        ),
    )
    p.add_argument(
        "--model",
        default="",
        help="覆盖默认权重路径（相对仓库根或绝对路径）",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=None,
        help="覆盖 confidence_threshold",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="保存后尝试用 OpenCV 弹窗显示（需要 GUI 环境）",
    )
    p.add_argument(
        "--img",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保存带框可视化图片（默认开启；仅要 txt 时可加 --no-img）",
    )
    p.add_argument(
        "--txt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="保存 YOLO 格式标签 txt（class xc yc w h 归一化；与源图 stem 同名）",
    )
    batch = p.add_argument_group("批量模式")
    batch.add_argument(
        "--auto",
        action="store_true",
        help="遍历 --input-dir 下图片并批量检测；结果写入 output/level1/",
    )
    batch.add_argument(
        "--input-dir",
        type=str,
        default="data/images_origin/",
        help="图片文件夹路径（与 --auto 联用；相对路径相对于仓库根目录）",
    )
    batch.add_argument(
        "--recursive",
        action="store_true",
        help="与 --auto 联用时递归子目录搜索图片",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FiveDictsPredictConfig()
    if args.model:
        cfg.model_path = args.model
    if args.conf is not None:
        cfg.confidence_threshold = args.conf

    if args.auto:
        if not str(args.input_dir).strip():
            print(
                "[ERROR] 使用 --auto 时必须指定 --input-dir <图片文件夹>",
                file=sys.stderr,
            )
            sys.exit(2)
        if not args.img and not args.txt:
            print(
                "[ERROR] 批量模式须至少开启 --img 或 --txt",
                file=sys.stderr,
            )
            sys.exit(2)
        in_dir = Path(args.input_dir).expanduser()
        if not in_dir.is_absolute():
            in_dir = (REPO_ROOT / in_dir).resolve()
        else:
            in_dir = in_dir.resolve()

        if args.show:
            print("[WARN] 批量模式（--auto）下忽略 --show", file=sys.stderr)

        ok, n_fail = auto_predict_five_dicts_folder(
            in_dir,
            cfg,
            out_dir=DEFAULT_VIS_OUTPUT_DIR,
            recursive=args.recursive,
            save_img=args.img,
            save_txt=args.txt,
        )
        print(f"[DONE] 批量完成：成功 {ok}，失败 {n_fail}，输出目录 {DEFAULT_VIS_OUTPUT_DIR}")
        return

    if not args.img and not args.txt and not args.show:
        print(
            "[ERROR] 须至少开启 --img、--txt 之一，或使用 --show",
            file=sys.stderr,
        )
        sys.exit(2)

    need_vis = args.img or args.show
    out = predict_five_dicts(args.image, cfg, return_vis=need_vis)

    img_in = Path(args.image).resolve()
    out_dir = DEFAULT_VIS_OUTPUT_DIR
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = (out_dir / out_path).resolve()
    else:
        stem = img_in.stem
        suf = img_in.suffix or ".png"
        out_path = out_dir / f"{stem}_five_dicts_vis{suf}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import cv2

    ih, iw = out["image_bgr"].shape[:2]
    if args.txt:
        txt_p = out_path.parent / f"{img_in.stem}.txt"
        write_yolo_labels_txt(
            txt_p,
            out["class_ids"],
            out["boxes_xyxy"],
            iw,
            ih,
        )
        print(f"[OK] 已保存标签: {txt_p}")

    if args.img:
        if out["vis_bgr"] is None:
            raise RuntimeError("内部错误：未生成可视化图")
        if not cv2.imwrite(str(out_path), out["vis_bgr"]):
            raise RuntimeError(f"写入失败: {out_path}")
        print(f"[OK] 已保存可视化: {out_path}")

    print(
        f"[INFO] 检测数: {out['boxes_xyxy'].shape[0]}  "
        f"类别映射: {out['names']}"
    )

    if args.show:
        if out["vis_bgr"] is None:
            raise RuntimeError("内部错误：--show 需要可视化图")
        cv2.imshow("five_dicts", out["vis_bgr"])
        print("[INFO] 按任意键关闭窗口…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
