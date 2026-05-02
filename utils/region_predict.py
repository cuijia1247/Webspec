# -*- coding: utf-8 -*-
"""
网页截图 **Level-2 组件分区** YOLO 检测：5 类
``comp_body`` / ``comp_footer`` / ``comp_header`` / ``comp_leftsider`` / ``comp_rightsider``，
并在原图上绘制框与标签。

推理与后处理流程与 ``five_dicts_predict.py`` 一致；默认权重 ``pretrainedModels/level2/best.pt``，
结果写入 ``output/level2/``。

**类别 id 与标签（约定与训练 data yaml 中 ``names`` 的 0–4 键顺序一致；本脚本以
``LEVEL2_LAYOUT_LABELS`` 下标为准，推理时实际显示名以权重 ``model.names`` 为准）：**

===========  ======================================
``class_id``  ``label``（英文 slug）
===========  ======================================
0            ``comp_body``（主体/主内容区）
1            ``comp_footer``（页脚区）
2            ``comp_header``（页头区）
3            ``comp_leftsider``（左侧栏）
4            ``comp_rightsider``（右侧栏）
===========  ======================================

若部署的权重 ``names`` 顺序与上表不一致，应以权重为准并相应调整训练配置或本脚本的
``layout_labels``；运行时会通过 ``_warn_if_layout_names_mismatch`` 在 stderr 提示。

用法（在仓库根目录 Webspec 下执行）::

    conda activate webspec
    python utils/region_predict.py -i path/to/screenshot.png
    python utils/region_predict.py --auto --input-dir data/images_origin
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ultralytics import YOLO

from yolo_label_export import write_yolo_labels_txt

# 复用与 five_dicts_predict 相同的工具与常量
from five_dicts_predict import (  # noqa: I001
    IMAGE_EXTENSIONS,
    REPO_ROOT,
    clip_boxes_xyxy,
    dedupe_one_per_class,
    draw_boxes_bgr,
    nms_xyxy_per_class,
    yolo_result_to_arrays,
    _list_images_in_folder,
    _resolve_model_path,
    _result_names_ordered,
    _warn_if_layout_names_mismatch,
)

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

# 默认可视化目录：Level-2
DEFAULT_VIS_OUTPUT_DIR = REPO_ROOT / "output" / "level2"

# Level-2 五类标签（与 level1 五分区语义对应；下标即约定 class_id 0–4）
#   0 comp_body, 1 comp_footer, 2 comp_header, 3 comp_leftsider, 4 comp_rightsider
# 须与训练 data yaml ``names`` 及权重 ``names`` 一致，详见模块顶部文档。
LEVEL2_LAYOUT_LABELS: tuple[str, ...] = (
    "comp_body",
    "comp_footer",
    "comp_header",
    "comp_leftsider",
    "comp_rightsider",
)


@dataclass
class RegionPredictConfig:
    """Level-2 五分区检测配置（与 five_dicts 类似：每类最多保留置信度最高的一个框）。"""

    model_path: str = "pretrainedModels/level2/best.pt"
    confidence_threshold: float = 0.25
    predict_nms_iou_threshold: float = 0.7
    enable_nms: bool = True
    enable_duplicate_filtering: bool = False
    nms_iou_threshold: float = 0.001
    max_clip_attempts: int = 4
    prefer_clipping: bool = True
    layout_labels: list[str] = field(
        default_factory=lambda: list(LEVEL2_LAYOUT_LABELS),
    )


def predict_region(
    image_path: str | Path,
    config: RegionPredictConfig | None = None,
    *,
    repo_root: Path | None = None,
    model: YOLO | None = None,
    return_vis: bool = True,
) -> dict[str, Any]:
    """
    对单张截图做 Level-2 五类 comp_* 区域检测并后处理。

    Returns:
        dict: image_bgr, vis_bgr, boxes_xyxy, scores, class_ids, names, result_raw
    """
    cfg = config or RegionPredictConfig()
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
        raise RuntimeError("需要 opencv-python 读取图像。") from e

    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"无法读取图像: {img_path}")

    h, w = image_bgr.shape[:2]
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


def auto_predict_region_folder(
    image_dir: str | Path,
    config: RegionPredictConfig | None = None,
    *,
    out_dir: Path | None = None,
    repo_root: Path | None = None,
    recursive: bool = False,
    save_img: bool = True,
    save_txt: bool = False,
) -> tuple[int, int]:
    """批量 Level-2 五类检测；可视化写入 ``output/level2`` 或 ``out_dir``。"""
    import cv2

    cfg = config or RegionPredictConfig()
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
            out = predict_region(
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
                out_path = dest / f"{img_path.stem}_region_vis{suf}"
                if not cv2.imwrite(str(out_path), out["vis_bgr"]):
                    raise RuntimeError(f"cv2.imwrite 失败: {out_path}")
            n_ok += 1
            parts: list[str] = []
            if save_img:
                parts.append(f"{img_path.stem}_region_vis{suf}")
            if save_txt:
                parts.append(f"{img_path.stem}.txt")
            print(
                f"[OK] {img_path.name} -> {' + '.join(parts)} "
                f"(det={out['boxes_xyxy'].shape[0]})"
            )
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[FAIL] {img_path}: {e}", file=sys.stderr)

    return n_ok, n_fail


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Level-2 五类分区 YOLO 预测（comp_body/footer/header/leftsider/rightsider）并可视化。"
        ),
    )
    p.add_argument(
        "-i",
        "--image",
        default=str(
            REPO_ROOT
            / "data/images_origin/gardening_products_website_ui_design_492370171740540158.jpg"
        ),
        help="输入截图路径；未指定时使用仓库内示例图",
    )
    p.add_argument(
        "-o",
        "--output",
        default="",
        help=(
            "可视化保存路径。留空则写入 output/level2/<主名>_region_vis<后缀>；"
            "相对路径相对于 output/level2；绝对路径原样使用。"
        ),
    )
    p.add_argument("--model", default="", help="覆盖默认权重（相对仓库根或绝对路径）")
    p.add_argument("--conf", type=float, default=None, help="覆盖 confidence_threshold")
    p.add_argument(
        "--show",
        action="store_true",
        help="保存后用 OpenCV 弹窗显示（单张模式）",
    )
    p.add_argument(
        "--img",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保存带框图（默认开启；仅要 txt 时可 --no-img）",
    )
    p.add_argument(
        "--txt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="保存 YOLO 格式标签 txt（与源图 stem 同名）",
    )
    batch = p.add_argument_group("批量模式")
    batch.add_argument(
        "--auto",
        action="store_true",
        help="遍历 --input-dir 下图片；结果写入 output/level2/",
    )
    batch.add_argument(
        "--input-dir",
        type=str,
        default="data/images_origin/",
        help="图片文件夹（与 --auto 联用；相对路径相对于仓库根）",
    )
    batch.add_argument("--recursive", action="store_true", help="递归子目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RegionPredictConfig()
    if args.model:
        cfg.model_path = args.model
    if args.conf is not None:
        cfg.confidence_threshold = args.conf

    if args.auto:
        if not str(args.input_dir).strip():
            print("[ERROR] --auto 需指定 --input-dir", file=sys.stderr)
            sys.exit(2)
        if not args.img and not args.txt:
            print("[ERROR] 批量模式须至少开启 --img 或 --txt", file=sys.stderr)
            sys.exit(2)
        in_dir = Path(args.input_dir).expanduser()
        if not in_dir.is_absolute():
            in_dir = (REPO_ROOT / in_dir).resolve()
        else:
            in_dir = in_dir.resolve()

        if args.show:
            print("[WARN] 批量模式忽略 --show", file=sys.stderr)

        ok, n_fail = auto_predict_region_folder(
            in_dir,
            cfg,
            out_dir=DEFAULT_VIS_OUTPUT_DIR,
            recursive=args.recursive,
            save_img=args.img,
            save_txt=args.txt,
        )
        print(f"[DONE] 成功 {ok}，失败 {n_fail}，输出目录 {DEFAULT_VIS_OUTPUT_DIR}")
        return

    if not args.img and not args.txt and not args.show:
        print(
            "[ERROR] 须至少开启 --img、--txt 之一，或使用 --show",
            file=sys.stderr,
        )
        sys.exit(2)

    need_vis = args.img or args.show
    out = predict_region(args.image, cfg, return_vis=need_vis)

    img_in = Path(args.image).resolve()
    out_dir = DEFAULT_VIS_OUTPUT_DIR
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = (out_dir / out_path).resolve()
    else:
        stem = img_in.stem
        suf = img_in.suffix or ".png"
        out_path = out_dir / f"{stem}_region_vis{suf}"

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
        f"[INFO] 检测数: {out['boxes_xyxy'].shape[0]}  类别: {out['names']}"
    )

    if args.show:
        if out["vis_bgr"] is None:
            raise RuntimeError("内部错误：--show 需要可视化图")
        cv2.imshow("level2_comp_regions", out["vis_bgr"])
        print("[INFO] 按任意键关闭窗口…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
