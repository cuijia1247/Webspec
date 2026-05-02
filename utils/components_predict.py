# -*- coding: utf-8 -*-
"""
网页截图 **UI 组件** YOLO 检测（类别见 ``pretrainedModels/yolo/models/ui_tag_data.yaml``），
并将检测框画回图像。

用法与 ``five_dicts_predict.py`` 类似：单张 ``-i`` 或批量 ``--auto --input-dir``；
默认可视化保存到仓库 ``output/components/``。

在仓库根目录 Webspec 下执行（建议 ``conda activate webspec``，需 ``ultralytics``、``opencv-python``、``pyyaml``）::

    python utils/components_predict.py -i path/to/screenshot.png
    python utils/components_predict.py --auto --input-dir data/images_origin
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# 仓库根目录：.../Webspec；优先加载 ``pretrainedModels/yolo`` 下 vendored ultralytics，
# 并与旧权重中的 ``auto_component`` 包名对齐（见同目录 ``register_auto_component_alias``）。
REPO_ROOT = Path(__file__).resolve().parent.parent
_UTILS_DIR = Path(__file__).resolve().parent
_YOLO_VENDOR_ROOT = REPO_ROOT / "pretrainedModels" / "yolo"
if _YOLO_VENDOR_ROOT.is_dir() and str(_YOLO_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_YOLO_VENDOR_ROOT))
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from register_auto_component_alias import ensure_auto_component_alias  # noqa: E402

ensure_auto_component_alias()

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.plotting import colors as ultra_colors  # noqa: E402
from yolo_label_export import write_yolo_labels_txt  # noqa: E402

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]
DEFAULT_VIS_OUTPUT_DIR = REPO_ROOT / "output" / "components"
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


@dataclass
class ComponentsPredictConfig:
    """组件检测配置（与 ``five_dicts_predict`` 对齐字段含义，默认不做「每类只留一个框」）。"""

    model_path: str = "pretrainedModels/yolo/models/best.pt"
    # 训练数据 YAML，仅用于解析参考类别名、与模型 names 一致性告警（不参与推理）
    data_yaml_path: str = "pretrainedModels/yolo/models/ui_tag_data.yaml"
    confidence_threshold: float = 0.25
    predict_nms_iou_threshold: float = 0.7
    max_det: int = 300
    # 组件密集场景一般只依赖 YOLO 内置 NMS；如需二次按类 NMS 可打开
    enable_nms: bool = False
    nms_iou_threshold: float = 0.45
    # 组件可出现多个同类实例，默认关闭「每类只保留最高分框」
    enable_duplicate_filtering: bool = False
    max_clip_attempts: int = 4
    prefer_clipping: bool = True


def _resolve_model_path(repo_root: Path, model_path: str) -> Path:
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


def load_reference_names_from_data_yaml(yaml_path: Path) -> list[str]:
    """
    从 Ultralytics 数据 YAML 中读取 ``names``，按类别 id 排序为列表。
    依赖 PyYAML（ultralytics 通常已带入）。
    """
    if not yaml_path.is_file():
        return []
    if yaml is None:
        print(
            "[WARN] 未安装 PyYAML，无法读取 data yaml 中的 names；pip install pyyaml",
            file=sys.stderr,
        )
        return []
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, dict):
        keys: list[int] = []
        for k in names:
            try:
                keys.append(int(k))
            except (TypeError, ValueError):
                continue
        return [str(names[k]) for k in sorted(keys)]
    if isinstance(names, list):
        return [str(x) for x in names]
    return []


def nms_xyxy_per_class(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """对每个类别分别做标准 NMS，返回保留下标。"""
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
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        b[:, 2] = np.maximum(x2, x1)
        b[:, 3] = np.maximum(y2, y1)
        if prev is not None and np.allclose(b, prev):
            break
        prev = b.copy()
    return b.astype(np.float32)


def _find_unicode_font_path() -> Path | None:
    """
    在系统目录查找可渲染中文的 TrueType/OpenType 字体（本地文件，不访问网络）。
    找不到则返回 None，绘制将退化为「类 id + 置信度」的 ASCII 标签。
    """
    roots = (
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path("/System/Library/Fonts"),  # macOS
    )
    patterns = (
        "**/NotoSansCJK*.ttc",
        "**/NotoSansCJK*.otf",
        "**/NotoSansSC*.otf",
        "**/NotoSansSC*.ttf",
        "**/wqy-microhei*.ttc",
        "**/WenQuanYi*.ttf",
        "**/SourceHanSans*.otf",
        "**/DroidSansFallback*.ttf",
    )
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
    return None


def draw_boxes_bgr(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    names: list[str],
) -> np.ndarray:
    """
    绘制检测框与标签。优先使用 **本地中文字体 + PIL**，避免 Ultralytics ``Annotator``
    在初始化时调用 ``check_font`` 访问 GitHub 下载字体导致离线/慢网络卡死。
    若无可用字体，退化为 OpenCV + 纯 ASCII 标签 ``{类id} {置信度}``。
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    n = boxes_xyxy.shape[0]
    if n == 0:
        return image_bgr.copy()

    h, w = image_bgr.shape[:2]
    line_width = max(1, int(round(min(h, w) / 400)))
    font_path = _find_unicode_font_path()
    pil_font: Any = None
    if font_path is not None:
        try:
            font_size = max(14, min(h, w) // 50)
            pil_font = ImageFont.truetype(str(font_path), font_size)
        except OSError:
            pil_font = None

    if pil_font is not None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im_pil)
        for i in range(n):
            cid = int(class_ids[i])
            x1, y1, x2, y2 = (int(round(v)) for v in boxes_xyxy[i].tolist())
            color_bgr = ultra_colors(cid, bgr=True)
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            name = names[cid] if 0 <= cid < len(names) else str(cid)
            label = f"{name} {float(scores[i]):.2f}"
            dr.rectangle([x1, y1, x2, y2], outline=color_rgb, width=line_width)
            bx = dr.textbbox((x1, y1), label, font=pil_font)
            tw, th = bx[2] - bx[0], bx[3] - bx[1]
            pad = 2
            ty = max(0, y1 - th - 2 * pad - 2)
            dr.rectangle(
                [x1, ty, x1 + tw + 2 * pad, ty + th + 2 * pad],
                fill=color_rgb,
            )
            dr.text(
                (x1 + pad, ty + pad),
                label,
                fill=(255, 255, 255),
                font=pil_font,
            )
        return cv2.cvtColor(np.asarray(im_pil), cv2.COLOR_RGB2BGR)

    if not getattr(draw_boxes_bgr, "_warned_ascii_fallback", False):
        print(
            "[INFO] 未检测到本地中文字体（如 fonts-noto-cjk），"
            "标签将以「类id 置信度」显示；安装系统 CJK 字体后可显示中文类名。",
            file=sys.stderr,
        )
        draw_boxes_bgr._warned_ascii_fallback = True

    out = image_bgr.copy()
    tf = max(line_width - 1, 1)
    sf = line_width / 3.0
    for i in range(n):
        cid = int(class_ids[i])
        x1, y1, x2, y2 = (int(round(v)) for v in boxes_xyxy[i].tolist())
        color_bgr = ultra_colors(cid, bgr=True)
        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            color_bgr,
            line_width,
            cv2.LINE_AA,
        )
        label = f"{cid} {float(scores[i]):.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            sf,
            tf,
        )
        ty = max(0, y1 - th - baseline - 4)
        cv2.rectangle(out, (x1, ty), (x1 + tw + 2, y1), color_bgr, -1, cv2.LINE_AA)
        cv2.putText(
            out,
            label,
            (x1 + 1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            sf,
            (255, 255, 255),
            tf,
            cv2.LINE_AA,
        )
    return out


def yolo_result_to_arrays(result: "Results") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    raw = getattr(result, "names", None)
    if not raw:
        return []
    if isinstance(raw, dict):
        return [raw[k] for k in sorted(raw)]
    return [str(x) for x in raw]


def _warn_if_names_mismatch(names: list[str], reference: list[str]) -> None:
    if not names or not reference:
        return
    if names == reference:
        return
    print(
        "[WARN] 模型输出 names 与 data yaml 中 names 列表不一致（顺序或内容）。\n"
        f"  模型({len(names)}): {names[:8]}{'...' if len(names) > 8 else ''}\n"
        f"  参考({len(reference)}): {reference[:8]}{'...' if len(reference) > 8 else ''}",
        file=sys.stderr,
    )


def predict_components(
    image_path: str | Path,
    config: ComponentsPredictConfig | None = None,
    *,
    repo_root: Path | None = None,
    model: YOLO | None = None,
    return_vis: bool = True,
) -> dict[str, Any]:
    """
    单张截图组件检测并后处理；返回 ``image_bgr``、``vis_bgr``、框数组及 ``result_raw``。

    Args:
        return_vis: 为 False 时不绘制框图，``vis_bgr`` 为 ``None``。
    """
    cfg = config or ComponentsPredictConfig()
    root = repo_root or REPO_ROOT
    img_path = Path(image_path).expanduser()
    if not img_path.is_file():
        raise FileNotFoundError(f"图片不存在: {img_path}")

    suffix = img_path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        print(
            f"[WARN] 扩展名 {suffix!r} 非常见图片格式，仍尝试读取。",
            file=sys.stderr,
        )

    data_yaml = _resolve_model_path(root, cfg.data_yaml_path)
    ref_names = load_reference_names_from_data_yaml(data_yaml)

    if model is None:
        model_file = _resolve_model_path(root, cfg.model_path)
        if not model_file.is_file():
            raise FileNotFoundError(
                f"权重不存在: {model_file}（model_path={cfg.model_path!r}）"
            )
        model = YOLO(str(model_file))

    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("需要 opencv-python。") from e

    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"无法读取图像: {img_path}")

    h, w = image_bgr.shape[:2]
    results = model.predict(
        source=[str(img_path)],
        conf=cfg.confidence_threshold,
        iou=cfg.predict_nms_iou_threshold,
        max_det=cfg.max_det,
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO 未返回任何结果")
    result0 = results[0]

    names = _result_names_ordered(result0)
    _warn_if_names_mismatch(names, ref_names)

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
        "reference_names": ref_names,
        "result_raw": result0,
    }


def _list_images_in_folder(folder: Path, *, recursive: bool = False) -> list[Path]:
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


def auto_predict_components_folder(
    image_dir: str | Path,
    config: ComponentsPredictConfig | None = None,
    *,
    out_dir: Path | None = None,
    repo_root: Path | None = None,
    recursive: bool = False,
    save_img: bool = True,
    save_txt: bool = False,
) -> tuple[int, int]:
    """批量组件检测；结果写入 ``output/components``（或 ``out_dir``）。"""
    import cv2

    cfg = config or ComponentsPredictConfig()
    root = repo_root or REPO_ROOT
    dest = out_dir or DEFAULT_VIS_OUTPUT_DIR
    dest.mkdir(parents=True, exist_ok=True)

    folder = Path(image_dir).expanduser().resolve()
    paths = _list_images_in_folder(folder, recursive=recursive)
    if not paths:
        print(
            f"[WARN] 未找到图片: {folder}",
            file=sys.stderr,
        )
        return 0, 0

    model_file = _resolve_model_path(root, cfg.model_path)
    if not model_file.is_file():
        raise FileNotFoundError(
            f"权重不存在: {model_file}（model_path={cfg.model_path!r}）"
        )
    model = YOLO(str(model_file))

    n_ok = 0
    n_fail = 0
    for img_path in paths:
        try:
            suf = img_path.suffix or ".png"
            out = predict_components(
                img_path,
                cfg,
                repo_root=root,
                model=model,
                return_vis=save_img,
            )
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
                out_path = dest / f"{img_path.stem}_components_vis{suf}"
                if not cv2.imwrite(str(out_path), out["vis_bgr"]):
                    raise RuntimeError(f"cv2.imwrite 失败: {out_path}")
            n_ok += 1
            parts: list[str] = []
            if save_img:
                parts.append(f"{img_path.stem}_components_vis{suf}")
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
    p = argparse.ArgumentParser(description="UI 组件 YOLO 检测并可视化（参考 ui_tag_data.yaml 类别）。")
    p.add_argument(
        "-i",
        "--image",
        default=str(
            REPO_ROOT
            / "data/images_origin/gardening_products_website_ui_design_492370171740540158.jpg"
        ),
        help="输入图片；未指定时为仓库内示例路径（若不存在请换 -i）",
    )
    p.add_argument(
        "-o",
        "--output",
        default="",
        help=(
            "保存路径。留空为 output/components/<主名>_components_vis<后缀>；"
            "相对路径相对于 output/components；绝对路径原样使用。"
        ),
    )
    p.add_argument("--model", default="", help="覆盖权重路径（相对仓库根或绝对路径）")
    p.add_argument(
        "--data-yaml",
        default="",
        help="覆盖数据 YAML（用于参考 names 校验；默认 pretrainedModels/yolo/models/ui_tag_data.yaml）",
    )
    p.add_argument("--conf", type=float, default=None, help="confidence 阈值")
    p.add_argument(
        "--show",
        action="store_true",
        help="保存后用 OpenCV 弹窗显示（单张模式）",
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
        help="保存 YOLO 格式标签 txt（与源图 stem 同名）",
    )
    batch = p.add_argument_group("批量")
    batch.add_argument(
        "--auto",
        action="store_true",
        help="遍历 --input-dir 下图片，输出到 output/components/",
    )
    batch.add_argument(
        "--input-dir",
        type=str,
        default="data/images_origin/",
        help="图片目录（与 --auto 联用；相对路径相对于仓库根）",
    )
    batch.add_argument("--recursive", action="store_true", help="递归子目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ComponentsPredictConfig()
    if args.model:
        cfg.model_path = args.model
    if args.data_yaml:
        cfg.data_yaml_path = args.data_yaml
    if args.conf is not None:
        cfg.confidence_threshold = args.conf

    if args.auto:
        if not str(args.input_dir).strip():
            print("[ERROR] --auto 需指定 --input-dir", file=sys.stderr)
            sys.exit(2)
        if not args.img and not args.txt:
            print(
                "[ERROR] 批量模式须至少开启 --img 或 --txt",
                file=sys.stderr,
            )
            sys.exit(2)
        in_dir = Path(args.input_dir).expanduser()
        in_dir = in_dir.resolve() if in_dir.is_absolute() else (REPO_ROOT / in_dir).resolve()
        if args.show:
            print("[WARN] 批量模式忽略 --show", file=sys.stderr)
        ok, n_fail = auto_predict_components_folder(
            in_dir,
            cfg,
            out_dir=DEFAULT_VIS_OUTPUT_DIR,
            recursive=args.recursive,
            save_img=args.img,
            save_txt=args.txt,
        )
        print(f"[DONE] 成功 {ok}，失败 {n_fail}，目录 {DEFAULT_VIS_OUTPUT_DIR}")
        return

    if not args.img and not args.txt and not args.show:
        print(
            "[ERROR] 须至少开启 --img、--txt 之一，或使用 --show",
            file=sys.stderr,
        )
        sys.exit(2)

    need_vis = args.img or args.show
    out = predict_components(args.image, cfg, return_vis=need_vis)

    img_in = Path(args.image).resolve()

    out_base = DEFAULT_VIS_OUTPUT_DIR
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = (out_base / out_path).resolve()
    else:
        suf = img_in.suffix or ".png"
        out_path = out_base / f"{img_in.stem}_components_vis{suf}"

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
        print(f"[OK] 已保存: {out_path}")
    print(f"[INFO] 检测数: {out['boxes_xyxy'].shape[0]}")

    if args.show:
        if out["vis_bgr"] is None:
            raise RuntimeError("内部错误：--show 需要可视化图")
        cv2.imshow("components", out["vis_bgr"])
        print("[INFO] 按键关闭窗口…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
