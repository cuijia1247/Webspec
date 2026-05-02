# -*- coding: utf-8 -*-
"""
组件双向匹配相似度（Component Similarity, CS）。

1. 对 source / target 图像分别调用 ``utils.components_predict.predict_components`` 做组件检测，
   并将 YOLO 格式标签 txt 写入 ``output/components``。
2. com_tar：对每个 target 检测框（映射到 source 坐标系），在 source 中寻找 **同类** 框且与本区域有足够 IoU，
   成功数 / target 组件总数。
3. com_source：对称地，对每个 source 框在 target 中匹配同类与 IoU。
4. CS = com_tar + com_source。

YOLO txt 格式：每行 ``class_id x_center y_center width height``（相对整图宽高 0~1）。

用法（在仓库 Webspec 根目录）::

    python component_similarity_calculation.py -s path/a.png -t path/b.jpg
    python component_similarity_calculation.py --dir
        # 与 layout_similarity_calculation.py 相同：配对 --source-dir/--target-dir 下同 stem（后缀可不同）的图片，
        # 标签写在 ``cs_dir_<source-dir名>/<stem>/``；整批明细与均值写入同批根目录 ``cs_result.md``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from dataclasses import dataclass

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
_UTILS_DIR = REPO_ROOT / "utils"
_YOLO_VENDOR_ROOT = REPO_ROOT / "pretrainedModels" / "yolo"
if _YOLO_VENDOR_ROOT.is_dir() and str(_YOLO_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_YOLO_VENDOR_ROOT))
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from register_auto_component_alias import ensure_auto_component_alias  # noqa: E402

ensure_auto_component_alias()

from components_predict import (  # noqa: E402
    IMAGE_EXTENSIONS,
    ComponentsPredictConfig,
    predict_components,
    _resolve_model_path,
)
from ultralytics import YOLO  # noqa: E402
from yolo_label_export import write_yolo_labels_txt  # noqa: E402

# 检测结果输出目录（与本仓库 components_predict 一致）
DEFAULT_OUT_DIR = REPO_ROOT / "output" / "components"


def _xyxy_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """两轴对齐框 IoU，box 为 [x1,y1,x2,y2]。"""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = a + b - inter
    return inter / union if union > 0 else 0.0


def _map_bbox_to_other_image(
    xyxy_tgt: np.ndarray,
    fw: int,
    fh: int,
    tw: int,
    th: int,
) -> np.ndarray:
    """
    将「第一幅图」像素框按比例映射到第二幅图画布。
    fw,fh → 框所在图尺寸；tw,th → 目标图尺寸。
    """
    if fw <= 0 or fh <= 0:
        raise ValueError("无效源图宽高")
    sx = tw / fw
    sy = th / fh
    out = xyxy_tgt.astype(np.float64).copy()
    tf, thf = float(tw), float(th)
    out[[0, 2]] *= sx
    out[[1, 3]] *= sy
    out[[0, 2]] = np.clip(out[[0, 2]], 0.0, tf)
    out[[1, 3]] = np.clip(out[[1, 3]], 0.0, thf)
    if out[2] < out[0]:
        out[0], out[2] = out[2], out[0]
    if out[3] < out[1]:
        out[1], out[3] = out[3], out[1]
    return out.astype(np.float32)


def predict_and_save_txt(
    image_path: Path,
    *,
    txt_path: Path,
    cfg: ComponentsPredictConfig,
    root: Path,
    model: YOLO | None,
) -> dict[str, Any]:
    """检测并写入 YOLO txt；返回 predict_components 同款 dict。"""
    out = predict_components(
        image_path,
        cfg,
        repo_root=root,
        model=model,
        return_vis=False,
    )
    h, w = out["image_bgr"].shape[:2]
    write_yolo_labels_txt(
        txt_path,
        out["class_ids"],
        out["boxes_xyxy"],
        w,
        h,
    )
    return out


def match_ratio(
    dets_primary: tuple[np.ndarray, np.ndarray],
    dets_secondary: tuple[np.ndarray, np.ndarray],
    sw: int,
    sh: int,
    pw: int,
    ph: int,
    *,
    iou_threshold: float,
) -> float:
    """
    对 primary 中每个框，映射到 (sw,sh) 画布，与 secondary 中 **同类** 框比较 IoU。
    primary 来自尺寸 (pw,ph) 的图；映射到 secondary 的尺寸 (sw,sh)。
    返回：成功匹配数 / primary 框数；若 primary 无框则返回 1.0（空集不产生惩罚）。
    """
    xyxy_p, cls_p = dets_primary
    xyxy_s, cls_s = dets_secondary
    n_p = xyxy_p.shape[0]
    if n_p == 0:
        return 1.0
    hits = 0
    for i in range(n_p):
        cid = int(cls_p[i])
        mapped = _map_bbox_to_other_image(xyxy_p[i], pw, ph, sw, sh)
        best_iou = 0.0
        for j in range(xyxy_s.shape[0]):
            if int(cls_s[j]) != cid:
                continue
            best_iou = max(best_iou, _xyxy_iou(mapped, xyxy_s[j]))
            if best_iou >= iou_threshold:
                break
        if best_iou >= iou_threshold:
            hits += 1
    return hits / n_p


def write_cs_markdown(
    path: Path,
    *,
    source_img: Path,
    target_img: Path,
    source_txt: Path,
    target_txt: Path,
    com_tar: float,
    com_source: float,
    cs: float,
    iou_thr: float,
    n_src: int,
    n_tgt: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 组件双向匹配结果（CS）\n",
        "\n",
        "## 输入\n\n",
        f"- source stem: `{Path(source_img).stem}`\n",
        f"- target stem: `{Path(target_img).stem}`\n",
        "\n",
        "## 标签文件（YOLO 检测格式）\n\n",
        f"- source txt: `{source_txt}`\n",
        f"- target txt: `{target_txt}`\n",
        "\n",
        "## 指标定义\n\n",
        "- **com_tar**：对每个 target 框（线性映射至 source 尺寸），若在 source "
        "中存在 **同类（class id 相同）** 且与其 IoU≥阈值的框，则计为成功；"
        "com_tar = 成功数 / target 组件数。\n",
        "- **com_source**：对每个 source 框映射至 target，对称定义；"
        "com_source = 成功数 / source 组件数。\n",
        "- **CS** = com_tar + com_source。\n",
        f"- 匹配 IoU 阈值：**{iou_thr:g}**\n",
        "\n",
        "## 数值\n\n",
        f"| 项 | 值 |\n| --- | --- |\n",
        f"| source 检测数 | {n_src} |\n",
        f"| target 检测数 | {n_tgt} |\n",
        f"| com_tar | {com_tar:.6f} |\n",
        f"| com_source | {com_source:.6f} |\n",
        f"| **CS** | **{cs:.6f}** |\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


@dataclass
class CSPairResult:
    """单对图像组件匹配数值与产物路径。"""

    stem: str
    source_img: Path
    target_img: Path
    source_txt: Path
    target_txt: Path
    result_md: Path | None
    com_tar: float
    com_source: float
    cs: float
    n_src: int
    n_tgt: int


def collect_same_stem_image_pairs(
    source_dir: str | Path,
    target_dir: str | Path,
) -> list[tuple[Path, Path]]:
    """
    列出两目录下文件名（stem）相同、后缀可不同的图片对；逻辑与 layout_similarity_calculation.batch_compute_ls 一致。
    仅遍历一层子目录（非递归）。扩展名见 ``IMAGE_EXTENSIONS``。
    """
    src_dir = Path(source_dir).expanduser().resolve()
    tgt_dir = Path(target_dir).expanduser().resolve()
    tgt_map: dict[str, Path] = {}
    for p in sorted(tgt_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            tgt_map.setdefault(p.stem, p)
    pairs: list[tuple[Path, Path]] = []
    for p in sorted(src_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            if p.stem in tgt_map:
                pairs.append((p, tgt_map[p.stem]))
    return pairs


def compute_cs_for_pair(
    src_img: Path,
    tgt_img: Path,
    *,
    out_dir: Path,
    repo_root: Path,
    cfg: ComponentsPredictConfig,
    model: YOLO,
    iou_thr: float,
    result_md: Path | None = None,
    write_markdown: bool = True,
) -> CSPairResult:
    """
    对一对图像检测、写 YOLO txt、算 com_tar / com_source / CS；可选写单对 markdown。

    Args:
        result_md: 单图报告路径；若为 None 且 write_markdown 为 True，则 ``out_dir / cs_result.md``。
        write_markdown: 为 False 时（批量模式由 ``cs_result.md`` 汇总）不写单对 md。
    """
    stem = src_img.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path: Path | None
    if write_markdown:
        md_path = result_md if result_md is not None else (out_dir / "cs_result.md")
    else:
        md_path = None

    # 分别以源/目标文件名区分标签；stem 一致时后缀 _source/_target 可避免覆盖。
    src_txt = out_dir / f"cs_labels_{src_img.stem}_source.txt"
    tgt_txt = out_dir / f"cs_labels_{tgt_img.stem}_target.txt"

    out_src = predict_and_save_txt(
        src_img, txt_path=src_txt, cfg=cfg, root=repo_root, model=model
    )
    out_tgt = predict_and_save_txt(
        tgt_img, txt_path=tgt_txt, cfg=cfg, root=repo_root, model=model
    )

    xyxy_src = out_src["boxes_xyxy"]
    cls_src = out_src["class_ids"]
    h_s, w_s = out_src["image_bgr"].shape[:2]
    xyxy_tgt = out_tgt["boxes_xyxy"]
    cls_tgt = out_tgt["class_ids"]
    h_t, w_t = out_tgt["image_bgr"].shape[:2]

    com_tar = match_ratio(
        (xyxy_tgt, cls_tgt),
        (xyxy_src, cls_src),
        w_s,
        h_s,
        w_t,
        h_t,
        iou_threshold=iou_thr,
    )
    com_source = match_ratio(
        (xyxy_src, cls_src),
        (xyxy_tgt, cls_tgt),
        w_t,
        h_t,
        w_s,
        h_s,
        iou_threshold=iou_thr,
    )
    cs = com_tar + com_source

    if md_path is not None:
        write_cs_markdown(
            md_path,
            source_img=src_img,
            target_img=tgt_img,
            source_txt=src_txt,
            target_txt=tgt_txt,
            com_tar=com_tar,
            com_source=com_source,
            cs=cs,
            iou_thr=iou_thr,
            n_src=int(xyxy_src.shape[0]),
            n_tgt=int(xyxy_tgt.shape[0]),
        )

    return CSPairResult(
        stem=stem,
        source_img=src_img,
        target_img=tgt_img,
        source_txt=src_txt,
        target_txt=tgt_txt,
        result_md=md_path,
        com_tar=com_tar,
        com_source=com_source,
        cs=cs,
        n_src=int(xyxy_src.shape[0]),
        n_tgt=int(xyxy_tgt.shape[0]),
    )


def write_dir_mode_cs_result_md(
    md_path: Path,
    *,
    src_dir: Path,
    tgt_dir: Path,
    iou_thr: float,
    n_pairs_total: int,
    results: list[CSPairResult],
    failures: list[tuple[str, str]],
) -> None:
    """
    ``--dir`` 模式聚合报告：每张成功配对的 CS / com_tar / com_source，以及三者在整个文件夹配对上的算术平均。
    平均值仅对已成功的配对计数（失败的配对不计入内）。
    """
    n_ok = len(results)
    avg_cs = sum(r.cs for r in results) / n_ok if n_ok else float("nan")
    avg_tar = sum(r.com_tar for r in results) / n_ok if n_ok else float("nan")
    avg_src = sum(r.com_source for r in results) / n_ok if n_ok else float("nan")
    rows = "\n".join(
        f"| `{r.stem}` | {r.cs:.6f} | {r.com_tar:.6f} | {r.com_source:.6f} |"
        for r in results
    )
    fail_block = ""
    if failures:
        ferr = "\n".join(f"| `{stem}` | {msg} |" for stem, msg in failures)
        fail_block = (
            "\n## 失败配对\n\n"
            "| stem | 错误信息 |\n"
            "| --- | --- |\n"
            f"{ferr}\n\n"
        )
    body = "".join(
        [
            "# 组件双向匹配 — 文件夹批量结果\n",
            "\n",
            "## 输入\n\n",
            f"- source_dir: `{src_dir}`\n",
            f"- target_dir: `{tgt_dir}`\n",
            f"- IoU 阈值: **{iou_thr:g}**\n",
            f"- 配对总数: {n_pairs_total}（计算成功 **{n_ok}**，失败 **{len(failures)}**）\n",
            "\n",
            "## 指标均值（成功配对）\n\n",
            "对下列「逐对明细」表中每一组 **CS、com_tar、com_source** 分别求算术平均值：\n\n",
            f"| avg_CS | avg_com_tar | avg_com_source |\n",
            "| --- | --- | --- |\n",
            f"| **{avg_cs:.6f}** | **{avg_tar:.6f}** | **{avg_src:.6f}** |\n",
            "\n",
            "## 逐对明细（各组数值）\n\n",
            "| stem | CS | com_tar | com_source |\n",
            "| --- | --- | --- | --- |\n",
            (rows + "\n" if rows else "（无成功条目）\n"),
            fail_block,
        ]
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(body, encoding="utf-8")


def batch_compute_cs_for_directories(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    repo_root: Path,
    cfg: ComponentsPredictConfig,
    model: YOLO,
    iou_thr: float,
    base_out_dir: Path,
) -> Path:
    """
    两文件夹内按 stem 配对（后缀可不同），逐对调用 ``compute_cs_for_pair``（仅写 txt，不写单对 md）。

    输出：
      - ``base_out_dir/cs_dir_<source 目录名>/<stem>/`` 下放 YOLO 标签；
      - 同一批根目录下放唯一 ``cs_result.md``（明细 + avg_CS / avg_com_tar / avg_com_source）。
    """
    def _resolve_input_dir(p: str | Path) -> Path:
        q = Path(p).expanduser()
        return q.resolve() if q.is_absolute() else (repo_root / q).resolve()

    src_dir = _resolve_input_dir(source_dir)
    tgt_dir = _resolve_input_dir(target_dir)
    pairs = collect_same_stem_image_pairs(src_dir, tgt_dir)
    if not pairs:
        raise RuntimeError(f"在两个文件夹中未找到同名图片对：{src_dir} / {tgt_dir}")

    batch_root = base_out_dir / f"cs_dir_{src_dir.name}"
    batch_root.mkdir(parents=True, exist_ok=True)

    ok_results: list[CSPairResult] = []
    failures: list[tuple[str, str]] = []
    for i, (sp, tp) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {sp.name} <-> {tp.name}")
        stem = sp.stem
        pair_out = batch_root / stem
        try:
            r = compute_cs_for_pair(
                sp,
                tp,
                out_dir=pair_out,
                repo_root=repo_root,
                cfg=cfg,
                model=model,
                iou_thr=iou_thr,
                write_markdown=False,
            )
            ok_results.append(r)
            print(f"  CS={r.cs:.6f}  (com_tar={r.com_tar:.4f} com_source={r.com_source:.4f})")
        except Exception as e:  # noqa: BLE001
            failures.append((stem, str(e)))
            print(f"  [FAIL] {e}", file=sys.stderr)

    result_path = batch_root / "cs_result.md"
    write_dir_mode_cs_result_md(
        result_path,
        src_dir=src_dir,
        tgt_dir=tgt_dir,
        iou_thr=iou_thr,
        n_pairs_total=len(pairs),
        results=ok_results,
        failures=failures,
    )
    avg_cs = sum(r.cs for r in ok_results) / len(ok_results) if ok_results else float("nan")
    print(f"\n[DONE] 平均 CS = {avg_cs:.6f}  ->  {result_path}")
    return result_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="两张图组件检测 + 双向匹配比例 CS；支持单图或与 layout 对齐的文件夹批量。")
    p.add_argument(
        "-s",
        "--source",
        default="data/ours/snapshot/gardening_products_website_ui_design_492370171740540158.png",
        type=str,
        help="source 截图路径（单图模式）",
    )
    p.add_argument(
        "-t",
        "--target",
        default="data/images_origin/gardening_products_website_ui_design_492370171740540158.jpg",
        type=str,
        help="target 截图路径（单图模式）",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUT_DIR),
        help=(
            f"输出根目录（默认 {DEFAULT_OUT_DIR}）；"
            "单图写入当前目录 cs_result.md；--dir 写入 cs_dir_<源文件夹名>/cs_result.md"
        ),
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.10,
        help="同类框判定 IoU 阈值（默认 0.10）",
    )
    p.add_argument("--model", default="", help="覆盖 YOLO 权重路径")
    p.add_argument("--conf", type=float, default=None, help="confidence 阈值")
    batch = p.add_argument_group("文件夹批量模式（与 layout_similarity_calculation 对齐）")
    batch.add_argument(
        "--dir",
        action="store_true",
        help="启用文件夹批量模式，需配合 --source-dir / --target-dir",
    )
    batch.add_argument(
        "--source-dir",
        # default="data/ours/snapshot",  # 与 layout_similarity_calculation ours
        # default="data/gemini3-1/snapshot", #gemini3-1
        # default="data/GLM/snapshot", #glm
        # default="data/internVL/snapshot", #internVL
        # default="data/QwenVL/snapshot", #QwenVL
        # default="data/gpt4o/snapshot", #gpt4o
        # default="data/gpt4omini/snapshot", #gpt4omini
        default="data/LLaVA/snapshot", #LLaVA
        help="source 图片文件夹（与 --dir 联用）",
    )
    batch.add_argument(
        "--target-dir",
        default="data/images_origin",
        help="target 图片文件夹（与 --dir 联用）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = REPO_ROOT
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ComponentsPredictConfig()
    if args.model:
        cfg.model_path = args.model
    if args.conf is not None:
        cfg.confidence_threshold = args.conf

    mf = _resolve_model_path(root, cfg.model_path)
    if not mf.is_file():
        print(f"[ERROR] 权重不存在: {mf}", file=sys.stderr)
        sys.exit(2)
    model = YOLO(str(mf))
    iou_thr = float(args.iou)

    if args.dir:
        batch_compute_cs_for_directories(
            args.source_dir,
            args.target_dir,
            repo_root=root,
            cfg=cfg,
            model=model,
            iou_thr=iou_thr,
            base_out_dir=out_dir,
        )
        return

    src_img = Path(args.source).expanduser()
    src_img = src_img.resolve() if src_img.is_absolute() else (root / src_img).resolve()
    tgt_img = Path(args.target).expanduser()
    tgt_img = tgt_img.resolve() if tgt_img.is_absolute() else (root / tgt_img).resolve()
    if not src_img.is_file() or not tgt_img.is_file():
        print("[ERROR] source 或 target 图像不存在。", file=sys.stderr)
        sys.exit(2)

    r = compute_cs_for_pair(
        src_img,
        tgt_img,
        out_dir=out_dir,
        repo_root=root,
        cfg=cfg,
        model=model,
        iou_thr=iou_thr,
        result_md=out_dir / "cs_result.md",
    )
    print(f"[OK] 标签: {r.source_txt}\n     {r.target_txt}")
    print(f"[OK] com_tar={r.com_tar:.6f} com_source={r.com_source:.6f} CS={r.cs:.6f}")
    print(f"[OK] 已写入 {r.result_md}")


if __name__ == "__main__":
    main()
