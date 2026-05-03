# -*- coding: utf-8 -*-
"""
基于 **内容相关组件** 的 OCR + 翻译 + 双向文本匹配指标（Content Similarity）。

1. 对 source / target 做组件检测，仅保留与内容相关的类别 id：0, 5, 7, 8, 11
   （与 ``ui_tag_data.yaml`` 中按钮、标题文本、顶/侧导航、面包屑等对应）。
2. 对各检测框裁剪区域做 EasyOCR（**默认仅用** ``<models-root>/easyocr`` 本地权重，
   ``download_enabled=False``；需官方下载请加 ``--easyocr-allow-download``）。
   非英文译为英文：**中文默认 Helsinki-NLP** ``opus-mt-zh-en``（本地 Marian），
   缓存于 ``<models-root>/huggingface/hub`` 或可离线目录 ``helsinki-mt/opus-mt-zh-en``；
   其它语种不予在线翻译，保留原文参与比对。
   若翻译最终失败：对 **source** 侧则用与其 **同类 + IoU 匹配** 的 target 框 OCR 原文替换；
   若无对应组件或替换为空，则 **丢弃** 该 source 组件。
   target 侧翻译失败时退回 ``content_similarity_calculation_ocr.ensure_english`` 既有逻辑（保留 OCR 原文）。
3. **con_preserve**：对每个 target 比对串，若在 source 比对串中存在足够相似的条目则记 1；
   con_preserve = 命中数 / target 条目数（target 无数则记 1.0）。
4. **con_relevant**：对每个 source 比对串在 target 中同理；
   con_relevant = 命中数 / source 条目数（source 无数则记 1.0）。
5. **con_s** = con_preserve + con_relevant。

中文翻译仅使用本地 **Helsinki-NLP Marian**；拉丁字母 OCR（如按钮文案）按规则视为英文，**不再调用 Google**。

用法（在仓库 Webspec 根目录）::

    python content_similarity_calculation.py -s path/a.png -t path/b.jpg
    python content_similarity_calculation.py --dir
        # ``--source-dir`` / ``--target-dir`` 默认与 ``component_similarity_calculation.py`` 一致；
        # 批量结果写入 ``<output>/content_dir_<源目录名>/content_result.md``。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import content_similarity_calculation_ocr as ct_ocr

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

# 与内容相关的组件类别（用户需求）
CONTENT_RELATED_CLASS_IDS: frozenset[int] = frozenset({0, 5, 7, 8, 11})

DEFAULT_OUT_DIR = REPO_ROOT / "output" / "content_similarity"
CONTENT_RESULT_MD = "content_result.md"

# 单图默认路径：与 component_similarity_calculation.parse_args 一致
DEFAULT_SOURCE_IMG = (
    "data/ours/snapshot/gardening_products_website_ui_design_492370171740540158.png"
)
DEFAULT_TARGET_IMG = (
    "data/images_origin/gardening_products_website_ui_design_492370171740540158.jpg"
)

# ``--dir`` 默认：与 component_similarity_calculation.py 当前注释一致（LLaVA + origin）
# DEFAULT_SOURCE_DIR = "data/LLaVA/snapshot" #LLaVA
# DEFAULT_SOURCE_DIR = "data/gemini3-1/snapshot" #gemini3-1
# DEFAULT_SOURCE_DIR = "data/GLM/snapshot" #glm
# DEFAULT_SOURCE_DIR = "data/internVL/snapshot" #internVL
# DEFAULT_SOURCE_DIR = "data/QwenVL/snapshot" #QwenVL
# DEFAULT_SOURCE_DIR = "data/gpt4o/snapshot" #gpt4o
DEFAULT_SOURCE_DIR = "data/gpt4omini/snapshot" #gpt4omini
# DEFAULT_SOURCE_DIR = "data/ours/snapshot" #ours

DEFAULT_TARGET_DIR = "data/images_origin"


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
    xyxy_src: np.ndarray,
    fw: int,
    fh: int,
    tw: int,
    th: int,
) -> np.ndarray:
    """将像素框从 (fw,fh) 画布线性映射到 (tw,th)。"""
    if fw <= 0 or fh <= 0:
        raise ValueError("无效源图宽高")
    sx = tw / fw
    sy = th / fh
    out = xyxy_src.astype(np.float64).copy()
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


def best_match_tgt_index_for_src(
    box_src: np.ndarray,
    cls_src: int,
    xyxy_tgt: np.ndarray,
    cls_tgt: np.ndarray,
    w_s: int,
    h_s: int,
    w_t: int,
    h_t: int,
    iou_threshold: float,
) -> int | None:
    """
    将 source 框映射到 target 画布，在 **同类** target 框中取最大 IoU；
    若最大 IoU < 阈值则无对应关系。
    """
    if xyxy_tgt.shape[0] == 0:
        return None
    mapped = _map_bbox_to_other_image(box_src, w_s, h_s, w_t, h_t)
    best_iou = 0.0
    best_j: int | None = None
    for j in range(xyxy_tgt.shape[0]):
        if int(cls_tgt[j]) != int(cls_src):
            continue
        iou = _xyxy_iou(mapped, xyxy_tgt[j])
        if iou > best_iou:
            best_iou = iou
            best_j = j
    if best_j is None or best_iou < iou_threshold:
        return None
    return int(best_j)


def filter_content_classes(
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    allowed: frozenset[int],
) -> tuple[np.ndarray, np.ndarray]:
    """仅保留类别在 allowed 内的检测框。"""
    if boxes_xyxy.size == 0:
        return boxes_xyxy, class_ids
    mask = np.array([int(c) in allowed for c in class_ids], dtype=bool)
    return boxes_xyxy[mask], class_ids[mask]


def ocr_bbox_bgr(reader: Any, image_bgr: np.ndarray, xyxy: np.ndarray) -> str:
    """裁剪轴对齐框区域并 OCR，返回规范化单行风格文本。"""
    x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy.tolist())
    h, w = image_bgr.shape[:2]
    x1 = max(0, min(x1, max(0, w - 1)))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, max(0, h - 1)))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return ""
    crop = image_bgr[y1:y2, x1:x2]
    lines = reader.readtext(crop, detail=0, paragraph=False)
    if not lines:
        return ""
    parts = [str(x).strip() for x in lines if str(x).strip()]
    return ct_ocr._normalize_ws("\n".join(parts))


def source_compare_text_or_discard(
    ocr_raw: str,
    *,
    matched_target_raw_ocr: str | None,
) -> str | None:
    """
    构建 source 侧参与匹配的文本；若应丢弃该组件则返回 None。
    英文化：先判英文（含 ASCII 界面启发式）→ 中文则 Helsinki-NLP；
    失败则用 **target 对应框 OCR**（非空），否则丢弃。
    """
    t = ct_ocr._normalize_ws(ocr_raw.replace("\n", " "))
    if not t:
        return None
    if ct_ocr.is_english_text(t):
        return t
    try:
        return ct_ocr.translate_to_english(t)
    except BaseException:
        sub = ct_ocr._normalize_ws((matched_target_raw_ocr or "").replace("\n", " "))
        return sub if sub else None


def target_compare_text(ocr_raw: str) -> str | None:
    """target 侧：ensure_english（与全图脚本一致）；空串返回 None 以不参与分母。"""
    t = ct_ocr._normalize_ws(ocr_raw.replace("\n", " "))
    if not t:
        return None
    en, _ = ct_ocr.ensure_english(ocr_raw, substitute_with_target_ocr=None)
    en = ct_ocr._normalize_ws(en.replace("\n", " "))
    return en if en else None


def texts_similar(
    a: str,
    b: str,
    *,
    st_model: Any | None,
    threshold: float,
) -> bool:
    """连续相似度得分 ≥ threshold 视为匹配。"""
    score = ct_ocr.english_text_similarity(a, b, st_model)
    return float(score) >= threshold


def con_preserve_score(target_texts: list[str], source_texts: list[str], **kw: Any) -> float:
    """对每个 target，若在 source 中存在相似项则记 1。"""
    if not target_texts:
        return 1.0
    hits = 0
    for t in target_texts:
        if any(texts_similar(t, s, **kw) for s in source_texts):
            hits += 1
    return hits / len(target_texts)


def con_relevant_score(source_texts: list[str], target_texts: list[str], **kw: Any) -> float:
    """对每个 source，若在 target 中存在相似项则记 1。"""
    if not source_texts:
        return 1.0
    hits = 0
    for s in source_texts:
        if any(texts_similar(s, t, **kw) for t in target_texts):
            hits += 1
    return hits / len(source_texts)


@dataclass
class ContentPairMetrics:
    stem: str
    source_img: Path
    target_img: Path
    con_preserve: float
    con_relevant: float
    con_s: float
    n_target_components: int
    n_source_components_kept: int
    target_texts: list[str]
    source_texts: list[str]


def compute_content_metrics_for_pair(
    src_img: Path,
    tgt_img: Path,
    *,
    repo_root: Path,
    cfg: ComponentsPredictConfig,
    model: YOLO,
    reader: Any,
    st_model: Any | None,
    iou_thr: float,
    text_sim_thr: float,
) -> ContentPairMetrics:
    """检测 → 过滤类别 → 区域 OCR → 翻译与兜底 → con_preserve / con_relevant / con_s。"""
    out_src = predict_components(
        src_img, cfg, repo_root=repo_root, model=model, return_vis=False
    )
    out_tgt = predict_components(
        tgt_img, cfg, repo_root=repo_root, model=model, return_vis=False
    )
    img_s = out_src["image_bgr"]
    img_t = out_tgt["image_bgr"]
    h_s, w_s = img_s.shape[:2]
    h_t, w_t = img_t.shape[:2]

    xyxy_s, cls_s = filter_content_classes(
        out_src["boxes_xyxy"], out_src["class_ids"], CONTENT_RELATED_CLASS_IDS
    )
    xyxy_t, cls_t = filter_content_classes(
        out_tgt["boxes_xyxy"], out_tgt["class_ids"], CONTENT_RELATED_CLASS_IDS
    )

    tgt_raws: list[str] = []
    for i in range(xyxy_t.shape[0]):
        tgt_raws.append(ocr_bbox_bgr(reader, img_t, xyxy_t[i]))

    target_texts: list[str] = []
    for raw in tgt_raws:
        ct = target_compare_text(raw)
        if ct is not None:
            target_texts.append(ct)

    source_texts: list[str] = []
    for i in range(xyxy_s.shape[0]):
        raw_s = ocr_bbox_bgr(reader, img_s, xyxy_s[i])
        mj = best_match_tgt_index_for_src(
            xyxy_s[i],
            int(cls_s[i]),
            xyxy_t,
            cls_t,
            w_s,
            h_s,
            w_t,
            h_t,
            iou_thr,
        )
        sub_raw = tgt_raws[mj] if mj is not None else None
        cmp_s = source_compare_text_or_discard(raw_s, matched_target_raw_ocr=sub_raw)
        if cmp_s is not None:
            source_texts.append(cmp_s)

    kw = {"st_model": st_model, "threshold": text_sim_thr}
    cp = con_preserve_score(target_texts, source_texts, **kw)
    cr = con_relevant_score(source_texts, target_texts, **kw)
    cs = cp + cr

    return ContentPairMetrics(
        stem=src_img.stem,
        source_img=src_img,
        target_img=tgt_img,
        con_preserve=cp,
        con_relevant=cr,
        con_s=cs,
        n_target_components=len(target_texts),
        n_source_components_kept=len(source_texts),
        target_texts=target_texts,
        source_texts=source_texts,
    )


def format_single_pair_md(m: ContentPairMetrics, *, header_extra: str) -> str:
    """单次对比写入 content_result.md 的正文。"""
    rows_tgt = "\n".join(f"| {i} | `{t[:200]}{'…' if len(t) > 200 else ''}` |" for i, t in enumerate(m.target_texts))
    rows_src = "\n".join(f"| {i} | `{t[:200]}{'…' if len(t) > 200 else ''}` |" for i, t in enumerate(m.source_texts))
    return "\n".join(
        [
            "# 内容组件 OCR 双向匹配（Content Similarity）",
            "",
            header_extra,
            "",
            "## 输入",
            "",
            f"- source: `{m.source_img}`",
            f"- target: `{m.target_img}`",
            "",
            "## 指标",
            "",
            "| 项 | 值 |",
            "| --- | --- |",
            f"| con_preserve | **{m.con_preserve:.6f}** |",
            f"| con_relevant | **{m.con_relevant:.6f}** |",
            f"| **con_s** | **{m.con_s:.6f}** |",
            f"| target 参与计数（有效 OCR） | {m.n_target_components} |",
            f"| source 参与计数（翻译失败时已匹配 target OCR，否则丢弃） | {m.n_source_components_kept} |",
            "",
            "## target 比对串列表",
            "",
            "| idx | text |",
            "| --- | --- |",
            rows_tgt if rows_tgt else "| — | （无） |",
            "",
            "## source 比对串列表",
            "",
            "| idx | text |",
            "| --- | --- |",
            rows_src if rows_src else "| — | （无） |",
            "",
        ]
    )


def format_batch_content_result_md(
    *,
    src_dir: Path,
    tgt_dir: Path,
    iou_thr: float,
    text_sim_thr: float,
    sim_backend: str,
    results: list[ContentPairMetrics],
    failures: list[tuple[str, str]],
) -> str:
    """文件夹批量：逐对 con_s / con_preserve / con_relevant + 三者算术均值。"""
    n_ok = len(results)
    avg_s = sum(r.con_s for r in results) / n_ok if n_ok else float("nan")
    avg_p = sum(r.con_preserve for r in results) / n_ok if n_ok else float("nan")
    avg_r = sum(r.con_relevant for r in results) / n_ok if n_ok else float("nan")

    rows = "\n".join(
        f"| `{r.stem}` | {r.con_s:.6f} | {r.con_preserve:.6f} | {r.con_relevant:.6f} | "
        f"{r.n_target_components} | {r.n_source_components_kept} |"
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

    return "".join(
        [
            "# 内容组件 OCR 双向匹配 — 文件夹批量\n",
            "\n",
            "## 输入\n\n",
            f"- source_dir: `{src_dir}`\n",
            f"- target_dir: `{tgt_dir}`\n",
            f"- IoU（source↔target 同类匹配）阈值: **{iou_thr:g}**\n",
            f"- 文本相似度阈值: **{text_sim_thr:g}**（{sim_backend}）\n",
            f"- 成功配对数: **{n_ok}**，失败 **{len(failures)}**\n",
            "\n",
            "## 均值（成功配对）\n\n",
            "| avg_con_s | avg_con_preserve | avg_con_relevant |\n",
            "| --- | --- | --- |\n",
            f"| **{avg_s:.6f}** | **{avg_p:.6f}** | **{avg_r:.6f}** |\n",
            "\n",
            "## 逐对明细\n\n",
            "| stem | con_s | con_preserve | con_relevant | n_target | n_source_kept |\n",
            "| --- | --- | --- | --- | --- | --- |\n",
            (rows + "\n" if rows else "（无成功条目）\n"),
            fail_block,
        ]
    )


def collect_same_stem_image_pairs(
    source_dir: str | Path,
    target_dir: str | Path,
) -> list[tuple[Path, Path]]:
    """与 ``component_similarity_calculation.collect_same_stem_image_pairs`` 一致。"""
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


def batch_compute_content_for_directories(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    repo_root: Path,
    cfg: ComponentsPredictConfig,
    model: YOLO,
    reader: Any,
    st_model: Any | None,
    iou_thr: float,
    text_sim_thr: float,
    sim_backend: str,
    base_out_dir: Path,
) -> Path:
    """批量目录模式；汇总写入 ``content_dir_<source 目录名>/content_result.md``。"""

    def _resolve_input_dir(p: str | Path) -> Path:
        q = Path(p).expanduser()
        return q.resolve() if q.is_absolute() else (repo_root / q).resolve()

    src_dir = _resolve_input_dir(source_dir)
    tgt_dir = _resolve_input_dir(target_dir)
    pairs = collect_same_stem_image_pairs(src_dir, tgt_dir)
    if not pairs:
        raise RuntimeError(f"在两个文件夹中未找到同名图片对：{src_dir} / {tgt_dir}")

    batch_root = base_out_dir / f"content_dir_{src_dir.name}"
    batch_root.mkdir(parents=True, exist_ok=True)

    ok_results: list[ContentPairMetrics] = []
    failures: list[tuple[str, str]] = []
    for i, (sp, tp) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {sp.name} <-> {tp.name}")
        stem = sp.stem
        try:
            m = compute_content_metrics_for_pair(
                sp,
                tp,
                repo_root=repo_root,
                cfg=cfg,
                model=model,
                reader=reader,
                st_model=st_model,
                iou_thr=iou_thr,
                text_sim_thr=text_sim_thr,
            )
            ok_results.append(m)
            print(
                f"  con_s={m.con_s:.6f}  "
                f"(con_preserve={m.con_preserve:.4f} con_relevant={m.con_relevant:.4f})"
            )
        except Exception as e:  # noqa: BLE001
            failures.append((stem, str(e)))
            print(f"  [FAIL] {e}", file=sys.stderr)

    result_path = batch_root / CONTENT_RESULT_MD
    result_path.write_text(
        format_batch_content_result_md(
            src_dir=src_dir,
            tgt_dir=tgt_dir,
            iou_thr=iou_thr,
            text_sim_thr=text_sim_thr,
            sim_backend=sim_backend,
            results=ok_results,
            failures=failures,
        ),
        encoding="utf-8",
    )
    n_ok = len(ok_results)
    avg_s = sum(r.con_s for r in ok_results) / n_ok if n_ok else float("nan")
    print(f"\n[DONE] 平均 con_s = {avg_s:.6f}  ->  {result_path}")
    return result_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="内容相关组件 OCR + 翻译 + con_preserve/con_relevant/con_s；支持 --dir 批量。"
    )
    p.add_argument(
        "-s",
        "--source",
        default=DEFAULT_SOURCE_IMG,
        type=str,
        help="source 截图（单图）",
    )
    p.add_argument(
        "-t",
        "--target",
        default=DEFAULT_TARGET_IMG,
        type=str,
        help="target 截图（单图）",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUT_DIR),
        help=(
            f"输出根目录（默认 {DEFAULT_OUT_DIR}）；"
            "单图写入其中 content_result.md；--dir 写入 content_dir_<源文件夹名>/content_result.md"
        ),
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.10,
        help="source↔target 同类组件 IoU 匹配阈值（默认 0.10，与 CS 脚本一致）",
    )
    p.add_argument(
        "--text-sim",
        type=float,
        default=0.58,
        help="english_text_similarity 得分≥该阈值视为双向匹配成功（默认 0.58）",
    )
    p.add_argument("--model", default="", help="覆盖 YOLO 权重路径")
    p.add_argument("--conf", type=float, default=None, help="YOLO confidence 阈值")

    p.add_argument(
        "--models-root",
        default=str(ct_ocr.DEFAULT_PRETRAINED_MODELS_ROOT),
        help=f"EasyOCR / SentenceTransformer 等资源根（默认 {ct_ocr.DEFAULT_PRETRAINED_MODELS_ROOT}）",
    )
    p.add_argument(
        "--st-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer 模型 id 或本地快照路径（与 OCR 脚本一致）",
    )
    p.add_argument(
        "--offline-st",
        action="store_true",
        help="不加载 SentenceTransformer，仅用 difflib 判相似",
    )
    p.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="EasyOCR 是否使用 GPU；默认自动检测 CUDA",
    )
    p.add_argument(
        "--easyocr-allow-download",
        action="store_true",
        help="无本地 EasyOCR 权重时允许官方下载（默认禁止，仅用 models-root/easyocr）",
    )

    batch = p.add_argument_group("文件夹批量（默认与 component_similarity_calculation 对齐）")
    batch.add_argument(
        "--dir",
        action="store_true",
        help="启用文件夹批量，需配合 --source-dir / --target-dir",
    )
    batch.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="source 图片目录")
    batch.add_argument("--target-dir", default=DEFAULT_TARGET_DIR, help="target 图片目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = REPO_ROOT
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models_root = Path(args.models_root).expanduser()
    models_root = models_root.resolve() if models_root.is_absolute() else (root / models_root).resolve()
    models_root.mkdir(parents=True, exist_ok=True)
    ct_ocr._configure_hf_env_under_pretrained(models_root)
    ct_ocr.set_translation_pretrained_root(models_root)
    ct_ocr.warmup_helsinki_zh_en(models_root)

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
    text_sim_thr = float(args.text_sim)

    reader = ct_ocr.load_easyocr_reader(
        args.gpu, models_root, allow_download=args.easyocr_allow_download
    )
    st_model = ct_ocr.load_sentence_model(args.st_model, models_root, offline_st=args.offline_st)
    sim_backend = ct_ocr.similarity_method_label(st_model)

    if args.dir:
        batch_compute_content_for_directories(
            args.source_dir,
            args.target_dir,
            repo_root=root,
            cfg=cfg,
            model=model,
            reader=reader,
            st_model=st_model,
            iou_thr=iou_thr,
            text_sim_thr=text_sim_thr,
            sim_backend=sim_backend,
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

    m = compute_content_metrics_for_pair(
        src_img,
        tgt_img,
        repo_root=root,
        cfg=cfg,
        model=model,
        reader=reader,
        st_model=st_model,
        iou_thr=iou_thr,
        text_sim_thr=text_sim_thr,
    )
    header_extra = (
        f"- **文本相似度后端**: {sim_backend}\n"
        f"- **中→英翻译**: 仅 Helsinki-NLP `{ct_ocr.HELSINKI_ZH_EN_MODEL_ID}`（本地）；不使用 Google\n"
        f"- **IoU 匹配阈值**: {iou_thr:g}\n"
        f"- **文本相似度阈值**: {text_sim_thr:g}"
    )
    md_path = out_dir / CONTENT_RESULT_MD
    md_path.write_text(format_single_pair_md(m, header_extra=header_extra), encoding="utf-8")
    print(
        f"[OK] con_preserve={m.con_preserve:.6f} "
        f"con_relevant={m.con_relevant:.6f} con_s={m.con_s:.6f}"
    )
    print(f"[OK] 已写入 {md_path}")


if __name__ == "__main__":
    main()
