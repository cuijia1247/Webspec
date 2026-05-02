# -*- coding: utf-8 -*-
"""
单对截图或文件夹批量 CLIP 余弦相似度，逻辑复用 utils/vs_clip_score.py。

- **默认**：与 layout_similarity_calculation 相同，一对 ``-s`` / ``-t`` 默认示例图。
- **``--dir``**：与同脚本相同的 ``--source-dir`` / ``--target-dir``，按 stem 配对（后缀可不同）。

用法（在仓库 Webspec 根目录）::

    python visual_similarity_calculation.py
    python visual_similarity_calculation.py -s path/a.png -t path/b.jpg
    python visual_similarity_calculation.py --dir
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


def _find_webspec_root() -> Path:
    """
    定位含有 utils/vs_clip_score.py 的仓库根目录。
    """
    start = Path(__file__).resolve().parent
    for d in (start, *start.parents):
        if (d / "utils" / "vs_clip_score.py").is_file():
            return d
    raise FileNotFoundError(
        f"无法在 {start} 的父目录链中找到 utils/vs_clip_score.py；"
        "请将脚本保留在 Webspec 项目内。"
    )


_REPO_ROOT = _find_webspec_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.vs_clip_score import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_HF_ENDPOINT,
    DEFAULT_MODEL_ID,
    ensure_clip_model,
    extract_clip_features,
    set_hf_endpoint,
    set_proxy,
)

# 配对时参与后缀集合 — 与 layout_similarity_calculation.batch_compute_ls 一致
PAIR_IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})

DEFAULT_VS_OUTPUT_DIR = _REPO_ROOT / "output" / "visual_similarity"

DEFAULT_SOURCE_IMG = (
    "data/ours/snapshot/gardening_products_website_ui_design_492370171740540158.png"
)
DEFAULT_TARGET_IMG = (
    "data/images_origin/gardening_products_website_ui_design_492370171740540158.jpg"
)
# --dir 默认值与 layout_similarity_calculation.parse_args 中一致
DEFAULT_SOURCE_DIR = "data/QwenVL/snapshot"
DEFAULT_TARGET_DIR = "data/images_origin"


def _resolve_under_repo(path: str | Path, repo: Path) -> Path:
    """相对路径按仓库根目录解析。"""
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def _is_pair_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in PAIR_IMAGE_SUFFIXES


def _stem_to_path_map(folder: Path, label: str) -> dict[str, Path]:
    """
    第一层图片按 stem 索引；同名多文件保留字典序更靠前全名，并警告。
    """
    by_stem: dict[str, Path] = {}
    for child in sorted(folder.iterdir()):
        if not _is_pair_image(child):
            continue
        stem_key = child.stem
        if stem_key not in by_stem:
            by_stem[stem_key] = child
            continue
        prev = by_stem[stem_key]
        keep, drop = (child, prev) if child.name < prev.name else (prev, child)
        by_stem[stem_key] = keep
        print(
            f"[WARN] {label} 主文件名重复 {stem_key!r}: 使用 {keep.name}，忽略 {drop.name}",
            file=sys.stderr,
        )
    return by_stem


def collect_name_pairs(dir_a: Path, dir_b: Path) -> list[tuple[str, Path, Path]]:
    """
    两目录第一层中 stem 相同的图片对；不要求后缀一致。
    返回: (stem, path_a, path_b)，按 stem 排序。
    """
    if not dir_a.is_dir():
        raise FileNotFoundError(f"source 文件夹不存在: {dir_a}")
    if not dir_b.is_dir():
        raise FileNotFoundError(f"target 文件夹不存在: {dir_b}")

    map_a = _stem_to_path_map(dir_a, "source_dir")
    map_b = _stem_to_path_map(dir_b, "target_dir")

    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    return [(s, map_a[s].resolve(), map_b[s].resolve()) for s in common]


def clip_similarity_two_images(
    path_a: Path,
    path_b: Path,
    *,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> float:
    """单对图像 CLIP L2 归一化向量余弦相似度。"""
    with torch.no_grad():
        feats = extract_clip_features(
            model=model,
            processor=processor,
            image_paths=[str(path_a), str(path_b)],
            device=device,
        )
    return float(
        F.cosine_similarity(feats[0].unsqueeze(0), feats[1].unsqueeze(0), dim=1).item()
    )


def format_single_pair_markdown(
    *,
    source_stem: str,
    target_stem: str,
    clip_score: float,
    model_id: str = "",
) -> str:
    lines = [
        "# 视觉相似度（CLIP 余弦相似度）",
        "",
        "## 输入",
        "",
        f"- source stem: `{source_stem}`",
        f"- target stem: `{target_stem}`",
        "",
        "## CLIP",
        "",
        f"- 模型: `{model_id or DEFAULT_MODEL_ID}`",
        f"- **CLIP 相似度**: **{clip_score:.6f}**",
        "",
    ]
    return "\n".join(lines)


def format_markdown_batch(
    rows: list[tuple[str, float]],
    dir_a_display: str,
    dir_b_display: str,
    n_ok: int,
    n_skipped: int,
    *,
    model_id: str = "",
) -> str:
    """批量结果：stem + 相似度表 + 统计摘要。"""
    h0 = "主文件名（stem）"
    h1 = "CLIP 相似度"

    w0 = max(len(h0), max((len(r[0]) for r in rows), default=0))
    w1 = max(len(h1), 12)

    def sep_left(n: int) -> str:
        return ":" + "-" * (max(n, 3) - 1)

    def sep_right(n: int) -> str:
        return "-" * (max(n, 3) - 1) + ":"

    header_line = f"| {h0:<{w0}} | {h1:>{w1}} |"
    align_line = f"| {sep_left(w0)} | {sep_right(w1)} |"

    mean_score = statistics.mean([r[1] for r in rows]) if rows else 0.0

    lines = [
        "# 视觉相似度（CLIP 余弦相似度）",
        "",
        f"- source_dir: `{dir_a_display}`",
        f"- target_dir: `{dir_b_display}`",
        f"- CLIP 模型: `{model_id or DEFAULT_MODEL_ID}`",
        "- 配对：stem 一致即可，后缀可不一致（后缀集合与 layout_similarity_calculation 一致）。",
        f"- 成功: **{n_ok}**；跳过/失败: **{n_skipped}**",
        "",
        "## 逐对结果",
        "",
        header_line,
        align_line,
    ]
    for stem, score in rows:
        lines.append(f"| {stem:<{w0}} | {score:>{w1}.6f} |")

    lines.extend(["", "## 统计摘要", ""])
    if rows:
        scores = [r[1] for r in rows]
        lines.append(f"- 配对数量: {len(rows)}")
        lines.append(f"- 平均相似度: {mean_score:.6f}")
        if len(rows) > 1:
            lines.append(f"- 最小: {min(scores):.6f}")
            lines.append(f"- 最大: {max(scores):.6f}")
            lines.append(f"- 标准差: {statistics.stdev(scores):.6f}")
        else:
            lines.append(f"- 最小 / 最大: {scores[0]:.6f}")
    else:
        lines.append("- 无有效配对。")

    lines.append("")
    return "\n".join(lines)


def batch_compute_visual_similarity(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    repo_root: Path,
    model_id: str,
    cache_dir: str,
    proxy: str | None,
    hf_endpoint: str,
    device: str | None,
    out_md: Path | None = None,
) -> Path:
    """
    文件夹批量 CLIP；结果写入 ``output/visual_similarity/<source 目录名>.md`` 或自定义 out_md。
    """
    src_dir = _resolve_under_repo(source_dir, repo_root)
    tgt_dir = _resolve_under_repo(target_dir, repo_root)

    pairs = collect_name_pairs(src_dir, tgt_dir)
    dest_dir = DEFAULT_VS_OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_md if out_md is not None else (dest_dir / f"{src_dir.name}.md")

    if not pairs:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            da = str(src_dir.relative_to(repo_root))
        except ValueError:
            da = str(src_dir)
        try:
            db = str(tgt_dir.relative_to(repo_root))
        except ValueError:
            db = str(tgt_dir)
        md_path.write_text(
            format_markdown_batch([], da, db, 0, 0, model_id=model_id),
            encoding="utf-8",
        )
        print(f"[WARN] 无配对。已写入: {md_path}")
        return md_path

    set_proxy(proxy)
    set_hf_endpoint(hf_endpoint)
    model_local = ensure_clip_model(model_id=model_id, cache_dir=cache_dir)
    dev = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[INFO] Device: {dev}")

    model = CLIPModel.from_pretrained(model_local).to(dev)
    processor = CLIPProcessor.from_pretrained(model_local)
    model.eval()

    rows: list[tuple[str, float]] = []
    n_err = 0
    for stem, path_a, path_b in pairs:
        try:
            sim = clip_similarity_two_images(
                path_a, path_b, model=model, processor=processor, device=dev
            )
            rows.append((stem, sim))
            print(f"[OK] {stem}: {sim:.6f}")
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {stem}: {e}", file=sys.stderr)
            n_err += 1

    try:
        da = str(src_dir.relative_to(repo_root))
    except ValueError:
        da = str(src_dir)
    try:
        db = str(tgt_dir.relative_to(repo_root))
    except ValueError:
        db = str(tgt_dir)

    md = format_markdown_batch(
        rows, da, db, len(rows), n_err, model_id=model_id
    )
    md_path.write_text(md, encoding="utf-8")
    avg = statistics.mean([r[1] for r in rows]) if rows else 0.0
    print(f"\n[INFO] 平均 CLIP = {avg:.6f}（{len(rows)} 对）")
    print(f"[INFO] 报告: {md_path}")
    return md_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CLIP 视觉相似度：默认单图对；--dir 时按文件夹 stem 配对（与 layout_similarity_calculation 参数对齐）。"
    )
    p.add_argument(
        "-s",
        "--source",
        default=DEFAULT_SOURCE_IMG,
        help="source 截图（单图模式）",
    )
    p.add_argument(
        "-t",
        "--target",
        default=DEFAULT_TARGET_IMG,
        help="target 截图（单图模式）",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help=(
            "输出 Markdown。"
            "单图：默认为 output/visual_similarity/<source stem>.md；"
            "--dir：默认为 output/visual_similarity/<source 目录名>.md"
        ),
    )

    batch = p.add_argument_group("文件夹批量（与 layout_similarity_calculation 对齐）")
    batch.add_argument(
        "--dir",
        action="store_true",
        help="启用文件夹模式，需配合 --source-dir / --target-dir",
    )
    batch.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        help="source 图目录（与 --dir 联用）",
    )
    batch.add_argument(
        "--target-dir",
        default=DEFAULT_TARGET_DIR,
        help="target 图目录（与 --dir 联用）",
    )

    p.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face CLIP 模型 ID")
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help="预训练模型缓存根目录")
    p.add_argument("--proxy", type=str, default=None, help="例如 http://127.0.0.1:7890")
    p.add_argument("--hf_endpoint", type=str, default=DEFAULT_HF_ENDPOINT, help="HF 端点或镜像")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu，省略则自动")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dir:
        out_custom: Path | None = None
        if args.output.strip():
            out_custom = Path(args.output).expanduser()
            if not out_custom.is_absolute():
                out_custom = (_REPO_ROOT / out_custom).resolve()
        batch_compute_visual_similarity(
            args.source_dir,
            args.target_dir,
            repo_root=_REPO_ROOT,
            model_id=args.model,
            cache_dir=args.cache_dir,
            proxy=args.proxy,
            hf_endpoint=args.hf_endpoint,
            device=args.device,
            out_md=out_custom,
        )
        return

    src_img = _resolve_under_repo(args.source, _REPO_ROOT)
    tgt_img = _resolve_under_repo(args.target, _REPO_ROOT)
    if not src_img.is_file() or not tgt_img.is_file():
        print("[ERROR] source 或 target 图片不存在。", file=sys.stderr)
        sys.exit(2)

    set_proxy(args.proxy)
    set_hf_endpoint(args.hf_endpoint)
    model_local = ensure_clip_model(model_id=args.model, cache_dir=args.cache_dir)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[INFO] Device: {device}")

    model = CLIPModel.from_pretrained(model_local).to(device)
    processor = CLIPProcessor.from_pretrained(model_local)
    model.eval()

    sim = clip_similarity_two_images(
        src_img, tgt_img, model=model, processor=processor, device=device
    )
    ss, ts = src_img.stem, tgt_img.stem
    md = format_single_pair_markdown(
        source_stem=ss,
        target_stem=ts,
        clip_score=sim,
        model_id=args.model,
    )

    if args.output.strip():
        out_path = Path(args.output).expanduser()
        if not out_path.is_absolute():
            out_path = (_REPO_ROOT / out_path).resolve()
    else:
        DEFAULT_VS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_VS_OUTPUT_DIR / f"{src_img.stem}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[OK] CLIP 相似度: {sim:.6f}")
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
