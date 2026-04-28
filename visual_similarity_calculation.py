# -*- coding: utf-8 -*-
"""
批量计算两个文件夹中「主文件名相同、忽略后缀」的图片对的 CLIP 余弦相似度，
复用 utils/vs_clip_score.py 中的模型与特征逻辑。

默认目录：
    文件夹 1: data/ours/snapshot/
    文件夹 2: data/images_origin/

用法（在仓库根目录 Webspec 下执行）:
    python visual_similarity_calculation.py
    python visual_similarity_calculation.py \\
        --dir1 data/ours/snapshot --dir2 data/images_origin \\
        --output results.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

def _find_webspec_root() -> Path:
    """
    定位含有 utils/vs_clip_score.py 的仓库根目录。
    脚本无论放在 Webspec/ 还是 Webspec/data/ 下，默认路径均相对该根目录。
    """
    start = Path(__file__).resolve().parent
    for d in (start, *start.parents):
        if (d / "utils" / "vs_clip_score.py").is_file():
            return d
    raise FileNotFoundError(
        f"无法在 {start} 的父目录链中找到 utils/vs_clip_score.py，"
        "请将脚本保留在 Webspec 项目内。"
    )


_REPO_ROOT = _find_webspec_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import CLIPModel, CLIPProcessor

from utils.vs_clip_score import (
    DEFAULT_CACHE_DIR,
    DEFAULT_HF_ENDPOINT,
    DEFAULT_MODEL_ID,
    ensure_clip_model,
    extract_clip_features,
    set_hf_endpoint,
    set_proxy,
)

# 仅当后缀在这些集合中时才视为图片参与配对
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _stem_to_path_map(folder: Path, label: str) -> dict[str, Path]:
    """
    将目录下第一层图片文件按 Path.stem 索引；同一主名多文件时保留字典序更靠前全名，
    并打印警告（与 utils/filter.py 一致）。
    """
    by_stem: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if not _is_image_file(p):
            continue
        stem = p.stem
        if stem not in by_stem:
            by_stem[stem] = p
            continue
        prev = by_stem[stem]
        keep, drop = (p, prev) if p.name < prev.name else (prev, p)
        by_stem[stem] = keep
        print(
            f"[WARN] {label} 主文件名重复 {stem!r}: 使用 {keep.name}，忽略 {drop.name}",
            file=sys.stderr,
        )
    return by_stem


def collect_name_pairs(dir_a: Path, dir_b: Path) -> list[tuple[str, Path, Path]]:
    """
    在两目录第一层图片中，按「主文件名 Path.stem 相同」配对（不要求后缀一致）。
    返回列表元素: (主文件名 stem, A 侧绝对路径, B 侧绝对路径)，按 stem 排序。
    """
    if not dir_a.is_dir():
        raise FileNotFoundError(f"文件夹 1 不存在: {dir_a}")
    if not dir_b.is_dir():
        raise FileNotFoundError(f"文件夹 2 不存在: {dir_b}")

    map_a = _stem_to_path_map(dir_a, "文件夹1")
    map_b = _stem_to_path_map(dir_b, "文件夹2")

    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    return [(stem, map_a[stem].resolve(), map_b[stem].resolve()) for stem in common]


def format_markdown_table(
    rows: list[tuple[str, float]],
    dir_a_display: str,
    dir_b_display: str,
    n_ok: int,
    n_skipped: int,
) -> str:
    """生成 results.md：逐对仅「主文件名 + 相似度」两列表格，文末统计摘要。"""
    h0 = "主文件名（无后缀）"
    h1 = "CLIP 相似度"

    w0 = max(len(h0), max((len(r[0]) for r in rows), default=0))
    w1 = max(len(h1), 10)

    def sep_left(n: int) -> str:
        n = max(n, 3)
        return ":" + "-" * (n - 1)

    def sep_right(n: int) -> str:
        n = max(n, 3)
        return "-" * (n - 1) + ":"

    header_line = f"| {h0:<{w0}} | {h1:>{w1}} |"
    align_line = f"| {sep_left(w0)} | {sep_right(w1)} |"

    mean_score = statistics.mean([r[1] for r in rows]) if rows else 0.0

    lines = [
        "# 视觉相似度（CLIP 余弦相似度）",
        "",
        f"- 文件夹 1: `{dir_a_display}`",
        f"- 文件夹 2: `{dir_b_display}`",
        "- 配对规则：主文件名（`Path.stem`）相同即可，**不要求扩展名一致**。",
        f"- 成功计算: **{n_ok}** 对；跳过/失败: **{n_skipped}**",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="对两目录下主文件名相同（忽略后缀）的图片批量计算 CLIP 相似度并写入报告。"
    )
    p.add_argument(
        "--dir1",
        "--dir_a",
        type=Path,
        dest="dir1",
        # default=_REPO_ROOT / "data" / "ours" / "snapshot",#ours
        default=_REPO_ROOT / "data" / "gemini3-1" / "snapshot",
        help="图文件夹 1（默认: data/ours/snapshot/）",
    )
    p.add_argument(
        "--dir2",
        "--dir_b",
        type=Path,
        dest="dir2",
        default=_REPO_ROOT / "data" / "images_origin",
        help="图文件夹 2（默认: data/images_origin/）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "results.md",
        help="输出 Markdown 路径（默认: 项目根目录 results.md）",
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face CLIP 模型 ID")
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help="预训练模型根目录")
    p.add_argument("--proxy", type=str, default=None, help="如 http://127.0.0.1:7890")
    p.add_argument("--hf_endpoint", type=str, default=DEFAULT_HF_ENDPOINT, help="HF 端点 / 镜像")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu，默认可自动")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dir1 = args.dir1.resolve()
    dir2 = args.dir2.resolve()
    out_path = args.output.resolve()

    set_proxy(args.proxy)
    set_hf_endpoint(args.hf_endpoint)

    pairs = collect_name_pairs(dir1, dir2)
    if not pairs:
        print(f"[WARN] 两目录下没有「主文件名（无后缀）」一致且均为图片的配对。")
        print(f"  文件夹1: {dir1}")
        print(f"  文件夹2: {dir2}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "# 视觉相似度（CLIP）\n\n无可用配对。\n",
            encoding="utf-8",
        )
        print(f"[INFO] 已写入: {out_path}")
        return

    model_local = ensure_clip_model(model_id=args.model, cache_dir=args.cache_dir)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading CLIP once: {model_local}")

    model = CLIPModel.from_pretrained(model_local).to(device)
    processor = CLIPProcessor.from_pretrained(model_local)
    model.eval()

    rows: list[tuple[str, float]] = []
    n_err = 0

    for stem, path_a, path_b in pairs:
        try:
            with torch.no_grad():
                feats = extract_clip_features(
                    model=model,
                    processor=processor,
                    image_paths=[str(path_a), str(path_b)],
                    device=device,
                )
            sim = F.cosine_similarity(
                feats[0].unsqueeze(0), feats[1].unsqueeze(0), dim=1
            ).item()
        except Exception as e:
            print(f"[SKIP] {stem}: {e}")
            n_err += 1
            continue

        rows.append((stem, sim))
        print(f"[OK] {stem}: {sim:.6f}")

    n_ok = len(rows)
    mean_score = statistics.mean([r[1] for r in rows]) if rows else 0.0

    try:
        da = str(dir1.relative_to(_REPO_ROOT))
    except ValueError:
        da = str(dir1)
    try:
        db = str(dir2.relative_to(_REPO_ROOT))
    except ValueError:
        db = str(dir2)

    md = format_markdown_table(
        rows,
        da,
        db,
        n_ok,
        n_err,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"\n[INFO] 平均 CLIP 相似度: {mean_score:.6f}（{n_ok} 对）")
    print(f"[INFO] 报告: {out_path}")


if __name__ == "__main__":
    main()
