# -*- coding: utf-8 -*-
"""
单对截图或文件夹批量：CLIP 视觉编码器 patch 特征 → Gram 矩阵风格特征 → 三种相关性。

- **默认**：与 visual_similarity_calculation 相同，一对 ``-s`` / ``-t`` 默认示例图。
- **``--dir``**：``--source-dir`` / ``--target-dir``，按 stem 配对（后缀可不同）。

相关性（均在 Gram 上三角展开后的等长向量上计算）：

- ``style_cos``：余弦相似度
- ``style_pearman``：斯皮尔曼相关系数
- ``style_kendell``：肯德尔相关系数（与需求拼写一致）

结果写入 ``output/style_similarity/style_analysis.md``（可用 ``-o`` 覆盖）。

依赖：除 transformers / torch 外，斯皮尔曼与肯德尔需 ``scipy``。

用法（在仓库 Webspec 根目录）::

    python style_similarity_calculation.py
    python style_similarity_calculation.py -s path/a.png -t path/b.jpg
    python style_similarity_calculation.py --dir
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

try:
    from scipy.stats import kendalltau, spearmanr
except ImportError as _e:  # pragma: no cover
    kendalltau = None  # type: ignore[misc, assignment]
    spearmanr = None  # type: ignore[misc, assignment]
    _SCIPY_IMPORT_ERROR = _e
else:
    _SCIPY_IMPORT_ERROR = None


def _find_webspec_root() -> Path:
    """定位含有 utils/vs_clip_score.py 的仓库根目录。"""
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
    load_image,
    set_hf_endpoint,
    set_proxy,
)

PAIR_IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})

DEFAULT_STYLE_OUTPUT_DIR = _REPO_ROOT / "output" / "style_similarity"
DEFAULT_STYLE_MD = DEFAULT_STYLE_OUTPUT_DIR / "style_analysis.md"

DEFAULT_SOURCE_IMG = (
    "data/ours/snapshot/gardening_products_website_ui_design_492370171740540158.png"
)
DEFAULT_TARGET_IMG = (
    "data/images_origin/gardening_products_website_ui_design_492370171740540158.jpg"
)
# DEFAULT_SOURCE_DIR = "data/ours/snapshot" #ours
# DEFAULT_SOURCE_DIR = "data/gemini3-1/snapshot" #gemini3-1
# DEFAULT_SOURCE_DIR = "data/GLM/snapshot" #glm
# DEFAULT_SOURCE_DIR = "data/internVL/snapshot" #internVL
# DEFAULT_SOURCE_DIR = "data/QwenVL/snapshot" #QwenVL
# DEFAULT_SOURCE_DIR = "data/gpt4o/snapshot" #gpt4o
# DEFAULT_SOURCE_DIR = "data/gpt4omini/snapshot" #gpt4omini
DEFAULT_SOURCE_DIR = "data/LLaVA/snapshot" #LLaVA
DEFAULT_TARGET_DIR = "data/images_origin"


def _resolve_under_repo(path: str | Path, repo: Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def _is_pair_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in PAIR_IMAGE_SUFFIXES


def _stem_to_path_map(folder: Path, label: str) -> dict[str, Path]:
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
    if not dir_a.is_dir():
        raise FileNotFoundError(f"source 文件夹不存在: {dir_a}")
    if not dir_b.is_dir():
        raise FileNotFoundError(f"target 文件夹不存在: {dir_b}")

    map_a = _stem_to_path_map(dir_a, "source_dir")
    map_b = _stem_to_path_map(dir_b, "target_dir")

    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    return [(s, map_a[s].resolve(), map_b[s].resolve()) for s in common]


def _ensure_scipy() -> None:
    if spearmanr is None or kendalltau is None:
        raise ImportError(
            "斯皮尔曼与肯德尔相关性需要 scipy。请安装: pip install scipy"
        ) from _SCIPY_IMPORT_ERROR


@torch.no_grad()
def extract_patch_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: list[str],
    device: torch.device,
) -> torch.Tensor:
    """
    CLIP 视觉编码器最后一层 patch 序列（不含 CLS），形状 [B, N, D]。
    """
    images = [load_image(path) for path in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    pixel_values = inputs["pixel_values"]
    vision_out = model.vision_model(pixel_values=pixel_values)
    hidden = vision_out.last_hidden_state
    # 首 token 为 CLS，与 Gatys 式 Gram 的「空间维」一致使用 patch token
    patches = hidden[:, 1:, :].contiguous()
    return patches


def gram_matrix(patches: torch.Tensor) -> torch.Tensor:
    """
    patches: [B, N, D]，返回 Gram [B, D, D]，对 N 做平均归一。
    """
    # [B, D, N] @ [B, N, D] -> [B, D, D]
    n = patches.shape[1]
    g = torch.bmm(patches.transpose(1, 2), patches) / float(max(n, 1))
    return g


def gram_triu_flat(gram: torch.Tensor) -> torch.Tensor:
    """单张图 Gram [D, D] 展平为上三角（含对角）向量。"""
    d = gram.shape[0]
    idx = torch.triu_indices(d, d, device=gram.device)
    return gram[idx[0], idx[1]].contiguous()


def style_correlation_metrics(
    vec_a: torch.Tensor,
    vec_b: torch.Tensor,
) -> tuple[float, float, float]:
    """
    对上三角 Gram 向量计算 style_cos, style_pearman, style_kendell。
    """
    _ensure_scipy()
    va = vec_a.detach().float().cpu().numpy().ravel()
    vb = vec_b.detach().float().cpu().numpy().ravel()
    if va.shape != vb.shape or va.size == 0:
        raise ValueError("风格向量长度不一致或为空。")

    t_a = torch.from_numpy(va).unsqueeze(0)
    t_b = torch.from_numpy(vb).unsqueeze(0)
    cos = float(F.cosine_similarity(t_a, t_b, dim=1).item())

    spr, _ = spearmanr(va, vb)
    if spr is None or (isinstance(spr, float) and np.isnan(spr)):
        spr_f = float("nan")
    else:
        spr_f = float(spr)

    ken, _ = kendalltau(va, vb)
    if ken is None or (isinstance(ken, float) and np.isnan(ken)):
        ken_f = float("nan")
    else:
        ken_f = float(ken)

    return cos, spr_f, ken_f


def style_similarity_two_images(
    path_a: Path,
    path_b: Path,
    *,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> tuple[float, float, float]:
    _ensure_scipy()
    patches = extract_patch_embeddings(
        model=model,
        processor=processor,
        image_paths=[str(path_a), str(path_b)],
        device=device,
    )
    grams = gram_matrix(patches)
    v0 = gram_triu_flat(grams[0])
    v1 = gram_triu_flat(grams[1])
    return style_correlation_metrics(v0, v1)


def format_single_pair_markdown(
    *,
    source_stem: str,
    target_stem: str,
    style_cos: float,
    style_pearman: float,
    style_kendell: float,
    model_id: str = "",
) -> str:
    lines = [
        "# 风格相关性（CLIP Gram 矩阵）",
        "",
        "## 输入",
        "",
        f"- source stem: `{source_stem}`",
        f"- target stem: `{target_stem}`",
        "",
        "## 模型与特征",
        "",
        f"- CLIP 模型: `{model_id or DEFAULT_MODEL_ID}`",
        "- 视觉编码器 patch 特征 → Gram 矩阵（按 patch 数平均）→ 上三角展平后比较。",
        "",
        "## 指标",
        "",
        f"- **style_cos**（余弦相似度）: **{style_cos:.6f}**",
        f"- **style_pearman**（斯皮尔曼）: **{style_pearman:.6f}**",
        f"- **style_kendell**（肯德尔）: **{style_kendell:.6f}**",
        "",
    ]
    return "\n".join(lines)


def format_markdown_batch(
    rows: list[tuple[str, float, float, float]],
    dir_a_display: str,
    dir_b_display: str,
    n_ok: int,
    n_skipped: int,
    *,
    model_id: str = "",
) -> str:
    h0 = "stem"
    h1 = "style_cos"
    h2 = "style_pearman"
    h3 = "style_kendell"

    w0 = max(len(h0), max((len(r[0]) for r in rows), default=0))
    w1 = max(len(h1), 12)
    w2 = max(len(h2), 12)
    w3 = max(len(h3), 12)

    def sep_left(n: int) -> str:
        return ":" + "-" * (max(n, 3) - 1)

    def sep_right(n: int) -> str:
        return "-" * (max(n, 3) - 1) + ":"

    header = f"| {h0:<{w0}} | {h1:>{w1}} | {h2:>{w2}} | {h3:>{w3}} |"
    align = f"| {sep_left(w0)} | {sep_right(w1)} | {sep_right(w2)} | {sep_right(w3)} |"

    lines = [
        "# 风格相关性（CLIP Gram 矩阵，目录批量）",
        "",
        f"- source_dir: `{dir_a_display}`",
        f"- target_dir: `{dir_b_display}`",
        f"- CLIP 模型: `{model_id or DEFAULT_MODEL_ID}`",
        "- 配对：stem 一致、后缀可不同。",
        f"- 成功: **{n_ok}**；跳过/失败: **{n_skipped}**",
        "",
        "## 逐对结果",
        "",
        header,
        align,
    ]
    for stem, c, p, k in rows:
        lines.append(
            f"| {stem:<{w0}} | {c:>{w1}.6f} | {p:>{w2}.6f} | {k:>{w3}.6f} |"
        )

    lines.extend(["", "## 统计（平均值）", ""])
    if rows:
        cs = [r[1] for r in rows]
        ps = [r[2] for r in rows]
        ks = [r[3] for r in rows]
        lines.append(f"- 配对数量: **{len(rows)}**")
        lines.append(f"- **style_cos** 均值: **{statistics.mean(cs):.6f}**")
        lines.append(f"- **style_pearman** 均值: **{statistics.mean(ps):.6f}**")
        lines.append(f"- **style_kendell** 均值: **{statistics.mean(ks):.6f}**")
    else:
        lines.append("- 无有效配对，未计算均值。")

    lines.append("")
    return "\n".join(lines)


def batch_compute_style_similarity(
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
    _ensure_scipy()
    src_dir = _resolve_under_repo(source_dir, repo_root)
    tgt_dir = _resolve_under_repo(target_dir, repo_root)

    pairs = collect_name_pairs(src_dir, tgt_dir)
    DEFAULT_STYLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = out_md if out_md is not None else DEFAULT_STYLE_MD

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

    rows: list[tuple[str, float, float, float]] = []
    n_err = 0
    for stem, path_a, path_b in pairs:
        try:
            c, p, k = style_similarity_two_images(
                path_a, path_b, model=model, processor=processor, device=dev
            )
            rows.append((stem, c, p, k))
            print(f"[OK] {stem}: cos={c:.6f} spearman={p:.6f} kendell={k:.6f}")
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

    md = format_markdown_batch(rows, da, db, len(rows), n_err, model_id=model_id)
    md_path.write_text(md, encoding="utf-8")
    if rows:
        print(
            f"\n[INFO] 均值: cos={statistics.mean(r[1] for r in rows):.6f}, "
            f"pearman={statistics.mean(r[2] for r in rows):.6f}, "
            f"kendell={statistics.mean(r[3] for r in rows):.6f}（{len(rows)} 对）"
        )
    print(f"[INFO] 报告: {md_path}")
    return md_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CLIP patch Gram 风格相关性：默认单图对；--dir 时按 stem 配对。"
        )
    )
    p.add_argument("-s", "--source", default=DEFAULT_SOURCE_IMG, help="source 截图（单图）")
    p.add_argument("-t", "--target", default=DEFAULT_TARGET_IMG, help="target 截图（单图）")
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(DEFAULT_STYLE_MD.relative_to(_REPO_ROOT)),
        help="输出 Markdown，默认 output/style_similarity/style_analysis.md",
    )

    batch = p.add_argument_group("文件夹批量")
    batch.add_argument("--dir", action="store_true", help="按目录 stem 配对")
    batch.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="source 图目录")
    batch.add_argument("--target-dir", default=DEFAULT_TARGET_DIR, help="target 图目录")

    p.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face CLIP 模型 ID")
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help="预训练缓存根目录")
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
        batch_compute_style_similarity(
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

    _ensure_scipy()
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

    c, p, k = style_similarity_two_images(
        src_img, tgt_img, model=model, processor=processor, device=device
    )
    md = format_single_pair_markdown(
        source_stem=src_img.stem,
        target_stem=tgt_img.stem,
        style_cos=c,
        style_pearman=p,
        style_kendell=k,
        model_id=args.model,
    )

    out_path = Path(args.output).expanduser()
    if not out_path.is_absolute():
        out_path = (_REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    print(
        f"[OK] style_cos={c:.6f}, style_pearman={p:.6f}, style_kendell={k:.6f}"
    )
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
