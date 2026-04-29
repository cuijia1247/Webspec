# -*- coding: utf-8 -*-
"""
根据 data/images_origin/ 中的文件主名（不含扩展名）匹配 SCUT 产出：

1. 图片：在 data/ours/SCUT_llm/SCUT_llm_image/ 中查找同主名文件，
   复制到 data/ours/snapshot/。
3. Spec：在 data/ours/SCUT_llm/SCUT_llm_spec/ 中查找同主名文件，
   复制到 data/ours/spec/。

输出文件名为「origin 主名 + SCUT 源文件后缀」，保证内容与后缀一致。

用法（在仓库根目录 Webspec 下执行）:
    python utils/filter.py
    python utils/filter.py --skip-html   # 跳过 HTML，仍处理图片与 spec
    python utils/filter.py --skip-spec   # 跳过 spec
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# 仓库根目录：.../Webspec（本文件位于 utils/filter.py）
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ORIGIN = REPO_ROOT / "data" / "images_origin"
DEFAULT_IMAGE_SOURCE = REPO_ROOT / "data" / "ours" / "SCUT_llm" / "snapshot"
DEFAULT_IMAGE_DEST = REPO_ROOT / "data" / "ours" / "snapshot"
DEFAULT_HTML_SOURCE = REPO_ROOT / "data" / "ours" / "SCUT_llm" / "html"
DEFAULT_HTML_DEST = REPO_ROOT / "data" / "ours" / "html"
DEFAULT_SPEC_SOURCE = REPO_ROOT / "data" / "ours" / "SCUT_llm" / "spec"
DEFAULT_SPEC_DEST = REPO_ROOT / "data" / "ours" / "spec"

# 兼容旧参数名：--source / --dest 仍表示图片链路
DEFAULT_SOURCE = DEFAULT_IMAGE_SOURCE
DEFAULT_DEST = DEFAULT_IMAGE_DEST


def collect_origin_files(origin_dir: Path) -> list[Path]:
    """仅遍历 origin 目录下「第一层」文件，返回路径列表（有序）。"""
    if not origin_dir.is_dir():
        raise FileNotFoundError(f"源目录不存在或不是文件夹: {origin_dir}")

    return [p for p in sorted(origin_dir.iterdir()) if p.is_file()]


def build_source_by_stem(source_dir: Path) -> dict[str, Path]:
    """
    将 SCUT 目录下文件按主文件名（Path.stem）索引；忽略扩展名。
    若同一主名对应多个文件，保留全路径字典序较小的一个，并打印警告。
    """
    by_stem: dict[str, Path] = {}
    for p in sorted(source_dir.iterdir()):
        if not p.is_file():
            continue
        stem = p.stem
        if stem not in by_stem:
            by_stem[stem] = p
            continue
        prev = by_stem[stem]
        keep, drop = (p, prev) if p.name < prev.name else (prev, p)
        by_stem[stem] = keep
        print(
            f"[WARN] SCUT 主文件名重复 '{stem}': 使用 {keep.name}，忽略 {drop.name}",
            file=sys.stderr,
        )
    return by_stem


def copy_matching(
    origin_dir: Path,
    source_dir: Path,
    dest_dir: Path,
    *,
    dry_run: bool = False,
    kind: str = "图片",
) -> tuple[int, int, int]:
    """
    按 origin 文件主名在 source_dir 中匹配并复制到 dest_dir。

    Args:
        kind: 日志前缀（如「图片」「HTML」「spec」）。

    Returns:
        (成功复制数, SCUT 中不存在的数量, origin 中文件总数)
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"{kind} 源目录不存在: {source_dir}")

    origin_files = collect_origin_files(origin_dir)
    by_stem = build_source_by_stem(source_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0

    for op in origin_files:
        stem = op.stem
        src = by_stem.get(stem)
        if src is None or not src.is_file():
            missing += 1
            print(
                f"[SKIP][{kind}] SCUT 中无主文件名匹配: {op.name} (stem={stem!r})",
                file=sys.stderr,
            )
            continue

        # 主名与 origin 一致，后缀与 SCUT 源文件一致（避免内容/后缀不符）
        dst_name = f"{stem}{src.suffix}"
        dst = dest_dir / dst_name

        if dry_run:
            print(f"[DRY-RUN][{kind}] 将复制: {src} -> {dst}  (参考 origin: {op.name})")
            copied += 1
            continue

        shutil.copy2(src, dst)
        copied += 1
        print(f"[OK][{kind}] {op.name} -> {dst_name}")

    return copied, missing, len(origin_files)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "按 images_origin 文件主名（不含后缀）匹配 SCUT_llm 下的图片、HTML 与 spec，"
            "分别复制到 snapshot、ours/html、ours/spec。"
        ),
    )
    p.add_argument(
        "--origin",
        type=Path,
        default=DEFAULT_ORIGIN,
        help=f"参考图片目录（默认: {DEFAULT_ORIGIN}）",
    )
    p.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"SCUT 图片目录（默认: {DEFAULT_SOURCE}）",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"输出目录（默认: {DEFAULT_DEST}）",
    )
    p.add_argument(
        "--html-source",
        type=Path,
        default=DEFAULT_HTML_SOURCE,
        help=f"SCUT HTML 目录（默认: {DEFAULT_HTML_SOURCE}）",
    )
    p.add_argument(
        "--html-dest",
        type=Path,
        default=DEFAULT_HTML_DEST,
        help=f"HTML 输出目录（默认: {DEFAULT_HTML_DEST}）",
    )
    p.add_argument(
        "--skip-html",
        action="store_true",
        help="跳过 HTML 拷贝。",
    )
    p.add_argument(
        "--spec-source",
        type=Path,
        default=DEFAULT_SPEC_SOURCE,
        help=f"SCUT spec 目录（默认: {DEFAULT_SPEC_SOURCE}）",
    )
    p.add_argument(
        "--spec-dest",
        type=Path,
        default=DEFAULT_SPEC_DEST,
        help=f"spec 输出目录（默认: {DEFAULT_SPEC_DEST}）",
    )
    p.add_argument(
        "--skip-spec",
        action="store_true",
        help="跳过 spec 拷贝。",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要复制的路径，不写入文件。",
    )
    return p.parse_args()


def _print_phase_summary(
    *,
    phase: str,
    dry_run: bool,
    total: int,
    copied: int,
    missing: int,
) -> None:
    if dry_run:
        print(
            f"\n[{phase}] 完成(dry-run): origin 共 {total} 个文件, "
            f"可复制 {copied}, SCUT 缺失 {missing}。"
        )
    else:
        print(
            f"\n[{phase}] 完成: origin 共 {total} 个文件, "
            f"已复制 {copied}, SCUT 缺失 {missing}。"
        )


def main() -> None:
    args = parse_args()
    origin = args.origin.resolve()
    image_src = args.source.resolve()
    image_dst = args.dest.resolve()

    try:
        print("========== 图片 -> snapshot ==========")
        img_copied, img_missing, img_total = copy_matching(
            origin,
            image_src,
            image_dst,
            dry_run=args.dry_run,
            kind="图片",
        )
        _print_phase_summary(
            phase="图片",
            dry_run=args.dry_run,
            total=img_total,
            copied=img_copied,
            missing=img_missing,
        )

        if not args.skip_html:
            html_src = args.html_source.resolve()
            html_dst = args.html_dest.resolve()
            print("\n========== HTML -> ours/html ==========")
            h_copied, h_missing, h_total = copy_matching(
                origin,
                html_src,
                html_dst,
                dry_run=args.dry_run,
                kind="HTML",
            )
            _print_phase_summary(
                phase="HTML",
                dry_run=args.dry_run,
                total=h_total,
                copied=h_copied,
                missing=h_missing,
            )

        if not args.skip_spec:
            spec_src = args.spec_source.resolve()
            spec_dst = args.spec_dest.resolve()
            print("\n========== spec -> ours/spec ==========")
            s_copied, s_missing, s_total = copy_matching(
                origin,
                spec_src,
                spec_dst,
                dry_run=args.dry_run,
                kind="spec",
            )
            _print_phase_summary(
                phase="spec",
                dry_run=args.dry_run,
                total=s_total,
                copied=s_copied,
                missing=s_missing,
            )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
