# -*- coding: utf-8 -*-
"""
布局相似度计算（Layout Similarity）。

流程：
  1. 对 source/target 分别调用 five_dicts_predict 和 components_predict，
     得到带 label 与 bounding-box 的检测结果。
  2. 基于 bounding-box 包含关系构建两层嵌套布局树：
       page -> level_1 区域(header/body/footer/leftsider/rightsider) -> 组件
  3. overall_ts：对完整树计算 Tree Edit Distance（TED）。
  4. region_ts：只保留 level_1 区域节点（剥离组件子节点）后计算 TED。
  5. LS = overall_ts + region_ts。

依赖（除 utils/ 内已有库外）：
    pip install zss    # Zhang-Shasha Tree Edit Distance

用法::

    python layout_similarity_calculation.py -s path/source.png -t path/target.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── 路径初始化（与 utils/ 内脚本对齐）─────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
_UTILS_DIR = REPO_ROOT / "utils"
_YOLO_VENDOR_ROOT = REPO_ROOT / "pretrainedModels" / "yolo"

if _YOLO_VENDOR_ROOT.is_dir() and str(_YOLO_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_YOLO_VENDOR_ROOT))
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from register_auto_component_alias import ensure_auto_component_alias  # noqa: E402
ensure_auto_component_alias()

from five_dicts_predict import FiveDictsPredictConfig, predict_five_dicts  # noqa: E402
from components_predict import ComponentsPredictConfig, predict_components  # noqa: E402

# ── Tree Edit Distance（zss：Zhang-Shasha）────────────────────────────────────
try:
    from zss import Node as _ZssNode
    from zss import simple_distance as _zss_distance
    _HAS_ZSS = True
except ImportError:
    _HAS_ZSS = False
    print("[WARN] 未安装 zss，TED 不可用。请执行：pip install zss", file=sys.stderr)

# level_1 区域标签集合（与 ui_layout_instruction.yaml 及 FiveDictsPredictConfig 对齐）
LEVEL1_LABELS: frozenset[str] = frozenset(
    {"body", "footer", "header", "leftsider", "rightsider"}
)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class DetBox:
    """单个检测框：标签名 + xyxy 坐标。"""

    label: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class LayoutNode:
    """布局树节点（label + 可选 bounding-box + 子节点列表）。"""

    label: str
    box: DetBox | None = None
    children: list["LayoutNode"] = field(default_factory=list)

    def add_child(self, child: "LayoutNode") -> None:
        self.children.append(child)

    def to_string(self) -> str:
        """转为方括号嵌套字符串，如 [page [header [button]] [body [card] [card]]]。"""
        if not self.children:
            return f"[{self.label}]"
        inner = " ".join(c.to_string() for c in self.children)
        return f"[{self.label} {inner}]"


# ── 检测与提取 ────────────────────────────────────────────────────────────────

def _result_to_detboxes(out: dict[str, Any]) -> list[DetBox]:
    """将 predict_* 返回的 dict 转为 DetBox 列表。"""
    names: list[str] = out["names"]
    boxes: list[DetBox] = []
    for i, box in enumerate(out["boxes_xyxy"]):
        cid = int(out["class_ids"][i])
        label = names[cid] if 0 <= cid < len(names) else str(cid)
        boxes.append(
            DetBox(label=label, x1=float(box[0]), y1=float(box[1]),
                   x2=float(box[2]), y2=float(box[3]))
        )
    return boxes


def extract_layout(
    image_path: str | Path,
    *,
    five_dicts_config: FiveDictsPredictConfig | None = None,
    components_config: ComponentsPredictConfig | None = None,
) -> tuple[list[DetBox], list[DetBox]]:
    """
    对单张图片运行两个检测模型。

    Returns:
        (five_dicts_boxes, components_boxes) — DetBox 列表，各含 label 与 xyxy。
    """
    fd_out = predict_five_dicts(image_path, five_dicts_config, return_vis=False)
    comp_out = predict_components(image_path, components_config, return_vis=False)
    return _result_to_detboxes(fd_out), _result_to_detboxes(comp_out)


# ── 包含关系分配 ──────────────────────────────────────────────────────────────

def _intersection_area(a: DetBox, b: DetBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _assign_to_regions(
    regions: list[DetBox],
    components: list[DetBox],
    min_overlap_ratio: float = 0.5,
) -> dict[int, list[int]]:
    """
    将每个组件分配给重叠比例最大的 level_1 区域。
    重叠比例 = intersection / component_area；低于 min_overlap_ratio 不分配（归入 -1）。

    Returns:
        {region_idx: [component_idx, ...], -1: [unassigned_idx, ...]}
    """
    assignment: dict[int, list[int]] = {i: [] for i in range(len(regions))}
    assignment[-1] = []

    for ci, comp in enumerate(components):
        best_ri, best_ratio = -1, min_overlap_ratio
        for ri, reg in enumerate(regions):
            ratio = _intersection_area(comp, reg) / comp.area if comp.area > 0 else 0.0
            if ratio > best_ratio:
                best_ratio, best_ri = ratio, ri
        assignment[best_ri].append(ci)

    return assignment


# ── 构建布局树 ────────────────────────────────────────────────────────────────

def build_layout_tree(
    five_dicts_boxes: list[DetBox],
    components_boxes: list[DetBox],
) -> LayoutNode:
    """
    构建两层嵌套布局树：
      page -> level_1 区域（按 y1 升序） -> 组件（按 y1 升序）
    未被任何区域包含（重叠率 < 0.5）的组件直接挂在 page 下。
    """
    root = LayoutNode(label="page")

    # level_1 区域按 y1（从上到下）排序
    sorted_ri = sorted(range(len(five_dicts_boxes)), key=lambda i: five_dicts_boxes[i].y1)
    sorted_regions = [five_dicts_boxes[i] for i in sorted_ri]
    region_nodes = [LayoutNode(label=r.label, box=r) for r in sorted_regions]

    assignment = _assign_to_regions(sorted_regions, components_boxes)

    for ri, rnode in enumerate(region_nodes):
        for ci in sorted(assignment.get(ri, []), key=lambda i: components_boxes[i].y1):
            comp = components_boxes[ci]
            rnode.add_child(LayoutNode(label=comp.label, box=comp))
        root.add_child(rnode)

    # 未归属组件直接挂 root
    for ci in sorted(assignment.get(-1, []), key=lambda i: components_boxes[i].y1):
        comp = components_boxes[ci]
        root.add_child(LayoutNode(label=comp.label, box=comp))

    return root


# ── Tree Edit Distance ────────────────────────────────────────────────────────

def _to_zss(node: LayoutNode) -> "_ZssNode":
    """递归将 LayoutNode 转为 zss.Node。"""
    z = _ZssNode(node.label)
    for child in node.children:
        z.addkid(_to_zss(child))
    return z


def compute_ted(tree1: LayoutNode, tree2: LayoutNode) -> float:
    """用 Zhang-Shasha 算法计算两棵布局树的 Tree Edit Distance。"""
    if not _HAS_ZSS:
        raise ImportError("缺少 zss 库，请执行：pip install zss")
    return float(_zss_distance(_to_zss(tree1), _to_zss(tree2)))


def _region_only_tree(full_tree: LayoutNode) -> LayoutNode:
    """
    从完整树中提取仅保留 level_1 区域节点（剥离所有组件子节点）的子树。
    只保留 label 属于 LEVEL1_LABELS 的直接子节点。
    """
    root = LayoutNode(label=full_tree.label)
    for child in full_tree.children:
        if child.label in LEVEL1_LABELS:
            root.add_child(LayoutNode(label=child.label, box=child.box))
    return root


# ── 主计算函数 ────────────────────────────────────────────────────────────────

def compute_layout_similarity(
    source_path: str | Path,
    target_path: str | Path,
    *,
    five_dicts_config: FiveDictsPredictConfig | None = None,
    components_config: ComponentsPredictConfig | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    计算 source 与 target 截图的布局相似度。

    Returns:
        dict 包含:
          - source_tree / target_tree:          完整布局树（LayoutNode）
          - source_region_tree / target_region_tree: 仅 level_1 的区域树
          - overall_ts:  完整树 TED
          - region_ts:   仅 level_1 区域树 TED
          - LS:          overall_ts + region_ts
    """
    # Step 1: 提取检测框
    src_fd, src_comp = extract_layout(
        source_path, five_dicts_config=five_dicts_config,
        components_config=components_config,
    )
    tgt_fd, tgt_comp = extract_layout(
        target_path, five_dicts_config=five_dicts_config,
        components_config=components_config,
    )

    # Step 2: 构建布局树
    src_tree = build_layout_tree(src_fd, src_comp)
    tgt_tree = build_layout_tree(tgt_fd, tgt_comp)

    if verbose:
        print(f"[SOURCE] {src_tree.to_string()}")
        print(f"[TARGET] {tgt_tree.to_string()}")

    # Step 3: overall_ts — 完整树 TED
    overall_ts = compute_ted(src_tree, tgt_tree)

    # Step 4: region_ts — 仅保留 level_1 节点后的 TED
    src_region = _region_only_tree(src_tree)
    tgt_region = _region_only_tree(tgt_tree)

    if verbose:
        print(f"[SOURCE region] {src_region.to_string()}")
        print(f"[TARGET region] {tgt_region.to_string()}")

    region_ts = compute_ted(src_region, tgt_region)

    # Step 5: LS
    ls = overall_ts + region_ts

    if verbose:
        print(f"\noverall_ts = {overall_ts:.4f}")
        print(f"region_ts  = {region_ts:.4f}")
        print(f"LS         = {ls:.4f}")

    return {
        "source_tree": src_tree,
        "target_tree": tgt_tree,
        "source_region_tree": src_region,
        "target_region_tree": tgt_region,
        "overall_ts": overall_ts,
        "region_ts": region_ts,
        "LS": ls,
    }


DEFAULT_LS_OUTPUT_DIR = REPO_ROOT / "output" / "layout_similarity"


def save_result(
    result: dict[str, Any],
    source_path: str | Path,
    out_path: str | Path | None = None,
) -> Path:
    """
    将计算结果写入文本文件。
    默认输出路径：output/layout_similarity/<source_stem>.txt。

    文件内容：
      source / target 的 overall_ts 树与 region_ts 树（方括号嵌套字符串），
      以及 overall_ts、region_ts、LS 数值。
    """
    src = Path(source_path)
    dest = Path(out_path) if out_path else DEFAULT_LS_OUTPUT_DIR / f"{src.stem}.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=== Overall Tree (source) ===",
        result["source_tree"].to_string(),
        "",
        "=== Overall Tree (target) ===",
        result["target_tree"].to_string(),
        "",
        "=== Region Tree (source) ===",
        result["source_region_tree"].to_string(),
        "",
        "=== Region Tree (target) ===",
        result["target_region_tree"].to_string(),
        "",
        "=== Scores ===",
        f"overall_ts = {result['overall_ts']:.4f}",
        f"region_ts  = {result['region_ts']:.4f}",
        f"LS         = {result['LS']:.4f}",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


# ── CLI ───────────────────────────────────────────────────────────────────────

def batch_compute_ls(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    five_dicts_config: FiveDictsPredictConfig | None = None,
    components_config: ComponentsPredictConfig | None = None,
    out_dir: Path | None = None,
) -> Path:
    """
    遍历 source_dir，找出与 target_dir 中文件名（stem）相同的图片对，
    逐对计算 LS，结果汇总写入单个文本文件。

    输出文件路径：output/layout_similarity/<source_dir_name>.txt（或 out_dir/<source_dir_name>.txt）。
    文件末尾附上所有对的 LS 均值。

    Returns:
        汇总文件路径。
    """
    src_dir = Path(source_dir).expanduser().resolve()
    tgt_dir = Path(target_dir).expanduser().resolve()

    # 构建 target stem -> Path 的映射（只取第一个匹配，忽略后缀）
    tgt_map: dict[str, Path] = {}
    for p in sorted(tgt_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            tgt_map.setdefault(p.stem, p)

    # 收集 source 中能匹配到 target 的图片对
    pairs: list[tuple[Path, Path]] = []
    for p in sorted(src_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            if p.stem in tgt_map:
                pairs.append((p, tgt_map[p.stem]))

    if not pairs:
        raise RuntimeError(f"在两个文件夹中未找到同名图片对：{src_dir} / {tgt_dir}")

    dest_dir = out_dir or DEFAULT_LS_OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_file = dest_dir / f"{src_dir.name}.txt"

    ls_values: list[float] = []
    record_blocks: list[str] = []

    for i, (src_path, tgt_path) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {src_path.name} <-> {tgt_path.name}")
        try:
            result = compute_layout_similarity(
                src_path, tgt_path,
                five_dicts_config=five_dicts_config,
                components_config=components_config,
                verbose=False,
            )
            ls_values.append(result["LS"])
            block = "\n".join([
                f"### {src_path.stem}",
                f"source : {src_path}",
                f"target : {tgt_path}",
                "",
                "-- Overall Tree (source) --",
                result["source_tree"].to_string(),
                "-- Overall Tree (target) --",
                result["target_tree"].to_string(),
                "",
                "-- Region Tree (source) --",
                result["source_region_tree"].to_string(),
                "-- Region Tree (target) --",
                result["target_region_tree"].to_string(),
                "",
                f"overall_ts = {result['overall_ts']:.4f}",
                f"region_ts  = {result['region_ts']:.4f}",
                f"LS         = {result['LS']:.4f}",
            ])
            print(f"  LS = {result['LS']:.4f}")
        except Exception as e:  # noqa: BLE001
            block = f"### {src_path.stem}\n[ERROR] {e}"
            print(f"  [FAIL] {e}", file=sys.stderr)
        record_blocks.append(block)

    avg_ls = sum(ls_values) / len(ls_values) if ls_values else float("nan")
    summary = "\n".join([
        "=" * 60,
        f"source_dir : {src_dir}",
        f"target_dir : {tgt_dir}",
        f"pairs      : {len(pairs)}  (computed: {len(ls_values)}, failed: {len(pairs) - len(ls_values)})",
        f"avg_LS     = {avg_ls:.4f}",
        "=" * 60,
    ])

    out_file.write_text(
        summary + "\n\n" + ("\n" + "-" * 60 + "\n").join(record_blocks),
        encoding="utf-8",
    )
    print(f"\n[DONE] avg_LS = {avg_ls:.4f}  ->  {out_file}")
    return out_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="计算截图布局相似度 LS。")
    p.add_argument("-s", "--source",
                   default="data/ours/snapshot/gardening_products_website_ui_design_492370171740540158.png",
                   help="source 图片路径（单图模式）")
    p.add_argument("-t", "--target",
                   default="data/images_origin/gardening_products_website_ui_design_492370171740540158.jpg",
                   help="target 图片路径（单图模式）")
    p.add_argument("--conf-fd", type=float, default=None,
                   help="five_dicts 置信度阈值（覆盖默认 0.25）")
    p.add_argument("--conf-comp", type=float, default=None,
                   help="components 置信度阈值（覆盖默认 0.25）")
    batch = p.add_argument_group("文件夹批量模式")
    batch.add_argument("--dir", action="store_true",
                       help="启用文件夹批量模式，需配合 --source-dir / --target-dir")
    # batch.add_argument("--source-dir", default="data/ours/snapshot", #ours
    # batch.add_argument("--source-dir", default="data/gemini3-1/snapshot", #gemini3-1
    batch.add_argument("--source-dir", default="data/GLM/snapshot", #glm
                       help="source 图片文件夹（与 --dir 联用）")
    batch.add_argument("--target-dir", default="data/images_origin",
                       help="target 图片文件夹（与 --dir 联用）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    fd_cfg = FiveDictsPredictConfig()
    if args.conf_fd is not None:
        fd_cfg.confidence_threshold = args.conf_fd

    comp_cfg = ComponentsPredictConfig()
    if args.conf_comp is not None:
        comp_cfg.confidence_threshold = args.conf_comp

    if args.dir:
        batch_compute_ls(
            args.source_dir, args.target_dir,
            five_dicts_config=fd_cfg,
            components_config=comp_cfg,
        )
        return

    result = compute_layout_similarity(
        args.source, args.target,
        five_dicts_config=fd_cfg,
        components_config=comp_cfg,
        verbose=True,
    )
    out_path = save_result(result, args.source)
    print(f"\n[RESULT] LS = {result['LS']:.4f}")
    print(f"[SAVED]  {out_path}")


if __name__ == "__main__":
    main()
