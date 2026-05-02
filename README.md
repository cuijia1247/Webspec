# Web UI 视觉相似度工具（WebSpec）

本仓库用于对 **Web UI 截图/设计稿** 等图像做 **视觉相似度** 计算与简单数据整理：双图 CLIP / DINOv2 打分、按目录批量 CLIP 报表、从 SCUT 目录按主文件名过滤拷贝。

**运行前请在项目根目录 `Webspec/` 下打开终端**（保证相对路径 `./pretrainedModels`、`./data` 与脚本一致）。

---

## 环境与依赖

```bash
pip install torch torchvision pillow
pip install transformers huggingface_hub
```

YOLO 五分区 / 组件 / Level-2 / 布局相似度等还需：

```bash
pip install ultralytics opencv-python pyyaml
pip install zss   # 布局树编辑距离（layout_similarity_calculation.py）
```

**Ultralytics 版本**：`pretrainedModels/level2/best.pt` 等较新权重在反序列化时可能依赖新版模块（如 `C3k2`）。请使用 **当前环境内 pip 安装的** `ultralytics`，并建议保持较新版本（`pip install -U ultralytics`）。**不要**在运行 `layout_similarity_calculation.py` 时让仓库内旧的 `pretrainedModels/yolo` 覆盖 pip 包（该脚本已不再强制插入该路径）；`utils/components_predict.py` 仍会优先使用 vendored yolo 以兼容旧权重与 `auto_component` pickle。

有 NVIDIA GPU 时可安装带 CUDA 的 PyTorch；脚本会自动优先使用 `cuda`。

---

## 预训练权重目录

默认使用项目下的 **`./pretrainedModels/`**（相对于当前工作目录）：

| 用途 | 说明 |
|------|------|
| CLIP | Hugging Face `snapshot_download` 到 `pretrainedModels/<模型子目录>/` |
| DINOv2 | `torch.hub` 缓存，一般在 `pretrainedModels/hub/` 等子路径 |

首次运行需联网下载；国内可配合 CLIP 的 `--hf_endpoint`、`--proxy`，以及 DINOv2 的 `--proxy`。仓库的 `.gitignore` 已忽略 `pretrainedModels/`、`data/` 与 **`output/`**（脚本产物目录，如 `output/level1/` 五分区、`output/level2/` Level-2、`output/components/` 组件检测、`output/layout_similarity/` 布局相似度汇总等），不纳入 Git。

---

## 脚本运行说明

### 1. 单对图像 — CLIP（`utils/vs_clip_score.py`）

```bash
python utils/vs_clip_score.py --img1 path/to/a.png --img2 path/to/b.png
```

常用参数：`--model`（默认 `openai/clip-vit-base-patch16`）、`--cache_dir`、`--proxy`、`--hf_endpoint`（默认 `https://hf-mirror.com`）、`--device`。

```bash
python utils/vs_clip_score.py --img1 a.png --img2 b.png --proxy http://127.0.0.1:7890
```

### 2. 单对图像 — DINOv2（`utils/vs_codino_score.py`）

```bash
python utils/vs_codino_score.py --img1 path/to/a.png --img2 path/to/b.png
```

常用参数：`--model`（默认 `dinov2_vitb14`）、`--cache_dir`、`--proxy`、`--image_size`（默认 224）、`--device`。

### 3. 批量 CLIP 报表（`visual_similarity_calculation.py`，项目根目录）

对两个文件夹中 **主文件名相同（`Path.stem`，不要求扩展名一致）** 的图片逐对计算 CLIP 余弦相似度，写入 Markdown（表列为「主文件名 + 相似度」，文末含统计摘要）。

脚本会通过查找 `utils/vs_clip_score.py` **自动定位仓库根目录**，相对路径 `./data`、`--output` 等均相对于该根目录解析。

```bash
python visual_similarity_calculation.py --help
```

典型用法（请按本机实际目录修改）：

```bash
python visual_similarity_calculation.py \
  --dir1 data/ours/snapshot \
  --dir2 data/images_origin \
  --output results.md
```

- `--dir1` / `--dir_a`：文件夹 1；`--dir2` / `--dir_b`：文件夹 2（默认：`data/images_origin/`）。  
- `--output`：报告 Markdown 路径，默认项目根目录 **`results.md`**。  
- 其余与 CLIP 一致：`--model`、`--cache_dir`、`--proxy`、`--hf_endpoint`、`--device`。

若某一主文件名在单侧目录中出现多个图片文件，会按全名字典序择优保留其一并打印警告（与下方 `filter.py` 行为一致）。

> 脚本内 `--dir1` / `--dir2` 的默认值以 `python visual_similarity_calculation.py --help` 为准（代码中可对不同 baseline snapshot 留有注释占位）；若与你目录不一致，请始终显式传入 `--dir1` / `--dir2`。

### 4. SCUT 产出过滤拷贝（`utils/filter.py`）

以 `data/images_origin/` 下文件 **主名（无后缀）** 为基准，从 `data/ours/SCUT_llm/` 下三路匹配并复制到 `data/ours/` 对应位置：

| 类型 | SCUT 源（默认） | 输出（默认） |
|------|-----------------|--------------|
| 图片 | `data/ours/SCUT_llm/snapshot/` | `data/ours/snapshot/` |
| HTML | `data/ours/SCUT_llm/html/` | `data/ours/html/` |
| Spec | `data/ours/SCUT_llm/spec/` | `data/ours/spec/` |

输出文件名为「origin 主名 + SCUT 源文件后缀」，保证内容与后缀一致。

```bash
python utils/filter.py
python utils/filter.py --dry-run
python utils/filter.py --skip-html
python utils/filter.py --skip-spec
```

可用 `--origin`、`--source`、`--dest`、`--html-source`、`--html-dest`、`--spec-source`、`--spec-dest` 覆盖默认路径（见 `python utils/filter.py --help`）。

### 5. 五分区布局检测（`utils/five_dicts_predict.py`，Ultralytics YOLO）

对网页截图做 **header / footer / body / leftsider / rightsider** 检测，并将框画回图像。默认权重路径为仓库内 `pretrainedModels/level1/best.ptt`（若仅存 `.pt` 会自动尝试）；默认将可视化写入 **`output/level1/`**（该目录已被 `.gitignore` 忽略，勿提交大体积出图）。

建议在 conda 等已安装 `ultralytics`、`opencv-python` 的环境中运行（见脚本内说明）。

单张：

```bash
python utils/five_dicts_predict.py -i path/to/screenshot.png
```

批量（遍历目录内 `png`/`jpg`/`jpeg`/`webp`/`bmp`，仅第一层；加 `--recursive` 则递归子目录）：

```bash
python utils/five_dicts_predict.py --auto --input-dir data/images_origin
```

**可视化与 YOLO 标签 txt**：`--img` / `--no-img` 控制是否保存带框图片（**默认保存图**）；`--txt` 在与图片相同的输出目录下写入与源图 **stem 同名** 的 `.txt`（每行 `class x_center y_center width height`，相对宽高归一化）。批量时 `--img` 与 `--txt` 至少选其一。

```bash
# 默认：只出图（与原先一致）
python utils/five_dicts_predict.py -i shot.png
# 图 + 标签
python utils/five_dicts_predict.py -i shot.png --txt
# 只要 txt，不要图
python utils/five_dicts_predict.py -i shot.png --no-img --txt
```

常用参数：`--model`、`--conf`、`--output`、`--show`（单张模式）。详细见 `python utils/five_dicts_predict.py --help`。

### 6. UI 组件框检测（`utils/components_predict.py`，Ultralytics YOLO）

对网页截图做 **UI 控件/组件** 多类检测（类别与训练数据 YAML 一致，默认参考 `pretrainedModels/yolo/models/ui_tag_data.yaml`）。权重默认 `pretrainedModels/yolo/models/best.pt`。可视化写入 **`output/components/`**（同级于五分区结果的 `output/level1/`，均在 `.gitignore` 的 `output/` 下）。

加载旧版 `best.pt` 时若 pickle 中模块名为 `auto_component.*`，脚本会通过 **`utils/register_auto_component_alias.py`** 在导入后映射到 `ultralytics`，避免 `ModuleNotFoundError`（该文件已纳入 Git；`pretrainedModels/` 整体仍默认不入库）。

**标签绘制**：优先使用系统中的 Noto CJK / 文泉驿等本地字体；若无中文字体，则退化为「类 id + 置信度」，避免 Ultralytics 默认逻辑联网下载字体导致卡住。

建议在已安装 `ultralytics`、`opencv-python`、`pyyaml` 的 conda 环境中运行；需要中文标签时可安装系统字体包（如 `fonts-noto-cjk`）。

单张与批量：

```bash
python utils/components_predict.py -i path/to/screenshot.png
python utils/components_predict.py --auto --input-dir data/images_origin
```

**`--img` / `--no-img`、 `--txt`** 与五分区脚本含义相同（默认保存图；`--txt` 写出 YOLO 格式 `stem.txt`）。

```bash
python utils/components_predict.py -i shot.png --txt
python utils/components_predict.py --auto --input-dir data/images_origin --txt --no-img
```

详细参数见 `python utils/components_predict.py --help`。

### 7. Level-2 组件五分区检测（`utils/region_predict.py`，Ultralytics YOLO）

对网页截图做与 level1 语义对应的 **细粒度五类** 检测，标签为 `comp_body` / `comp_footer` / `comp_header` / `comp_leftsider` / `comp_rightsider`（约定 **class_id 0–4** 与训练 `data.yaml` 中 `names` 一致；详见脚本模块文档与 `utils/ui_layout_instruction.yaml` 中的 level-1 / component 说明可对齐业务）。

默认权重 **`pretrainedModels/level2/best.pt`**，可视化写入 **`output/level2/`**，文件名为 `*_region_vis.<扩展名>`；`--txt` 写出同名 YOLO 标签。

```bash
python utils/region_predict.py -i path/to/screenshot.png
python utils/region_predict.py --auto --input-dir data/images_origin
```

常用参数与五分区脚本类似：`--model`、`--conf`、`--img` / `--no-img`、`--txt`、`--show`（单张）。`python utils/region_predict.py --help`。

### 8. 布局相似度 LS（`layout_similarity_calculation.py`，项目根目录）

结合 **五分区**（`five_dicts_predict`）与 **Level-2 五类区域**（`region_predict`，`comp_body` / `comp_footer` / `comp_header` / `comp_leftsider` / `comp_rightsider`），将两图各自的检测框归并为 **嵌套布局树**（`page` → level1 区域 → `comp_*` 子节点），再计算：

- **overall_ts**：完整树（含 `comp_*` 层）的 Tree Edit Distance（TED，`zss`）。
- **region_ts**：仅保留 level1 五区域节点（剥离子节点）后的 TED。
- **LS** = overall_ts + region_ts（**数值越小越接近**，非 0–1 相似度）。

单对图片：结果写入 **`output/layout_similarity/<source 主文件名>.txt`**（含两棵 overall 树、两棵 region 树及三个分数）。常用参数：`--conf-fd`（五分区置信度）、`--conf-region`（Level-2 置信度）。

```bash
pip install zss ultralytics opencv-python
python layout_similarity_calculation.py -s path/source.png -t path/target.jpg
```

**文件夹批量**：`--dir` 与 **`--source-dir` / `--target-dir`** 联用，按 **主文件名相同（忽略后缀）** 配对；汇总写入 **`output/layout_similarity/<source 文件夹名>.txt`**，文末含 **avg_LS**。

```bash
python layout_similarity_calculation.py --dir \
  --source-dir data/ours/snapshot \
  --target-dir data/images_origin
```

### 9. 布局类别说明（`utils/ui_layout_instruction.yaml`）

集中说明 **level_1**（五分区）与 **component**（`ui_tag_data.yaml` 中 49 类 slug）的 id / label，供指令生成、标签解析与各预测脚本对齐引用。

---

## 归档：视觉相似度 Markdown 报告（`results/visual_similarity/`）

将各模型 / 实验的 **批量 CLIP 报表** 汇总在 `results/visual_similarity/` 下，便于对比与引用（数据仍依赖本地 `data/`，仅报告入仓）。当前包含例如：

| 文件 | 说明（概览） |
|------|----------------|
| `ours_before_results.md` / `ours_after_results.md` | 本方法前后或其他对照 |
| `gimini_results.md`、`glm_results.md`、`internVL_results.md`、`qwenvl_results.md` 等 | 各基线模型 snapshot 相对 `data/images_origin` 的相似度表 |
| `gpt4o_results.md`、`gpt4omini_results.md`、`LLaVA_results.md` | 其他 API / 模型结果 |

根目录下由脚本 **新生成** 的 `results.md`、以及 `*_results.md` 仍由 `.gitignore` 忽略，避免覆盖本地实验输出；需要长期保留时请复制到 `results/visual_similarity/` 或改名后再提交。

---

## 方法简介

| 方式 | 脚本 | 默认骨干 | 特点（简要） |
|------|------|-----------|----------------|
| CLIP | `utils/vs_clip_score.py` | ViT-B/16 | 语义/内容与布局倾向更明显 |
| DINOv2 | `utils/vs_codino_score.py` | `dinov2_vitb14` | 纹理与细粒度外观更敏感 |
| 布局树 TED | `layout_similarity_calculation.py` | five_dicts + region_predict + `zss` | 结构差异（overall 含 comp_* + region），LS 为距离之和 |

CLIP / DINOv2 均为 **L2 归一化后的余弦相似度**；分数区间与「高低」不宜跨模型直接对比绝对值，更适合同一指标内排序或自建阈值。布局 **LS** 为树编辑距离之和，**数值越小表示结构越接近**，与余弦相似度含义不同，请勿混比绝对值。

---

## 仓库结构（与脚本相关）

| 路径 | 说明 |
|------|------|
| `utils/vs_clip_score.py` | 单对 CLIP 相似度 |
| `utils/vs_codino_score.py` | 单对 DINOv2 相似度 |
| `utils/filter.py` | 按主名从 SCUT_llm（snapshot/html/spec）拷贝至 `data/ours/…` |
| `utils/five_dicts_predict.py` | 五分区 YOLO 检测与可视化，默认输出 `output/level1/` |
| `utils/components_predict.py` | UI 组件 YOLO 检测与可视化，默认输出 `output/components/` |
| `utils/region_predict.py` | Level-2 五类 `comp_*` YOLO 检测，默认输出 `output/level2/` |
| `utils/yolo_label_export.py` | 检测框转 YOLO txt 行（供上述脚本 `--txt` 使用） |
| `utils/register_auto_component_alias.py` | 加载旧权重时 `auto_component` → `ultralytics` 的 pickle 别名（供组件脚本等使用） |
| `utils/ui_layout_instruction.yaml` | level_1 / component 类别与脚本对齐说明 |
| `layout_similarity_calculation.py` | 基于 five_dicts + region_predict 的布局树 LS，默认写入 `output/layout_similarity/*.txt` |
| `visual_similarity_calculation.py` | 两目录批量 CLIP，输出 Markdown |
| `results/visual_similarity/` | 归档的各 baseline 批量 CLIP 报告（Markdown） |
| `README.md` | 本说明 |
