# Web UI 视觉相似度工具（WebSpec）

本仓库用于对 **Web UI 截图/设计稿** 等图像做 **视觉相似度** 计算与简单数据整理：双图 CLIP / DINOv2 打分、按目录批量 CLIP 报表、从 SCUT 目录按主文件名过滤拷贝。

**运行前请在项目根目录 `Webspec/` 下打开终端**（保证相对路径 `./pretrainedModels`、`./data` 与脚本一致）。

---

## 环境与依赖

```bash
pip install torch torchvision pillow
pip install transformers huggingface_hub
```

有 NVIDIA GPU 时可安装带 CUDA 的 PyTorch；脚本会自动优先使用 `cuda`。

---

## 预训练权重目录

默认使用项目下的 **`./pretrainedModels/`**（相对于当前工作目录）：

| 用途 | 说明 |
|------|------|
| CLIP | Hugging Face `snapshot_download` 到 `pretrainedModels/<模型子目录>/` |
| DINOv2 | `torch.hub` 缓存，一般在 `pretrainedModels/hub/` 等子路径 |

首次运行需联网下载；国内可配合 CLIP 的 `--hf_endpoint`、`--proxy`，以及 DINOv2 的 `--proxy`。仓库的 `.gitignore` 已忽略 `pretrainedModels/`、`data/` 与 **`output/`**（脚本产物目录，如 `output/level1/` 五分区可视化、`output/components/` 组件检测图），不纳入 Git。

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

详细参数见 `python utils/components_predict.py --help`。

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

二者均为 **L2 归一化后的余弦相似度**；分数区间与「高低」不宜跨模型直接对比绝对值，更适合同一指标内排序或自建阈值。

---

## 仓库结构（与脚本相关）

| 路径 | 说明 |
|------|------|
| `utils/vs_clip_score.py` | 单对 CLIP 相似度 |
| `utils/vs_codino_score.py` | 单对 DINOv2 相似度 |
| `utils/filter.py` | 按主名从 SCUT_llm（snapshot/html/spec）拷贝至 `data/ours/…` |
| `utils/five_dicts_predict.py` | 五分区 YOLO 检测与可视化，默认输出 `output/level1/` |
| `utils/components_predict.py` | UI 组件 YOLO 检测与可视化，默认输出 `output/components/` |
| `utils/register_auto_component_alias.py` | 加载旧权重时 `auto_component` → `ultralytics` 的 pickle 别名（供组件脚本等使用） |
| `visual_similarity_calculation.py` | 两目录批量 CLIP，输出 Markdown |
| `results/visual_similarity/` | 归档的各 baseline 批量 CLIP 报告（Markdown） |
| `README.md` | 本说明 |
