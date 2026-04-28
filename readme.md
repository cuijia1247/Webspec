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

首次运行需联网下载；国内可配合 CLIP 的 `--hf_endpoint`、`--proxy`，以及 DINOv2 的 `--proxy`。仓库的 `.gitignore` 已忽略 `pretrainedModels/` 与 `data/`，不纳入 Git。

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

- `--dir1` / `--dir_a`：文件夹 1；`--dir2` / `--dir_b`：文件夹 2。  
- `--output`：报告 Markdown 路径，默认项目根目录 **`results.md`**。  
- 其余与 CLIP 一致：`--model`、`--cache_dir`、`--proxy`、`--hf_endpoint`、`--device`。

> 脚本内 `--dir1` / `--dir2` 的默认值以 `python visual_similarity_calculation.py --help` 为准；若与你目录不一致，请始终显式传入 `--dir1` / `--dir2`。

### 4. SCUT 产出过滤拷贝（`utils/filter.py`）

以 `data/images_origin/` 下文件 **主名（无后缀）** 为基准，从 `data/ours/SCUT_llm/` 下匹配并复制到 `data/ours/` 对应子目录（图片 → `snapshot/`，HTML → `html/`，spec → `spec/`）。输出文件名为「origin 主名 + SCUT 源文件后缀」。

```bash
python utils/filter.py
python utils/filter.py --dry-run
python utils/filter.py --skip-html
python utils/filter.py --skip-spec
```

可用 `--origin`、`--source`、`--dest`、`--html-source`、`--html-dest`、`--spec-source`、`--spec-dest` 覆盖默认路径（见 `python utils/filter.py --help`）。

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
| `utils/filter.py` | 按主名从 SCUT 目录拷贝至 `data/ours/…` |
| `visual_similarity_calculation.py` | 两目录批量 CLIP + `results.md` |
| `readme.md` | 本说明 |
