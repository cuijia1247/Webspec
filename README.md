# Web UI 视觉相似度工具（WebSpec）

本仓库用于对 **Web UI 截图/设计稿** 进行 **视觉相似度** 与 **布局结构相似度** 计算，以及从 SCUT 目录按主文件名过滤拷贝等数据整理工作。

**运行前请在项目根目录 `Webspec/` 下打开终端**（所有相对路径 `./pretrainedModels`、`./data`、`./output` 均基于此）。

---

## 目录

1. [环境与依赖](#环境与依赖)
2. [预训练权重目录](#预训练权重目录)
3. [脚本运行说明](#脚本运行说明)
   - [1. 单对图像 — CLIP](#1-单对图像--clip-utilsvs_clip_scorepy)
   - [2. 单对图像 — DINOv2](#2-单对图像--dinov2-utilsvs_codino_scorepy)
   - [3. 批量 CLIP 报表](#3-批量-clip-报表-visual_similarity_calculationpy)
   - [4. SCUT 产出过滤拷贝](#4-scut-产出过滤拷贝-utilsfilterpy)
   - [5. 五分区布局检测](#5-五分区布局检测-utilsfive_dicts_predictpy)
   - [6. UI 组件框检测](#6-ui-组件框检测-utilscomponents_predictpy)
   - [7. Level-2 组件分区检测](#7-level-2-组件分区检测-utilsregion_predictpy)
   - [8. 布局相似度 LS](#8-布局相似度-ls-layout_similarity_calculationpy)
   - [9. 布局类别说明](#9-布局类别说明-utilsui_layout_instructionyaml)
4. [方法简介](#方法简介)
5. [仓库结构](#仓库结构)

---

## 环境与依赖

```bash
# 基础（CLIP / DINOv2 视觉相似度）
pip install torch torchvision pillow
pip install transformers huggingface_hub

# YOLO 检测与布局相似度
pip install ultralytics opencv-python pyyaml
pip install zss          # 布局树编辑距离（layout_similarity_calculation.py）
```

> **Ultralytics 版本注意**：`pretrainedModels/level2/best.pt` 等较新权重依赖新版架构（如 `C3k2`）。
> 请使用 pip 安装的 `ultralytics`，并保持较新版本：`pip install -U ultralytics`。
> `layout_similarity_calculation.py` **不再**将 `pretrainedModels/yolo` 插入 `sys.path`，
> 以避免旧 vendored 版本遮蔽 pip 包；`utils/components_predict.py` 仍优先使用 vendored
> yolo 以兼容旧权重与 `auto_component` pickle。

有 NVIDIA GPU 时安装对应 CUDA 版 PyTorch；脚本自动使用 `cuda`。

---

## 预训练权重目录

所有权重默认存放在 **`./pretrainedModels/`** 下（`.gitignore` 已忽略）：

| 路径 | 用途 |
|------|------|
| `pretrainedModels/<模型子目录>/` | CLIP（Hugging Face `snapshot_download`） |
| `pretrainedModels/hub/` | DINOv2（`torch.hub` 缓存） |
| `pretrainedModels/level1/best.ptt` | Level-1 五分区 YOLO 权重 |
| `pretrainedModels/level2/best.pt` | Level-2 `comp_*` 五类分区 YOLO 权重 |
| `pretrainedModels/yolo/models/best.pt` | UI 组件 YOLO 权重（vendored ultralytics） |

首次运行需联网下载；国内可用 `--hf_endpoint https://hf-mirror.com`（CLIP）或 `--proxy http://127.0.0.1:7890`。

---

## 脚本运行说明

### 1. 单对图像 — CLIP（`utils/vs_clip_score.py`）

计算两张截图的 **CLIP 余弦相似度**（L2 归一化后）。

```bash
python utils/vs_clip_score.py --img1 path/a.png --img2 path/b.png
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--img1` / `--img2` | — | 两张图片路径（必须） |
| `--model` | `openai/clip-vit-base-patch16` | 也支持 `clip-vit-base-patch32`、`clip-vit-large-patch14` |
| `--cache_dir` | `./pretrainedModels` | 模型权重缓存目录 |
| `--hf_endpoint` | `https://hf-mirror.com` | HuggingFace 镜像地址 |
| `--proxy` | — | HTTP 代理，如 `http://127.0.0.1:7890` |
| `--device` | 自动 | `cuda` / `cpu` |

```bash
# 示例：指定代理
python utils/vs_clip_score.py --img1 a.png --img2 b.png --proxy http://127.0.0.1:7890
```

---

### 2. 单对图像 — DINOv2（`utils/vs_codino_score.py`）

计算两张截图的 **DINOv2 余弦相似度**，对纹理与细粒度外观更敏感。

```bash
python utils/vs_codino_score.py --img1 path/a.png --img2 path/b.png
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--img1` / `--img2` | — | 两张图片路径（必须） |
| `--model` | `dinov2_vitb14` | 也支持 `dinov2_vits14`（快）、`dinov2_vitl14`（强） |
| `--cache_dir` | `./pretrainedModels` | Torch Hub 缓存目录 |
| `--proxy` | — | HTTP 代理 |
| `--image_size` | `224` | 输入分辨率 |
| `--device` | 自动 | `cuda` / `cpu` |

```bash
# 示例：使用轻量模型
python utils/vs_codino_score.py --img1 a.png --img2 b.png --model dinov2_vits14
```

---

### 3. 批量 CLIP 报表（`visual_similarity_calculation.py`）

遍历两个目录，对 **主文件名相同（忽略后缀）** 的图片对批量计算 CLIP 相似度，输出 Markdown 报告（含每对分数与统计摘要）。

```bash
python visual_similarity_calculation.py \
  --dir1 data/ours/snapshot \
  --dir2 data/images_origin \
  --output results.md
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dir1` / `--dir_a` | `data/ours/snapshot` | 第一个图片目录 |
| `--dir2` / `--dir_b` | `data/images_origin` | 第二个图片目录 |
| `--output` | `results.md` | 报告输出路径（根目录相对） |
| `--model`、`--cache_dir`、`--proxy`、`--hf_endpoint`、`--device` | — | 与 CLIP 脚本一致 |

> 某主名在单侧目录有多文件时按字典序择优保留其一并打印警告。
> 根目录下生成的 `results.md` / `*_results.md` 已被 `.gitignore` 忽略；
> 需长期保留请复制至 `results/visual_similarity/`。

---

### 4. SCUT 产出过滤拷贝（`utils/filter.py`）

以 `data/images_origin/` 的主名为基准，从 `SCUT_llm` 产出中匹配并复制图片、HTML、Spec 到对应目录，输出文件名保持「origin 主名 + SCUT 后缀」。

```bash
python utils/filter.py                   # 处理全部（图片 + HTML + Spec）
python utils/filter.py --dry-run         # 仅预览，不实际拷贝
python utils/filter.py --skip-html       # 跳过 HTML
python utils/filter.py --skip-spec       # 跳过 Spec
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--origin` | `data/images_origin` | 主名来源目录 |
| `--source` / `--image-source` | `data/ours/SCUT_llm/snapshot` | 图片源目录 |
| `--dest` / `--image-dest` | `data/ours/snapshot` | 图片输出目录 |
| `--html-source` | `data/ours/SCUT_llm/html` | HTML 源目录 |
| `--html-dest` | `data/ours/html` | HTML 输出目录 |
| `--spec-source` | `data/ours/SCUT_llm/spec` | Spec 源目录 |
| `--spec-dest` | `data/ours/spec` | Spec 输出目录 |
| `--dry-run` | — | 仅打印操作，不实际复制 |

---

### 5. 五分区布局检测（`utils/five_dicts_predict.py`）

对截图做 **header / footer / body / leftsider / rightsider** 五类 YOLO 检测，并将框画回图像。
权重：`pretrainedModels/level1/best.ptt`；输出：`output/level1/`。

```bash
# 单张（默认保存带框可视化图）
python utils/five_dicts_predict.py -i path/to/screenshot.png

# 批量（遍历目录，仅第一层；--recursive 递归子目录）
python utils/five_dicts_predict.py --auto --input-dir data/images_origin

# 同时保存 YOLO 标签 txt（class xc yc w h，归一化）
python utils/five_dicts_predict.py -i shot.png --txt

# 只要 txt，不保存图片
python utils/five_dicts_predict.py -i shot.png --no-img --txt
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `-i` / `--image` | 示例图 | 单张截图路径 |
| `-o` / `--output` | `output/level1/<stem>_five_dicts_vis<ext>` | 输出路径 |
| `--model` | `pretrainedModels/level1/best.ptt` | 覆盖权重路径 |
| `--conf` | `0.25` | 置信度阈值 |
| `--img` / `--no-img` | 开启 | 是否保存可视化图片 |
| `--txt` | 关闭 | 是否保存 YOLO 标签 txt |
| `--show` | — | 单张模式弹窗显示 |
| `--auto` | — | 批量模式开关 |
| `--input-dir` | `data/images_origin/` | 批量模式图片目录 |
| `--recursive` | — | 递归子目录 |

---

### 6. UI 组件框检测（`utils/components_predict.py`）

对截图做 **49 类 UI 控件/组件** 细粒度检测（类别见 `pretrainedModels/yolo/models/ui_tag_data.yaml`）。
权重：`pretrainedModels/yolo/models/best.pt`；输出：`output/components/`。

标签绘制优先使用系统本地 Noto CJK / 文泉驿字体；无中文字体时退化为「类id 置信度」。

```bash
# 单张
python utils/components_predict.py -i path/to/screenshot.png

# 批量
python utils/components_predict.py --auto --input-dir data/images_origin

# 同时保存 YOLO txt
python utils/components_predict.py -i shot.png --txt

# 仅 txt，不保存图
python utils/components_predict.py --auto --input-dir data/images_origin --txt --no-img
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `-i` / `--image` | 示例图 | 单张截图路径 |
| `-o` / `--output` | `output/components/<stem>_components_vis<ext>` | 输出路径 |
| `--model` | `pretrainedModels/yolo/models/best.pt` | 覆盖权重路径 |
| `--data-yaml` | `pretrainedModels/yolo/models/ui_tag_data.yaml` | 参考 names yaml |
| `--conf` | `0.25` | 置信度阈值 |
| `--img` / `--no-img` | 开启 | 是否保存可视化图片 |
| `--txt` | 关闭 | 是否保存 YOLO 标签 txt |
| `--show` | — | 单张弹窗显示 |
| `--auto` | — | 批量模式 |
| `--input-dir` | `data/images_origin/` | 批量图片目录 |
| `--recursive` | — | 递归子目录 |

---

### 7. Level-2 组件分区检测（`utils/region_predict.py`）

对截图做 **5 类 `comp_*`** 细粒度分区检测，与 level-1 五分区在语义上对应。
权重：`pretrainedModels/level2/best.pt`；输出：`output/level2/`。

**class_id → 标签映射（以训练 data.yaml 中 `names` 的 0–4 顺序为准）**：

| class_id | label | 说明 |
|----------|-------|------|
| 0 | `comp_body` | 主体/主内容区 |
| 1 | `comp_footer` | 页脚区 |
| 2 | `comp_header` | 页头区 |
| 3 | `comp_leftsider` | 左侧栏 |
| 4 | `comp_rightsider` | 右侧栏 |

```bash
# 单张
python utils/region_predict.py -i path/to/screenshot.png

# 批量
python utils/region_predict.py --auto --input-dir data/images_origin

# 调低置信度（默认 0.1）
python utils/region_predict.py -i shot.png --conf 0.05

# 仅保存 YOLO txt，不保存图
python utils/region_predict.py -i shot.png --no-img --txt
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `-i` / `--image` | 示例图 | 单张截图路径 |
| `-o` / `--output` | `output/level2/<stem>_region_vis<ext>` | 输出路径 |
| `--model` | `pretrainedModels/level2/best.pt` | 覆盖权重路径 |
| `--conf` | `0.1` | 置信度阈值 |
| `--img` / `--no-img` | 开启 | 是否保存可视化图片 |
| `--txt` | 关闭 | 是否保存 YOLO 标签 txt |
| `--show` | — | 单张弹窗显示 |
| `--auto` | — | 批量模式 |
| `--input-dir` | `data/images_origin/` | 批量图片目录 |
| `--recursive` | — | 递归子目录 |

---

### 8. 布局相似度 LS（`layout_similarity_calculation.py`）

结合 **五分区**（`five_dicts_predict`）与 **Level-2 comp_* 五类**（`region_predict`），将检测框归并为两层嵌套布局树（`page → level1 区域 → comp_* 子节点`），计算：

- **overall_ts**：完整树（含 `comp_*` 层）的 Tree Edit Distance（TED，`zss` Zhang-Shasha）。
- **region_ts**：仅保留 level1 五区域节点（剥离 `comp_*` 子节点）后的 TED。
- **LS = overall_ts + region_ts**（数值越小结构越相似，非 0–1 相似度）。

结果文件格式：两棵 overall 树字符串、两棵 region 树字符串，以及 overall_ts / region_ts / LS 数值。

#### 单对图片

```bash
python layout_similarity_calculation.py -s path/source.png -t path/target.jpg
```

结果写入 **`output/layout_similarity/<source 主文件名>.txt`**。

#### 文件夹批量

```bash
python layout_similarity_calculation.py --dir \
  --source-dir data/ours/snapshot \
  --target-dir data/images_origin
```

匹配规则：两目录下 **主文件名相同（忽略后缀）** 的图片配对。
结果汇总写入 **`output/layout_similarity/<source 目录名>.txt`**，文件头部含 **avg_LS**。

| 参数 | 默认 | 说明 |
|------|------|------|
| `-s` / `--source` | 示例图 | source 截图路径（单图模式） |
| `-t` / `--target` | 示例图 | target 截图路径（单图模式） |
| `--conf-fd` | `0.25` | five_dicts 置信度阈值 |
| `--conf-region` | `0.1` | region comp_* 置信度阈值 |
| `--dir` | — | 启用文件夹批量模式 |
| `--source-dir` | `data/ours/snapshot` | source 图片目录（批量模式） |
| `--target-dir` | `data/images_origin` | target 图片目录（批量模式） |

---

### 9. 布局类别说明（`utils/ui_layout_instruction.yaml`）

集中说明 **level_1**（五分区 class_id 0–4）与 **component**（`ui_tag_data.yaml` 中 49 类 slug / 中文名 / id）的对应关系，供指令生成、标签解析与各预测脚本对齐引用。

---

## 方法简介

| 方式 | 脚本 | 骨干 | 特点 |
|------|------|------|------|
| CLIP | `utils/vs_clip_score.py` | ViT-B/16 | 语义/内容与整体布局感知较强 |
| DINOv2 | `utils/vs_codino_score.py` | `dinov2_vitb14` | 纹理与细粒度外观更敏感 |
| 布局树 TED | `layout_similarity_calculation.py` | five_dicts + region_predict + `zss` | 结构差异（overall 含 comp_* 层 + region 层），LS 越小越相似 |

CLIP / DINOv2 输出 **L2 归一化余弦相似度**（0–1，越大越相似）。布局 **LS** 为树编辑距离之和（越小越相似），两类指标不宜混比绝对值。

---

## 仓库结构

```
Webspec/
├── layout_similarity_calculation.py   # 布局相似度 LS（overall_ts + region_ts）
├── visual_similarity_calculation.py   # 批量 CLIP 相似度报表（Markdown 输出）
├── utils/
│   ├── vs_clip_score.py               # 单对 CLIP 相似度
│   ├── vs_codino_score.py             # 单对 DINOv2 相似度
│   ├── filter.py                      # SCUT_llm 产出按主名过滤拷贝
│   ├── five_dicts_predict.py          # Level-1 五分区 YOLO 检测 → output/level1/
│   ├── components_predict.py          # 49 类 UI 组件 YOLO 检测 → output/components/
│   ├── region_predict.py              # Level-2 comp_* 五类分区 YOLO 检测 → output/level2/
│   ├── yolo_label_export.py           # 检测框转 YOLO txt（--txt 选项依赖）
│   ├── register_auto_component_alias.py  # 旧权重 auto_component pickle 别名修复
│   └── ui_layout_instruction.yaml    # level_1 / component 类别 id-label 对照表
├── pretrainedModels/                  # 权重目录（.gitignore 忽略）
├── data/                              # 图片与 HTML 数据（.gitignore 忽略）
├── output/                            # 脚本产物（.gitignore 忽略）
│   ├── level1/                        # 五分区可视化
│   ├── level2/                        # Level-2 comp_* 可视化
│   ├── components/                    # UI 组件可视化
│   └── layout_similarity/             # LS 计算结果 txt
└── results/
    └── visual_similarity/             # 归档的 CLIP 报表（已纳入 Git）
```
