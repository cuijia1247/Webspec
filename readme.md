# Web UI 视觉相似度工具

本目录用于对 **Web UI 截图/设计稿** 等成对图像计算 **视觉相似度**，便于对比原稿与生成稿、不同版本界面或 A/B 稿的一致性。

当前提供两种基于深度视觉特征的度量方式：

| 方式 | 脚本 |  backbone | 特点 |
|------|------|-----------|------|
| **CLIP** | `vs_clip_score.py` | 默认 `openai/clip-vit-base-patch16` | 图文联合训练，对「语义级」布局与内容较敏感，可按需切换 HF 上其他 CLIP 变体。 |
| **DINOv2** | `vs_codino_score.py` | 默认 `dinov2_vitb14`（Torch Hub） | 自监督视觉表征，对纹理、结构、细粒度外观往往更稳；可选 s/b/l/g 规模。 |

两者均对两张图分别提取 **归一化图像特征**，再计算 **余弦相似度**（输出约在 \([-1, 1]\)，实际多为正且接近 1 表示更相似）。脚本内会对分数给出粗粒度文字档位，仅供参考。

---

## 环境与依赖

建议在虚拟环境中安装：

```bash
pip install torch torchvision pillow
# CLIP 脚本额外需要：
pip install transformers huggingface_hub
```

- **CUDA**：若已安装 GPU 版 PyTorch，脚本会自动优先使用 `cuda`。
- **网络**：首次运行会下载预训练权重；国内可对 CLIP 使用脚本默认的 Hugging Face 镜像，对 DINOv2 可配合 `--proxy`。

---

## 预训练模型目录

两种脚本默认将权重放在项目下的 **`./pretrainedModels/`**（相对于你执行命令时的当前工作目录）：

- **CLIP**：`snapshot_download` 到 `pretrainedModels/<模型子目录>/`（如 `clip-vit-base-patch16`）。若本地已有完整文件则跳过下载。
- **DINOv2**：通过 `torch.hub.set_dir` 指定根目录为 `pretrainedModels`，实际缓存位于 `pretrainedModels/hub/`。

可用 `--cache_dir`（两个脚本均支持）改为其他路径。

---

## 使用方法

### 1. CLIP 视觉相似度

```bash
python vs_clip_score.py --img1 path/to/ui_a.png --img2 path/to/ui_b.png
```

常用参数：

- `--model`：Hugging Face 模型 ID，默认 `openai/clip-vit-base-patch16`；可选如 `openai/clip-vit-base-patch32`、`openai/clip-vit-large-patch14`。
- `--cache_dir`：预训练根目录，默认 `./pretrainedModels`。
- `--proxy`：下载走代理，例如 `http://127.0.0.1:7890`。
- `--hf_endpoint`：HF 端点，默认 `https://hf-mirror.com`。
- `--device`：`cuda` / `cpu`，默认自动。

### 2. DINOv2 视觉相似度（`vs_codino_score.py`）

```bash
python vs_codino_score.py --img1 path/to/ui_a.png --img2 path/to/ui_b.png
```

常用参数：

- `--model`：`dinov2_vits14` | `dinov2_vitb14`（默认）| `dinov2_vitl14` | `dinov2_vitg14`。
- `--image_size`：输入边长，默认 `224`（脚本使用 `Resize`，避免裁切丢失 UI 边缘信息）。
- `--cache_dir`：Torch Hub 根目录，默认 `./pretrainedModels`。
- `--proxy`：同上。
- `--device`：同上。

---

## 结果解读（简要）

- **分数越高**：两图在对应特征空间越接近，一般可理解为视觉越相似。
- **CLIP 与 DINOv2 不可直接横向比绝对值**：二者特征定义不同，更适合在同一指标下做 **相对排序** 或与各自阈值对比。
- **业务结论** 应结合具体阈值与人工抽查标定；脚本中的 `Very high` / `High` 等档位为启发式提示。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `vs_clip_score.py` | 基于 Transformers CLIP 的双图余弦相似度 |
| `vs_codino_score.py` | 基于 Torch Hub DINOv2 的双图余弦相似度 |
| `readme.md` | 本说明 |
