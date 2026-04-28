# -*- coding: utf-8 -*-
"""
vs_codino_score.py

Compute visual similarity between two UI screenshots using DINOv2.

Usage:
    python vs_codino_score.py --img1 ui_a.png --img2 ui_b.png

With proxy:
    python vs_codino_score.py --img1 ui_a.png --img2 ui_b.png \
        --proxy http://127.0.0.1:7890

With local cache (default: ./pretrainedModels):
    python vs_codino_score.py --img1 ui_a.png --img2 ui_b.png \
        --cache_dir ./pretrainedModels

Optional model:
    python vs_codino_score.py --img1 ui_a.png --img2 ui_b.png \
        --model dinov2_vits14

Recommended models:
    dinov2_vits14  # faster, smaller
    dinov2_vitb14  # balanced, default
    dinov2_vitl14  # stronger, slower
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 与 vs_clip_score 一致：预训练权重默认存放于该目录（Torch Hub 会写入 hub/ 子目录）
DEFAULT_PRETRAINED_ROOT = "./pretrainedModels"


def set_proxy(proxy: str | None):
    """
    Set HTTP/HTTPS proxy for downloading model weights.
    Example:
        proxy = "http://127.0.0.1:7890"
    """
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
        print(f"[INFO] Proxy enabled: {proxy}")


def set_torch_cache(cache_dir: str | None):
    """
    Set torch hub cache directory.
    This is useful when downloading models in China or reusing local weights.
    """
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_path))
        print(f"[INFO] Torch hub cache dir: {cache_path.resolve()}")


def build_transform(image_size: int = 224):
    """
    DINOv2 commonly uses ImageNet-style normalization.
    For UI screenshots, center crop may remove edge information.
    Therefore, this script uses Resize((224, 224)) by default.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_image(image_path: str, transform, device: torch.device):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    return image_tensor


@torch.no_grad()
def extract_feature(model, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    提取 DINOv2 全局特征：显式调用 forward_features，取 x_norm_clstoken
    （最后一层 LayerNorm 后的 [CLS]，与 eval 下 head 为 Identity 时的 model(x) 等价）。
    """
    if not hasattr(model, "forward_features"):
        raise AttributeError(
            "当前模型不支持 forward_features，请确认加载的是 facebookresearch/dinov2 Hub 主干。"
        )

    feats = model.forward_features(image_tensor)
    if not isinstance(feats, dict) or "x_norm_clstoken" not in feats:
        raise ValueError(
            "forward_features 应返回含 x_norm_clstoken 的字典；"
            f"实际: {type(feats).__name__}"
        )

    feature = feats["x_norm_clstoken"]
    return F.normalize(feature, p=2, dim=1)


@torch.no_grad()
def compute_dinov2_similarity(
    img1_path: str,
    img2_path: str,
    model_name: str = "dinov2_vitb14",
    image_size: int = 224,
    device: str | None = None
):
    """
    Compute cosine similarity between two UI images using DINOv2.

    Returns:
        float: cosine similarity score.
    """
    device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading DINOv2 model: {model_name}")

    model = torch.hub.load(
        "facebookresearch/dinov2",
        model_name
    ).to(device)

    model.eval()

    transform = build_transform(image_size=image_size)

    img1 = load_image(img1_path, transform, device)
    img2 = load_image(img2_path, transform, device)

    feat1 = extract_feature(model, img1)
    feat2 = extract_feature(model, img2)

    similarity = torch.sum(feat1 * feat2, dim=1).item()

    return similarity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute visual similarity between two UI screenshots using DINOv2."
    )

    parser.add_argument(
        "--img1",
        type=str,
        required=True,
        help="Path to the first UI image."
    )

    parser.add_argument(
        "--img2",
        type=str,
        required=True,
        help="Path to the second UI image."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="dinov2_vitb14",
        choices=[
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14"
        ],
        help="DINOv2 model name."
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size. Default is 224."
    )

    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Proxy address, e.g., http://127.0.0.1:7890"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_PRETRAINED_ROOT,
        help=(
            "Torch Hub 缓存目录，DINOv2 权重将下载到此路径下。"
            f"默认: {DEFAULT_PRETRAINED_ROOT}"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device, e.g., cuda, cpu. Default: auto."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_proxy(args.proxy)
    # 默认将 Hub 缓存指向 ./pretrainedModels，与 vs_clip_score 中 Hugging Face 模型根目录一致
    set_torch_cache(args.cache_dir or DEFAULT_PRETRAINED_ROOT)

    score = compute_dinov2_similarity(
        img1_path=args.img1,
        img2_path=args.img2,
        model_name=args.model,
        image_size=args.image_size,
        device=args.device
    )

    print("\n========== Visual Similarity Result ==========")
    print(f"Image 1: {args.img1}")
    print(f"Image 2: {args.img2}")
    print(f"Model  : {args.model}")
    print(f"Score  : {score:.6f}")

    if score >= 0.90:
        level = "Very high similarity"
    elif score >= 0.80:
        level = "High similarity"
    elif score >= 0.65:
        level = "Moderate similarity"
    elif score >= 0.50:
        level = "Low-to-moderate similarity"
    else:
        level = "Low similarity"

    print(f"Level  : {level}")
    print("==============================================\n")


if __name__ == "__main__":
    main()