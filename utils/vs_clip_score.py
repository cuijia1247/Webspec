# -*- coding: utf-8 -*-
"""
vs_clip_score.py

Compute visual similarity between two UI screenshots using CLIP ViT-B/16.

Usage:
    python vs_clip_score.py --img1 ui_sketch.png --img2 ui_generated.png

With proxy:
    python vs_clip_score.py --img1 ui_sketch.png --img2 ui_generated.png \
        --proxy http://127.0.0.1:7890

With local cache (default: ./pretrainedModels，与 DINOv2 脚本共用根目录):
    python vs_clip_score.py --img1 ui_sketch.png --img2 ui_generated.png

Optional model:
    python vs_clip_score.py --img1 ui_sketch.png --img2 ui_generated.png \
        --model openai/clip-vit-base-patch16
"""

import os
import glob
import argparse
import sys
import errno
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import snapshot_download

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


# ==========================================
# 1. Default configuration
# ==========================================

# 与 vs_codino_score 一致：所有预训练模型根目录
PRETRAINED_ROOT = "./pretrainedModels"

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

# 官方 Hub；在镜像不可达时可作为备选（需本机能访问 huggingface.co）
HF_HUB_OFFICIAL = "https://huggingface.co"

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch16"

DEFAULT_CACHE_DIR = PRETRAINED_ROOT

MODEL_DIR_NAME_MAP = {
    "openai/clip-vit-base-patch16": "clip-vit-base-patch16",
    "openai/clip-vit-base-patch32": "clip-vit-base-patch32",
    "openai/clip-vit-large-patch14": "clip-vit-large-patch14",
}


# ==========================================
# 2. Environment and path configuration
# ==========================================

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


def set_hf_endpoint(endpoint: str | None):
    """
    Set Hugging Face endpoint.

    For users in China, the default mirror is:
        https://hf-mirror.com

    注意: huggingface_hub 在 import 时已读取一次 HF_ENDPOINT 到 constants.ENDPOINT，
    仅设置环境变量不足以让未传 endpoint 的 API 使用新值；下载时须显式传入 endpoint（见 ensure_clip_model）。
    """
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"[INFO] Hugging Face endpoint: {endpoint}")


def resolve_hf_endpoint() -> str:
    """当前生效的 Hub 根 URL（与 set_hf_endpoint 写入的环境变量一致）。"""
    return (os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT).rstrip("/")


def _is_hub_network_error(exc: BaseException) -> bool:
    """
    判断是否为网络不可达、超时等，便于在镜像与官方端点之间回退。
    会沿 __cause__ / __context__ 遍历异常链。
    """
    chain: BaseException | None = exc
    seen: set[int] = set()

    while chain is not None and id(chain) not in seen:
        seen.add(id(chain))

        if isinstance(chain, OSError):
            en = getattr(chain, "errno", None)
            if en in (errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ECONNREFUSED, 101):
                return True

        if httpx is not None and isinstance(
            chain,
            (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError),
        ):
            return True

        text = str(chain).lower()
        if "network is unreachable" in text or "errno 101" in text:
            return True

        chain = chain.__cause__ or chain.__context__

    return False


def get_local_model_dir(model_id: str, cache_dir: str) -> str:
    """
    Generate local model directory according to model id.
    """
    model_dir_name = MODEL_DIR_NAME_MAP.get(
        model_id,
        model_id.replace("/", "__")
    )

    return os.path.join(cache_dir, model_dir_name)


# ==========================================
# 3. Model preparation
# ==========================================

def clip_model_files_ready(local_dir: str) -> bool:
    """
    Check whether the local directory contains the minimum files required
    to load CLIP with Hugging Face Transformers.

    Required:
        - config.json
        - preprocessor_config.json
        - tokenizer files
        - pytorch_model.bin or *.safetensors
    """
    if not os.path.isdir(local_dir):
        return False

    required_files = [
        "config.json",
        "preprocessor_config.json",
    ]

    for file_name in required_files:
        if not os.path.isfile(os.path.join(local_dir, file_name)):
            return False

    bin_path = os.path.join(local_dir, "pytorch_model.bin")
    safetensors_files = glob.glob(os.path.join(local_dir, "*.safetensors"))

    if not os.path.isfile(bin_path) and len(safetensors_files) == 0:
        return False

    return True


def ensure_clip_model(
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> str:
    """
    If the local CLIP model does not exist, download it from Hugging Face.
    If it already exists, directly return the local path.
    """
    local_dir = get_local_model_dir(model_id, cache_dir)

    os.makedirs(local_dir, exist_ok=True)

    if clip_model_files_ready(local_dir):
        abs_path = os.path.abspath(local_dir)
        print(f"[INFO] Local CLIP model found, skip download: {abs_path}")
        return abs_path

    print(f"[INFO] Local model incomplete or missing.")
    print(f"[INFO] Downloading CLIP model: {model_id}")
    print(f"[INFO] Target directory: {os.path.abspath(local_dir)}")

    ignore_patterns = [
        "*.msgpack",
        "*.h5",
        "*.ot",
        "flax_model.msgpack",
        "tf_model.h5",
    ]

    # 必须显式传入 endpoint：否则仍可能使用 import 时缓存的 constants.ENDPOINT
    primary = resolve_hf_endpoint()
    endpoint_candidates = [primary]
    if primary.rstrip("/") != HF_HUB_OFFICIAL.rstrip("/"):
        endpoint_candidates.append(HF_HUB_OFFICIAL.rstrip("/"))

    last_err: BaseException | None = None
    for idx, ep in enumerate(endpoint_candidates, start=1):
        try:
            print(f"[INFO] Hub 下载端点 ({idx}/{len(endpoint_candidates)}): {ep}")
            model_path = snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                ignore_patterns=ignore_patterns,
                max_workers=4,
                endpoint=ep,
            )
            print(f"[INFO] Model is ready: {model_path}")
            return model_path
        except Exception as e:
            last_err = e
            if not _is_hub_network_error(e):
                raise
            print(
                f"[WARN] 无法连接 Hub ({ep}): {e}\n"
                f"       若需代理请加: --proxy http://127.0.0.1:端口",
                file=sys.stderr,
            )

    hint = (
        "\n[提示] 所有 Hub 端点均连接失败。请检查：\n"
        "  1) 网络/防火墙/IPv6；可尝试: python ... --proxy http://127.0.0.1:7890\n"
        "  2) 或在一台可联网机器下载 openai/clip-vit-base-patch16 后，"
        f"放入目录: {os.path.abspath(local_dir)}\n"
        "  3) 或指定官方源: --hf_endpoint https://huggingface.co（需能访问外网）\n"
    )
    raise RuntimeError(f"CLIP 模型下载失败。{hint}") from last_err


# ==========================================
# 4. Image loading and feature extraction
# ==========================================

def load_image(image_path: str) -> Image.Image:
    """
    Load one image and convert it to RGB.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return Image.open(image_path).convert("RGB")


@torch.no_grad()
def extract_clip_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: list[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Extract normalized CLIP visual features for a list of images.
    """
    images = [load_image(path) for path in image_paths]

    inputs = processor(
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    vision_out = model.get_image_features(**inputs)
    # Transformers 5.x：返回 BaseModelOutputWithPooling，投影后特征在 pooler_output；
    # 旧版本可能直接返回 Tensor。
    if hasattr(vision_out, "pooler_output"):
        image_features = vision_out.pooler_output
    else:
        image_features = vision_out

    image_features = F.normalize(image_features, p=2, dim=-1)

    return image_features


@torch.no_grad()
def compute_clip_similarity(
    img1_path: str,
    img2_path: str,
    model_local_path: str,
    device: str | None = None,
) -> float:
    """
    Compute cosine similarity between two UI images using CLIP.

    Returns:
        float: cosine similarity score.
    """
    device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading CLIP model from: {model_local_path}")

    model = CLIPModel.from_pretrained(model_local_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_local_path)

    model.eval()

    features = extract_clip_features(
        model=model,
        processor=processor,
        image_paths=[img1_path, img2_path],
        device=device,
    )

    feat1 = features[0].unsqueeze(0)
    feat2 = features[1].unsqueeze(0)

    similarity = F.cosine_similarity(feat1, feat2, dim=1).item()

    return similarity


# ==========================================
# 5. Argument parser
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute visual similarity between two UI screenshots using CLIP."
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
        default=DEFAULT_MODEL_ID,
        choices=[
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
        ],
        help="CLIP model id from Hugging Face."
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"预训练模型根目录（默认: {PRETRAINED_ROOT}）。",
    )

    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Proxy address, e.g., http://127.0.0.1:7890"
    )

    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default=DEFAULT_HF_ENDPOINT,
        help="Hugging Face endpoint. Default: https://hf-mirror.com"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device, e.g., cuda, cpu. Default: auto."
    )

    return parser.parse_args()


# ==========================================
# 6. Main entry
# ==========================================

def main():
    args = parse_args()

    set_proxy(args.proxy)
    set_hf_endpoint(args.hf_endpoint)

    try:
        model_local_path = ensure_clip_model(
            model_id=args.model,
            cache_dir=args.cache_dir,
        )

        score = compute_clip_similarity(
            img1_path=args.img1,
            img2_path=args.img2,
            model_local_path=model_local_path,
            device=args.device,
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

    except Exception as e:
        print(f"\n[ERROR] Failed to compute visual similarity: {e}")


if __name__ == "__main__":
    main()