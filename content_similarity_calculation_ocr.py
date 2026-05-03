# -*- coding: utf-8 -*-
"""
【暂时作废】请勿直接 ``python content_similarity_calculation_ocr.py``。
请改用仓库根目录 **`content_similarity_calculation.py`**（内容相关组件 OCR + Helsinki 中→英 + 双向文本指标）。

本模块内的函数仍被 ``content_similarity_calculation.py`` **作为库导入**（EasyOCR、Marian、SentenceTransformer、配对工具等），勿删除文件。

----

截图 **文本内容** 相似度（历史说明）：OCR → 判断是否英文 → 非英文则译为英文 → 对两段英文做向量余弦相似度。

- **单图对**：`-s` / `-t`（默认路径与 ``layout_similarity_calculation.py`` 一致）。
- **`--dir`**：``--source-dir`` / ``--target-dir``，按 stem 配对（后缀可不同），与同脚本 layout 批量规则一致。
- **输出**：写入 **`content_result.md`**（默认 ``output/content_similarity``，可用 ``-o``）：表列为 **主文件名（stem，无后缀）** 与 **相似度**，文末 **平均相似度**。

依赖（按需安装）::

    pip install easyocr langdetect sentence-transformers pillow transformers sentencepiece torch

- **本地模型**：所需权重均在 ``pretrainedModels/``（EasyOCR 默认 **仅本地** ``easyocr/``、不联网下载；
  需要首次下载请加 ``--easyocr-allow-download``；另有 ``huggingface/hub`` ST / Marian 等）。
  若尚无完整快照，会 **优先联网下载到该目录**；下载或命中本地快照后，
  **运行期对 Hugging Face 使用离线环境变量**（避免重复拉 Hub）。无网且无本地 ST 可加 ``--offline-st``（仅用 difflib）。

- **翻译**：**仅中文→英文**使用 Helsinki-NLP ``opus-mt-zh-en``（本地 Marian；
  可置于 ``<models-root>/helsinki-mt/opus-mt-zh-en`` 或 HF hub 缓存）。
  **不使用 Google 在线翻译**：拉丁字母界面 OCR（如 ``BUY NOW``）通过规则视为英文；
  其它非中文语种保留原文参与比对。
  中文 Marian 失败时，仍可按脚本约定使用配对 target OCR 等兜底。

用法::

    python content_similarity_calculation.py
    python content_similarity_calculation.py --dir
    python content_similarity_calculation.py --offline-st
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path
from typing import Any

# Helsinki-NLP Marian：中文→英文（默认路径，权重缓存至 ``<models-root>/huggingface/hub``）
HELSINKI_ZH_EN_MODEL_ID = "Helsinki-NLP/opus-mt-zh-en"
HELSINKI_ZH_EN_CHUNK_CHARS = 180
# 离线整包放置目录（与 Hugging Face 仓库根目录文件一致即可）
HELSINKI_ZH_EN_LOCAL_RELATIVE = Path("helsinki-mt") / "opus-mt-zh-en"

_TRANSLATION_PRETRAINED_ROOT: Path | None = None
_helsinki_zh_en_bundle: tuple[Any, Any, str] | None = None

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PRETRAINED_MODELS_ROOT = REPO_ROOT / "pretrainedModels"
DEFAULT_CONTENT_OUT = REPO_ROOT / "output" / "content_similarity"
CONTENT_RESULT_MD = "content_result.md"

PAIR_IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})

DEFAULT_SOURCE_IMG = (
    "data/ours/snapshot/gardening_products_website_ui_design_492370171740540158.png"
)
DEFAULT_TARGET_IMG = (
    "data/images_origin/gardening_products_website_ui_design_492370171740540158.jpg"
)
# 与 layout_similarity_calculation.parse_args 中 --source-dir/--target-dir 默认一致（勿随意改单侧）
# DEFAULT_SOURCE_DIR = "data/QwenVL/snapshot" #QwenVL
# DEFAULT_SOURCE_DIR = "data/ours/snapshot" #ours
DEFAULT_SOURCE_DIR = "data/gemini3-1/snapshot" #gemini3-1
# DEFAULT_SOURCE_DIR = "data/GLM/snapshot" #glm
# DEFAULT_SOURCE_DIR = "data/internVL/snapshot" #internVL
# DEFAULT_SOURCE_DIR = "data/LLaVA/snapshot" #LLaVA
# DEFAULT_SOURCE_DIR = "data/gpt4o/snapshot" #gpt4o
# DEFAULT_SOURCE_DIR = "data/gpt4omini/snapshot" #gpt4omini
# DEFAULT_SOURCE_DIR = "data/LLaVA/snapshot" #LLaVA
DEFAULT_TARGET_DIR = "data/images_origin"


def _resolve_under_repo(path: str | Path, repo: Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def collect_same_stem_image_pairs(src_dir: Path, tgt_dir: Path) -> list[tuple[Path, Path]]:
    """第一层目录 stem 配对，后缀限定与 layout 批量一致。"""
    if not src_dir.is_dir():
        raise FileNotFoundError(f"source 目录不存在: {src_dir}")
    if not tgt_dir.is_dir():
        raise FileNotFoundError(f"target 目录不存在: {tgt_dir}")

    tgt_map: dict[str, Path] = {}
    for q in sorted(tgt_dir.iterdir()):
        if q.is_file() and q.suffix.lower() in PAIR_IMAGE_SUFFIXES:
            tgt_map.setdefault(q.stem, q)

    pairs: list[tuple[Path, Path]] = []
    for p in sorted(src_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in PAIR_IMAGE_SUFFIXES:
            if p.stem in tgt_map:
                pairs.append((p.resolve(), tgt_map[p.stem].resolve()))
    return pairs


def stem_label_for_pair(src_img: Path, tgt_img: Path) -> str:
    """
    报告中仅使用主文件名（无后缀）：两侧 stem 相同只写一处，不同则 ``a / b``。
    """
    if src_img.stem == tgt_img.stem:
        return src_img.stem
    return f"{src_img.stem} / {tgt_img.stem}"


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# 中日韩谚文：若不含这些字符且大多为 ASCII，则把 OCR 当作英文界面文案（避免 langdetect 误判去走翻译）
_NON_CJK_UI_SCRIPTS_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def looks_like_ascii_ui_text(text: str) -> bool:
    """无中日韩字符且 ≥85% 码点为 ASCII → 视为英文 UI OCR。"""
    t = text.strip()
    if not t:
        return True
    if _NON_CJK_UI_SCRIPTS_RE.search(t):
        return False
    n = len(t)
    ascii_chars = sum(1 for c in t if ord(c) < 128)
    return ascii_chars >= max(1, int(0.85 * n))


def is_english_text(text: str) -> bool:
    """
    粗判是否为英文：过短视为英文；ASCII 界面文案视为英文；
    否则 langdetect 主语言为 en。
    """
    t = text.strip()
    if len(t) < 4:
        return True
    if looks_like_ascii_ui_text(t):
        return True
    try:
        from langdetect import detect  # type: ignore[import-untyped]

        return detect(t) == "en"
    except Exception:  # noqa: BLE001
        return True


def ocr_image(image_path: Path, reader: Any) -> str:
    """EasyOCR：返回整图文本（多行用换行拼接）。"""
    lines = reader.readtext(str(image_path), detail=0, paragraph=False)
    if not lines:
        return ""
    parts = [str(x).strip() for x in lines if str(x).strip()]
    return _normalize_ws("\n".join(parts))


def set_translation_pretrained_root(path: Path | None) -> None:
    """
    设置 Marian / HF 缓存根目录（一般为 ``--models-root``）。
    切换目录时会清空已载入的 Helsinki 模型句柄。
    """
    global _TRANSLATION_PRETRAINED_ROOT, _helsinki_zh_en_bundle
    _TRANSLATION_PRETRAINED_ROOT = path.resolve() if path is not None else None
    _helsinki_zh_en_bundle = None


def _translation_models_root() -> Path:
    return _TRANSLATION_PRETRAINED_ROOT or DEFAULT_PRETRAINED_MODELS_ROOT


_JP_KANA_RE = re.compile(r"[\u3040-\u30ff]")


def should_translate_zh_via_helsinki(text: str) -> bool:
    """
    是否使用 Helsinki ``opus-mt-zh-en``：**默认中文→英文**。
    - ``langdetect`` 主语种为 zh* → 是；
    - 检测失败且含中日韩汉字但 **无日文假名** → 视为中文界面 OCR，走 Marian；
    - 含假名 → 不走 zh-en Marian（避免误喂日文）。
    """
    if _JP_KANA_RE.search(text):
        return False
    try:
        from langdetect import detect  # type: ignore[import-untyped]

        code = detect(text)
        if str(code).startswith("zh"):
            return True
        if code == "en":
            return False
    except Exception:  # noqa: BLE001
        pass
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _marian_zh_en_dir_complete(model_dir: Path) -> bool:
    """目录是否为可用的 Marian zh-en 快照（不含 Hub 元数据亦可）。"""
    if not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").is_file():
        return False
    if not (
        (model_dir / "pytorch_model.bin").is_file() or (model_dir / "model.safetensors").is_file()
    ):
        return False
    return (model_dir / "tokenizer_config.json").is_file() or (model_dir / "tokenizer.json").is_file()


def iter_helsinki_zh_en_local_candidates(pretrained_root: Path):
    """先手动目录，再 HF hub 缓存 snapshots。"""
    yield (pretrained_root / HELSINKI_ZH_EN_LOCAL_RELATIVE).resolve()
    snap_root = (
        pretrained_root / "huggingface" / "hub" / "models--Helsinki-NLP--opus-mt-zh-en" / "snapshots"
    )
    if snap_root.is_dir():
        for child in sorted(snap_root.iterdir(), key=lambda p: p.name):
            if child.is_dir():
                yield child.resolve()


def resolve_helsinki_zh_en_local_or_none(pretrained_root: Path) -> Path | None:
    """返回可用的本地 Marian 目录；若无则 None。"""
    for cand in iter_helsinki_zh_en_local_candidates(pretrained_root):
        if _marian_zh_en_dir_complete(cand):
            return cand
    return None


def repair_sentence_transformer_snapshot_layout(model_dir: Path) -> None:
    """
    修复不完整或过旧的 SentenceTransformer 导出目录，避免载入时出现::
        Pooling.__init__() missing 1 required positional argument: 'embedding_dimension'

    典型问题：仅有 ``Pooling/`` 而 ``modules.json`` 指向 ``1_Pooling``；
    ``Pooling/config.json`` 仅有 ``word_embedding_dimension``；
    缺少 ``2_Normalize``。
    """
    import json

    root = model_dir.resolve()
    if not root.is_dir() or not (root / "modules.json").is_file():
        return
    try:
        modules: Any = json.loads((root / "modules.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return
    if not isinstance(modules, list):
        return
    paths_needed: list[str] = []
    for m in modules:
        if isinstance(m, dict):
            p = m.get("path")
            if isinstance(p, str) and p.strip():
                paths_needed.append(p)

    legacy_pool = root / "Pooling"
    pool = root / "1_Pooling"
    if "1_Pooling" in paths_needed and legacy_pool.is_dir() and not pool.is_dir():
        legacy_pool.rename(pool)
        print(
            "[INFO] SentenceTransformer 快照修复：已将 Pooling/ 重命名为 1_Pooling/",
            file=sys.stderr,
        )

    pc = pool / "config.json"
    if pc.is_file():
        try:
            cfg = json.loads(pc.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cfg = {}
        if isinstance(cfg, dict) and "embedding_dimension" not in cfg:
            wed = cfg.get("word_embedding_dimension")
            if isinstance(wed, int):
                cfg["embedding_dimension"] = wed
                pc.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(
                    "[INFO] SentenceTransformer 快照修复：1_Pooling/config.json 已补充 embedding_dimension",
                    file=sys.stderr,
                )

    norm = root / "2_Normalize"
    if "2_Normalize" in paths_needed and not norm.is_dir():
        norm.mkdir(parents=True, exist_ok=True)
        (norm / "config.json").write_text("{}\n", encoding="utf-8")
        print(
            "[INFO] SentenceTransformer 快照修复：已创建 2_Normalize/config.json",
            file=sys.stderr,
        )


def _get_helsinki_zh_en_bundle(pretrained_root: Path) -> tuple[Any, Any, str]:
    """懒加载 MarianTokenizer + MarianMTModel + device。"""
    global _helsinki_zh_en_bundle
    if _helsinki_zh_en_bundle is not None:
        return _helsinki_zh_en_bundle
    try:
        import torch  # type: ignore[import-untyped]
        from transformers import MarianMTModel, MarianTokenizer  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "本地 Helsinki-NLP 中→英需要 transformers + torch：pip install transformers sentencepiece torch"
        ) from e

    hub = pretrained_root / "huggingface" / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    local_snap = resolve_helsinki_zh_en_local_or_none(pretrained_root)
    if local_snap is not None:
        src = str(local_snap)
        load_kw: dict[str, Any] = {"local_files_only": True}
        print(
            f"[INFO] Helsinki-NLP 中→英从本机载入 `{src}`",
            file=sys.stderr,
        )
    else:
        src = HELSINKI_ZH_EN_MODEL_ID
        load_kw = {"cache_dir": str(hub)}
        print(
            "[INFO] Helsinki-NLP：未检测到本地模型目录 "
            f"`{pretrained_root / HELSINKI_ZH_EN_LOCAL_RELATIVE}` 或 hub snapshots；"
            "将访问 Hugging Face（无网请将仓库文件放入前者，或设置 HF_ENDPOINT=https://hf-mirror.com）…",
            file=sys.stderr,
        )

    tok = MarianTokenizer.from_pretrained(src, **load_kw)
    model = MarianMTModel.from_pretrained(src, **load_kw)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    _helsinki_zh_en_bundle = (tok, model, device)
    print(
        f"[INFO] Helsinki-NLP 中→英已就绪 `{src}`，device={device}",
        file=sys.stderr,
    )
    return _helsinki_zh_en_bundle


def warmup_helsinki_zh_en(pretrained_root: Path | None = None) -> None:
    """
    尽早载入 Helsinki Marian（若尚未缓存则在仍有 Hub 访问能力时下载）。
    应在 ``load_sentence_model`` 设置离线环境 **之前** 调用。
    """
    root = pretrained_root if pretrained_root is not None else _translation_models_root()
    try:
        _get_helsinki_zh_en_bundle(root)
    except BaseException as e:  # noqa: BLE001 — 缺依赖或无网时首次翻译再报错
        print(
            "[INFO] Helsinki-NLP 预载入跳过（可将 opus-mt-zh-en 放入 "
            f"`{root / HELSINKI_ZH_EN_LOCAL_RELATIVE}`，或设置 HF_ENDPOINT 镜像）。"
            f"详情：{type(e).__name__}: {e}",
            file=sys.stderr,
        )


def translate_zh_en_helsinki(text: str, *, pretrained_root: Path | None = None) -> str:
    """使用 Helsinki-NLP opus-mt-zh-en 将中文规范化为英文单行风格文本。"""
    root = pretrained_root if pretrained_root is not None else _translation_models_root()
    tok, model, device = _get_helsinki_zh_en_bundle(root)
    import torch  # type: ignore[import-untyped]

    t = text.strip()
    if not t:
        return ""
    step = HELSINKI_ZH_EN_CHUNK_CHARS
    pieces: list[str] = []
    for i in range(0, len(t), step):
        chunk = t[i : i + step].strip()
        if not chunk:
            continue
        inputs = tok(chunk, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        pieces.append(tok.batch_decode(gen, skip_special_tokens=True)[0])
    return _normalize_ws(" ".join(pieces))


def translate_to_english(text: str) -> str:
    """
    **仅中文**译为英文（Helsinki Marian）；其它语种不再联网翻译，返回规范化原文。
    （拉丁字母 OCR 应先由 ``is_english_text`` / ``looks_like_ascii_ui_text`` 判为英文。）
    """
    t = _normalize_ws(text.replace("\n", " "))
    if not t:
        return ""
    if not should_translate_zh_via_helsinki(t):
        return t
    root = _translation_models_root()
    try:
        return translate_zh_en_helsinki(t, pretrained_root=root)
    except BaseException as e_h:  # noqa: BLE001
        print(
            "[INFO] Helsinki-NLP 中→英失败，保留原文："
            f"{type(e_h).__name__}: {e_h}",
            file=sys.stderr,
        )
        return t


def ensure_english(
    text: str,
    *,
    substitute_with_target_ocr: str | None = None,
) -> tuple[str, bool]:
    """
    若非英文：仅 **中文** 走 Helsinki-NLP；**不使用 Google**。
    其它语种保留原文。

    **substitute_with_target_ocr**：当中文 Marian 失败且该串非空时，
    用其规范化文本参与相似度（典型为配对 **target** 图 OCR）；否则退回本侧原文。

    返回 (比对用字符串, 是否已由 Marian 译为英文)。
    """
    t = _normalize_ws(text.replace("\n", " "))
    if not t:
        return "", False
    if is_english_text(t):
        return t, False
    if should_translate_zh_via_helsinki(t):
        try:
            out = translate_zh_en_helsinki(t, pretrained_root=_translation_models_root())
            return out, True
        except BaseException as e:  # noqa: BLE001
            truncated = (t[:200] + "…") if len(t) > 200 else t
            sub = ""
            if substitute_with_target_ocr:
                sub = _normalize_ws(substitute_with_target_ocr.replace("\n", " "))
            if sub:
                stab = (sub[:120] + "…") if len(sub) > 120 else sub
                print(
                    "[WARN] Helsinki-NLP 不可用：已改用配对 target 图的 OCR 文本参与相似度: "
                    f"{type(e).__name__}: {e}\n"
                    f"       原中文本摘录: {truncated!r}\n"
                    f"       替换 OCR 摘录: {stab!r}",
                    file=sys.stderr,
                )
                return sub, False
            print(
                f"[WARN] Helsinki-NLP 不可用，已退回本侧 OCR 原文: {type(e).__name__}: {e}\n"
                f"       摘录: {truncated!r}",
                file=sys.stderr,
            )
            return t, False
    return t, False


def english_text_similarity(a: str, b: str, st_model: Any | None) -> float:
    """
    两段英文相似度，范围约 [0,1]：优先 sentence-transformers 余弦；否则 difflib。
    """
    if not a.strip() and not b.strip():
        return 1.0
    if not a.strip() or not b.strip():
        return 0.0

    if st_model is not None:
        try:
            import numpy as np

            e1 = st_model.encode(a, convert_to_numpy=True, normalize_embeddings=True)
            e2 = st_model.encode(b, convert_to_numpy=True, normalize_embeddings=True)
            return float(np.dot(e1, e2))
        except Exception:  # noqa: BLE001
            from difflib import SequenceMatcher

            return float(SequenceMatcher(None, a.lower(), b.lower()).ratio())

    from difflib import SequenceMatcher

    return float(SequenceMatcher(None, a.lower(), b.lower()).ratio())


def _easyocr_weights_look_present(ocr_dir: Path) -> bool:
    """是否已有可用的 EasyOCR 权重（``.pth`` / ``.zip``）；用于避免重复下载。"""
    if not ocr_dir.is_dir():
        return False
    pths = list(ocr_dir.rglob("*.pth"))
    if len(pths) >= 2:
        return True
    return len(list(ocr_dir.rglob("*.zip"))) >= 1


def load_easyocr_reader(
    gpu: bool | None,
    pretrained_root: Path,
    *,
    allow_download: bool = False,
) -> Any:
    """
    EasyOCR 权重目录：``pretrained_root/easyocr``。

    **默认** ``allow_download=False``：仅使用本地权重（``download_enabled=False``），不触发联网下载；
    若目录内未见可用 ``.pth`` / ``.zip``，则抛出 ``RuntimeError``。
    传入 ``allow_download=True`` 可在无本地权重时退回 EasyOCR 官方下载。
    """
    import inspect

    import easyocr  # type: ignore[import-untyped]

    ocr_dir = pretrained_root / "easyocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    use_gpu = gpu if gpu is not None else __import__("torch").cuda.is_available()

    present = _easyocr_weights_look_present(ocr_dir)
    extras: dict[str, Any] = {}
    try:
        sig = inspect.signature(easyocr.Reader.__init__)
    except (TypeError, ValueError):
        sig = None

    if sig is not None and "download_enabled" in sig.parameters:
        if allow_download:
            extras["download_enabled"] = False if present else True
        else:
            extras["download_enabled"] = False
            if not present:
                raise RuntimeError(
                    f"EasyOCR 未在本地找到权重（目录 `{ocr_dir}` 内需有若干 *.pth 或 *.zip）。"
                    "请将权重放入该目录，或传入 allow_download=True / CLI `--easyocr-allow-download`。"
                )
    elif not present and not allow_download:
        raise RuntimeError(
            f"EasyOCR 未在本地找到权重（目录 `{ocr_dir}`）。当前 Reader 不支持 download_enabled，"
            "无法禁用下载；请放入权重或使用 `--easyocr-allow-download`。"
        )

    if present:
        den = extras.get("download_enabled")
        note = f"download_enabled={den}" if den is not None else "Reader 无 download_enabled 参数"
        print(
            f"[INFO] EasyOCR 使用本地权重 `{ocr_dir}`（{note}）。",
            file=sys.stderr,
        )
    elif allow_download:
        print(
            f"[INFO] EasyOCR 本地 `{ocr_dir}` 未见权重，将向官方源下载…",
            file=sys.stderr,
        )

    return easyocr.Reader(
        ["ch_sim", "en"],
        gpu=use_gpu,
        verbose=False,
        model_storage_directory=str(ocr_dir),
        **extras,
    )


def _configure_hf_env_under_pretrained(pretrained_root: Path) -> None:
    """
    将 Hugging Face 相关缓存默认放在 ``pretrained_root/huggingface``，
    便于与仓库内其它模型目录统一；若进程外已设置 HF_HOME 则不覆盖。
    """
    import os

    hf_root = pretrained_root / "huggingface"
    hf_root.mkdir(parents=True, exist_ok=True)
    hub = hf_root / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_root))
    os.environ.setdefault("HF_HUB_CACHE", str(hub))


def _resolve_st_model_id(model_arg: str, repo: Path) -> str:
    """
    Hugging Face 模型 id，或已是本机快照目录时为绝对路径字符串。
    相对仓库路径若为目录则按本地模型加载。
    """
    raw = Path(model_arg).expanduser()
    cand = raw if raw.is_absolute() else _resolve_under_repo(raw, repo)
    if cand.is_dir():
        return str(cand.resolve())
    return model_arg


def _hub_org_repo_from_model_id(hub_id: str) -> tuple[str, str] | None:
    """
    若 ``hub_id`` 为 Hub id（非本机路径外观），解析为 ``(organization, repo_name)``。
    无斜杠的简单名默认为 ``sentence-transformers/<name>``。
    """
    m = hub_id.strip()
    if not m or m.startswith((".", "/")) or "\\" in m:
        return None
    # Windows 盘上路径如 C:\ 已由 strip 前缀排除；"data/foo" 会误判为 Hub id
    if "/" in m:
        parts = m.split("/", 1)
        o, r = parts[0].strip(), parts[1].strip()
        if o and r and not Path(m).suffix:
            return o, r
        return None
    if "/" not in m and ":" not in m and Path(m).suffix:
        return None
    # 单层名：SentenceTransformer / Hub 约定的 mini 模型多在 sentence-transformers 组织下
    return "sentence-transformers", m


def _canonical_st_hub_repo_id(hub_model_id: str) -> str | None:
    """形如 ``sentence-transformers/all-MiniLM-L6-v2``；仅当字符串可解析为 Hub id。"""
    p = _hub_org_repo_from_model_id(hub_model_id)
    return None if not p else f"{p[0]}/{p[1]}"


def _hf_hub_cached_model_folder_name(org: str, repo: str) -> str:
    """与 huggingface_hub 本地缓存目录名一致：``models--<org>--<repo>``。"""
    return f"models--{org}--{repo}"


def _sentence_transformers_dir_complete(d: Path) -> bool:
    """判断是否像可用的 SentenceTransformer 快照根目录（避免误把空目录当模型）。"""
    if not d.is_dir():
        return False
    if not (d / "modules.json").is_file():
        return False
    if not (d / "config.json").is_file():
        return False
    if not ((d / "model.safetensors").is_file() or (d / "pytorch_model.bin").is_file()):
        return False
    return (d / "tokenizer.json").is_file() or (d / "vocab.txt").is_file()


def _st_folder_matches_repo(folder: Path, repo: str) -> bool:
    """在 ``sentence_transformers/`` 等多模型缓存中粗筛与当前 Hub repo 相关的子目录名。"""
    name = folder.name.lower()
    r = repo.lower().replace("_", "-")
    if name == r:
        return True
    if r in name:
        return True
    return name.endswith(r)


def find_pretrained_sentence_transformers_snapshot(
    pretrained_root: Path,
    hub_model_id: str,
) -> Path | None:
    """
    在 ``pretrained_root`` 下查找已存在的、与 ``hub_model_id`` 对应的 ST 快照目录。
    顺序：``<root>/<repo>`` → ``huggingface/hub/models--org--repo/snapshots/*`` →
    ``sentence_transformers/*`` 下名称匹配且结构完整者。
    """
    parsed = _hub_org_repo_from_model_id(hub_model_id)
    if not parsed:
        return None
    org, repo = parsed

    direct = (pretrained_root / repo).resolve()
    if _sentence_transformers_dir_complete(direct):
        return direct

    hub_snap_base = (
        pretrained_root / "huggingface" / "hub" / _hf_hub_cached_model_folder_name(org, repo)
    )
    snap_root = hub_snap_base / "snapshots"
    if snap_root.is_dir():
        # 任一字目录即一次完整下载；通常仅一个 revision
        for child in sorted(snap_root.iterdir(), key=lambda p: p.name):
            if child.is_dir() and _sentence_transformers_dir_complete(child):
                return child.resolve()

    st_cache = pretrained_root / "sentence_transformers"
    if st_cache.is_dir():
        for child in sorted(st_cache.iterdir(), key=lambda p: p.name):
            if (
                child.is_dir()
                and _st_folder_matches_repo(child, repo)
                and _sentence_transformers_dir_complete(child)
            ):
                return child.resolve()

    return None


def _enable_hf_hub_offline_env() -> None:
    """已确认使用本地快照时打开，避免 transformers / huggingface_hub 再连网。"""
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def download_sentence_transformer_into_pretrained_models(
    pretrained_root: Path,
    hub_model_id: str,
) -> Path | None:
    """
    若 ``hub_model_id`` 可解析且本地尚无完整 ST 快照，则通过 huggingface_hub 下载到
    ``pretrained_root/huggingface/hub``（与 SentenceTransformer 缓存布局一致）。
    成功返回快照目录路径，失败返回 ``None``。下载前后会临时撤销进程内离线环境变量。
    """
    existing = find_pretrained_sentence_transformers_snapshot(pretrained_root, hub_model_id)
    if existing is not None:
        return existing

    repo = _canonical_st_hub_repo_id(hub_model_id)
    if not repo:
        return None

    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
    except ImportError:
        print("[WARN] 无法预下载 SentenceTransformer：请安装 huggingface_hub。", file=sys.stderr)
        return None

    import os

    hub_root = pretrained_root / "huggingface" / "hub"
    hub_root.mkdir(parents=True, exist_ok=True)

    saved_hf = os.environ.pop("HF_HUB_OFFLINE", None)
    saved_tf = os.environ.pop("TRANSFORMERS_OFFLINE", None)
    print(
        f"[INFO] SentenceTransformer 模型在 pretrainedModels 中未就绪，从 Hub 拉取 `{repo}` → {hub_root} …",
        file=sys.stderr,
    )
    try:
        snapshot_download(
            repo_id=repo,
            cache_dir=str(hub_root),
            local_files_only=False,
        )
    except BaseException as e:  # noqa: BLE001 — 打印后继续走兜底加载逻辑
        print(
            f"[WARN] `{repo}` 下载失败 ({type(e).__name__}: {e})",
            file=sys.stderr,
        )
        if saved_hf is not None:
            os.environ["HF_HUB_OFFLINE"] = saved_hf
        if saved_tf is not None:
            os.environ["TRANSFORMERS_OFFLINE"] = saved_tf
        return None

    got = find_pretrained_sentence_transformers_snapshot(pretrained_root, hub_model_id)
    if got is None:
        print("[WARN] 下载结束但未解析到可用的 ST 快照目录，仍将尝试在线加载。", file=sys.stderr)
    return got


def load_sentence_model(
    model_arg: str,
    pretrained_root: Path,
    *,
    offline_st: bool,
) -> Any | None:
    """
    载入 SentenceTransformer；失败或无网报错时返回 None（调用方用 difflib）。
    ``--offline-st``：不触发任何 Hub 加载。

    若非 ``--offline-st``：优先使用 ``pretrainedModels`` 下已有完整快照；
    若仅能解析为 Hub id 且无快照，则先 ``snapshot_download`` 写入该目录，
    再启用 ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``，仅从本地路径载入。
    """
    if offline_st:
        print(
            "[INFO] --offline-st：跳过 SentenceTransformer，文本相似度使用 difflib。",
            file=sys.stderr,
        )
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    except ImportError:
        print(
            "[WARN] 未安装 sentence-transformers，相似度使用 difflib。"
            "有网时可：pip install sentence-transformers",
            file=sys.stderr,
        )
        return None

    st_dir = pretrained_root / "sentence_transformers"
    st_dir.mkdir(parents=True, exist_ok=True)
    model_id = _resolve_st_model_id(model_arg, REPO_ROOT)
    cand_path = Path(model_id)

    if cand_path.is_dir():
        _enable_hf_hub_offline_env()
        print(
            "[INFO] SentenceTransformer 使用本机目录，已设置 HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE。",
            file=sys.stderr,
        )
    else:
        # 先有本地完整快照则用本地并离线；否则优先下载到 pretrainedModels 再离线使用
        local_snap = find_pretrained_sentence_transformers_snapshot(pretrained_root, model_id)
        if local_snap is not None:
            print(
                f"[INFO] 检测到本地 SentenceTransformer 快照 {local_snap}，"
                "使用 HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE，不再访问 Hub。",
                file=sys.stderr,
            )
            _enable_hf_hub_offline_env()
            model_id = str(local_snap)
            cand_path = local_snap
        elif _canonical_st_hub_repo_id(model_id):
            fetched = download_sentence_transformer_into_pretrained_models(
                pretrained_root, model_id
            )
            if fetched is not None:
                _enable_hf_hub_offline_env()
                print(
                    "[INFO] 已下载并完成解析，载入前已设置 HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE。",
                    file=sys.stderr,
                )
                model_id = str(fetched)
                cand_path = fetched

    if cand_path.is_dir():
        repair_sentence_transformer_snapshot_layout(cand_path)

    last_err: BaseException | None = None
    # 先试新版接口（减少对 cache_folder 顶层弃用警告）
    ctor_args: tuple[dict[str, Any], ...]
    if cand_path.is_dir():
        ctor_args = ({},)
    else:
        ctor_args = (
            {"model_kwargs": {"cache_dir": str(st_dir)}},
            {"cache_folder": str(st_dir)},
        )

    for kwargs in ctor_args:
        try:
            return SentenceTransformer(model_id, **kwargs)
        except TypeError as e:
            last_err = e
            continue
        except BaseException as e:  # 含网络异常、HF RuntimeError（如 client closed）
            last_err = e
            continue

    print(
        "[WARN] SentenceTransformer 无法加载 "
        f"({model_id!r}; 缓存根 {st_dir})。最后一个错误类型: "
        f"{type(last_err).__name__}: {last_err}",
        file=sys.stderr,
    )
    print(
        "[WARN] 已改用 difflib 文本相似度。可选：pip install -U sentence-transformers transformers "
        "对齐版本；离线请加 --offline-st；或修复 --st-model 指向完整快照目录。",
        file=sys.stderr,
    )
    return None


def similarity_method_label(st_model: Any | None) -> str:
    return (
        "SentenceTransformer（向量余弦，[0,1] 量级）"
        if st_model is not None
        else "difflib.SequenceMatcher（字符级比值）"
    )


def run_one_pair(
    src_img: Path,
    tgt_img: Path,
    *,
    reader: Any,
    st_model: Any | None,
) -> dict[str, Any]:
    ocr_s = ocr_image(src_img, reader)
    ocr_t = ocr_image(tgt_img, reader)
    en_s, _ = ensure_english(ocr_s, substitute_with_target_ocr=ocr_t)
    en_t, _ = ensure_english(ocr_t)
    sim = english_text_similarity(en_s, en_t, st_model)
    return {
        "stem_label": stem_label_for_pair(src_img, tgt_img),
        "similarity": sim,
    }


def format_content_result_md(
    rows: list[dict[str, Any]],
    *,
    avg_similarity: float,
    header_note: str,
    n_failed: int = 0,
) -> str:
    """
    统一报告：主文件名（stem）、相似度表格 + 末尾平均相似度。
    rows 每项须含 ``stem_label``、``similarity``。
    """
    lines = [
        "# 内容文本相似度",
        "",
        header_note,
        "",
        "| 主文件名（stem） | 相似度 |",
        "| --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['stem_label']}` | {float(r['similarity']):.6f} |"
        )
    if not rows:
        lines.append("| （无成功条目） | — |")

    summary_lines = [
        "",
        "## 汇总",
        "",
        f"- **平均相似度（仅成功条目）**: **{avg_similarity:.6f}**",
        f"- 成功条目数: {len(rows)}",
    ]
    if n_failed:
        summary_lines.append(f"- 失败条目数: {n_failed}")

    summary_lines.append("")
    return "\n".join(lines + summary_lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OCR + 英文化 + 英文文本相似度；单图或与 layout 对齐的 --dir 批量。"
    )
    p.add_argument(
        "-s",
        "--source",
        default=DEFAULT_SOURCE_IMG,
        help="source 截图（单图）",
    )
    p.add_argument(
        "-t",
        "--target",
        default=DEFAULT_TARGET_IMG,
        help="target 截图（单图）",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_CONTENT_OUT),
        help=f"输出目录；报告固定为 ``{CONTENT_RESULT_MD}``（默认目录 {DEFAULT_CONTENT_OUT}）",
    )
    p.add_argument(
        "--models-root",
        default=str(DEFAULT_PRETRAINED_MODELS_ROOT),
        help=f"预训练模型根目录（默认 {DEFAULT_PRETRAINED_MODELS_ROOT}；子目录 easyocr、sentence_transformers、huggingface）",
    )
    p.add_argument(
        "--st-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer Hub 模型 id（缺省时若本地无快照会先下载到 pretrainedModels）；或本机快照路径",
    )
    p.add_argument(
        "--offline-st",
        action="store_true",
        help="不加载 SentenceTransformer（不访问 Hub），仅用 difflib",
    )
    p.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="EasyOCR 是否用 GPU；默认自动检测 cuda",
    )
    p.add_argument(
        "--easyocr-allow-download",
        action="store_true",
        help="无本地 EasyOCR 权重时允许从官方下载（默认禁止，仅用 pretrainedModels/easyocr）",
    )

    batch = p.add_argument_group("文件夹批量（与 layout_similarity_calculation 对齐）")
    batch.add_argument("--dir", action="store_true", help="启用批量，需 --source-dir / --target-dir")
    batch.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        help="source 图目录（与 layout_similarity_calculation 默认一致）",
    )
    batch.add_argument(
        "--target-dir",
        default=DEFAULT_TARGET_DIR,
        help="target 图目录（与 layout_similarity_calculation 默认一致）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = _resolve_under_repo(args.output_dir, REPO_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    models_root = _resolve_under_repo(args.models_root, REPO_ROOT)
    models_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 预训练权重目录: {models_root}")

    _configure_hf_env_under_pretrained(models_root)
    set_translation_pretrained_root(models_root)
    warmup_helsinki_zh_en(models_root)

    reader = load_easyocr_reader(args.gpu, models_root, allow_download=args.easyocr_allow_download)
    st_model = load_sentence_model(args.st_model, models_root, offline_st=args.offline_st)
    sim_backend = similarity_method_label(st_model)

    if args.dir:
        src_dir = _resolve_under_repo(args.source_dir, REPO_ROOT)
        tgt_dir = _resolve_under_repo(args.target_dir, REPO_ROOT)
        pairs = collect_same_stem_image_pairs(src_dir, tgt_dir)
        if not pairs:
            print(f"[ERROR] 无 stem 配对: {src_dir} / {tgt_dir}", file=sys.stderr)
            sys.exit(2)

        ok: list[dict[str, Any]] = []
        n_fail = 0
        for i, (sp, tp) in enumerate(pairs, 1):
            print(f"[{i}/{len(pairs)}] stem={stem_label_for_pair(sp, tp)!r}")
            try:
                r = run_one_pair(sp, tp, reader=reader, st_model=st_model)
                ok.append(r)
                print(f"  similarity={r['similarity']:.6f}")
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                print(f"  [FAIL] {e}", file=sys.stderr)

        avg_sim = statistics.mean([x["similarity"] for x in ok]) if ok else float("nan")
        note = (
            f"- **运行模式**：`--dir`（按 stem 配对）\n"
            f"- source_dir: `{src_dir}`\n"
            f"- target_dir: `{tgt_dir}`\n"
            f"- **文本相似度**: {sim_backend}"
        )
        result_path = out_root / CONTENT_RESULT_MD
        result_path.write_text(
            format_content_result_md(
                ok,
                avg_similarity=avg_sim,
                header_note=note,
                n_failed=n_fail,
            ),
            encoding="utf-8",
        )
        print(f"\n[DONE] avg_similarity={avg_sim:.6f} -> {result_path}")
        return

    src_img = _resolve_under_repo(args.source, REPO_ROOT)
    tgt_img = _resolve_under_repo(args.target, REPO_ROOT)
    if not src_img.is_file() or not tgt_img.is_file():
        print("[ERROR] source 或 target 图片不存在", file=sys.stderr)
        sys.exit(2)

    r = run_one_pair(src_img, tgt_img, reader=reader, st_model=st_model)
    rows = [r]
    avg_sim = float(r["similarity"])
    note = (
        "- **运行模式**：单图对\n"
        f"- source stem: `{src_img.stem}`\n"
        f"- target stem: `{tgt_img.stem}`\n"
        f"- **文本相似度**: {sim_backend}"
    )
    result_path = out_root / CONTENT_RESULT_MD
    result_path.write_text(
        format_content_result_md(
            rows,
            avg_similarity=avg_sim,
            header_note=note,
            n_failed=0,
        ),
        encoding="utf-8",
    )
    print(f"[OK] similarity={r['similarity']:.6f}")
    print(f"[SAVED] {result_path}")


if __name__ == "__main__":
    print(
        "[DEPRECATED] content_similarity_calculation_ocr.py 已暂时作废，请勿直接运行。\n"
        "请改用 content_similarity_calculation.py（内容相关组件 OCR + 文本双向指标）。\n"
        "本文件仍以库形式供上述脚本 import（EasyOCR / Helsinki / ST 工具函数等）。",
        file=sys.stderr,
    )
    sys.exit(3)
