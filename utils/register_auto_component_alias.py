# -*- coding: utf-8 -*-
"""
旧 checkpoint（如 ``pretrainedModels/yolo/models/best.pt``）在 ``torch.save`` 时可能把模型类挂在
``auto_component.*`` 名下（训练工程将 Ultralytics 源码目录命名为 auto_component）。

在 **已经能够 ``import ultralytics``** 之后、**``YOLO(weights)``** 之前调用
``ensure_auto_component_alias()``，为 pickle 反序列化提供同名的 ``auto_component``
模块别名，避免::

    ModuleNotFoundError: No module named 'auto_component'

实现方式：把 ``sys.modules['auto_component']`` 及 ``auto_component.`` 前缀的
子模块项指向与之对应的 ``ultralytics`` 模块对象；补齐训练目录中常见的
``auto_component.yolo``（对应 ``ultralytics.models.yolo``）以及
``auto_component.yolo.ultralytics``（在 yolo 子目录内又嵌套了一份 ultralytics）等别名。
"""

from __future__ import annotations

import importlib
import pkgutil
import sys


def _register_parallel_alias(real: str, alias: str) -> None:
    """将已加载包 ``real`` 及其全部子模块以 ``alias`` 为前缀复制一份 sys.modules 项。"""
    root = importlib.import_module(real)
    sys.modules[alias] = root
    for _finder, name, _ispkg in pkgutil.walk_packages(
        root.__path__,
        prefix=f"{real}.",
    ):
        importlib.import_module(name)
        sys.modules[alias + name[len(real) :]] = sys.modules[name]


def ensure_auto_component_alias() -> None:
    """将 ``auto_component`` 与 ``ultralytics`` 在 ``sys.modules`` 中设为同一套模块。"""
    _register_parallel_alias("ultralytics", "auto_component")

    # auto_component/yolo -> ultralytics.models.yolo
    real_yolo = "ultralytics.models.yolo"
    alias_yolo = "auto_component.yolo"
    try:
        root_y = importlib.import_module(real_yolo)
    except ImportError:
        root_y = None

    if root_y is not None:
        sys.modules[alias_yolo] = root_y
        for _finder, name, _ispkg in pkgutil.walk_packages(
            root_y.__path__,
            prefix=f"{real_yolo}.",
        ):
            importlib.import_module(name)
            sys.modules[alias_yolo + name[len(real_yolo) :]] = sys.modules[name]

    # auto_component/yolo/ultralytics -> 顶层 ultralytics（嵌套克隆目录）
    _register_parallel_alias("ultralytics", "auto_component.yolo.ultralytics")
