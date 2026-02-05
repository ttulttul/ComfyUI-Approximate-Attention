import os
import sys

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODULE_DIR not in sys.path:
    sys.path.append(_MODULE_DIR)

try:
    from nodes_taylor_attention import TaylorAttentionExtension, comfy_entrypoint  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name == "comfy_api":
        TaylorAttentionExtension = None
        comfy_entrypoint = None
        __all__ = []
    else:
        raise
else:
    __all__ = ["TaylorAttentionExtension", "comfy_entrypoint"]
