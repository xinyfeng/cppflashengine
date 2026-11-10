# ============================================================================
# flash_cann.py — C++ 自定义算子的 Python 封装（接入引擎的桥）
#
# 职责：把编译后的 cpp_flashengine 自定义算子库暴露成与 BackendBase 调用点
#       一致的函数签名，供 integration/backend_cann.py 直接替换原 ACL 算子。
#
# 调用约定（待板上对齐）：算子库通过 torch_npu 自定义算子机制注册为
#   torch.ops.cpp_flashengine.<op>(...)
#   —— ops/*.cpp 里的注册名必须与此处一致；若改为 aicpu 注册路径，则此处
#      需改为 acl 接口调用（见 README “板上构建”一节）。
#
# 诚实状态：本文件可在板端随引擎导入，但真正生效依赖 cpp_flashengine 算子库
#           已正确编译注册；编译/注册部分尚未在本机验证。
# ============================================================================
from __future__ import annotations

import torch

# 编译产物 cpp_flashengine.*.so 需已置于 PYTHONPATH 或本目录。
try:
    import cpp_flashengine  # noqa: F401  触发自定义算子注册（torch.ops.cpp_flashengine.*）
except ImportError as e:  # 未编译时给出明确提示，不静默失败
    raise ImportError(
        "未找到编译后的 cpp_flashengine 算子库。请先在开发板按 README 构建，"
        "并将生成的 .so 放到 PYTHONPATH。原始错误: %s" % e
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """对应 BackendBase.rms_norm：返回归一化后的张量（fp16）。"""
    return torch.ops.cpp_flashengine.rms_norm(x, weight, eps)


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor,
                 weight: torch.Tensor, eps: float):
    """对应 BackendBase.add_rms_norm：返回 (归一化输出, 合并后的残差)。"""
    y, residual = torch.ops.cpp_flashengine.add_rms_norm(x, residual, weight, eps)
    return y, residual


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """对应 BackendBase.silu_and_mul：SwiGLU 激活，最后一维减半。"""
    return torch.ops.cpp_flashengine.silu_and_mul(x)


def rope(positions, q: torch.Tensor, k: torch.Tensor,
         cos_sin_cache: torch.Tensor):
    """对应 BackendBase.rope：对 q/k 应用旋转位置编码，返回 (q, k)。

    说明：自定义算子内部已按 positions 从 cos_sin_cache 取 [cos;sin]，
    因此这里直接透传，保持与原 rope 的外部契约一致。
    """
    q_out, k_out = torch.ops.cpp_flashengine.rope(positions, q, k, cos_sin_cache)
    return q_out, k_out
