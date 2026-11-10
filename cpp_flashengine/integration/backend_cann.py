# ============================================================================
# backend_cann.py — 接入引擎的接缝：用 C++ 自定义算子替换 310P 的原生 ACL 算子
#
# 用法：在引擎选择后端时改用本类（或在 config 中指定 cann_cpp 分支）。
#   本类继承 Ascend310PBackend，保留其全部 310P 特性（NZ 布局、NPUGraph、
#   分页 KV、投机图等），仅覆盖四个热路径方法，改调 cpp_flashengine 自定义算子。
#
# 诚实说明：Ascend310PBackend 原本已通过 torch_npu 调用 ACL 融合算子
#   （npu_rms_norm / npu_swiglu / npu_apply_rotary_pos_emb 等）。这里的覆盖是
#   “C++ 转化”的接入点——自定义算子目前为 AI CPU 形态，板上实测若不及 ACL 融合，
#   应回退到父类实现（见各方法末尾的 fallback 注释），并把自定义算子改写为
#   AICore / Ascend C 核以真正发挥 NPU 算力（见 README）。
# ============================================================================
from __future__ import annotations

import torch
import torch.nn.functional as F

# 沿用原 310P 后端实现，避免重复 310P 专属逻辑。
from flashengine.backends.ascend import Ascend310PBackend
from flashengine.backends.base import BackendBase

# C++ 自定义算子封装（接口见 python/flash_cann.py）。
from cpp_flashengine.python.flash_cann import (  # type: ignore
    rms_norm as cann_rms_norm,
    add_rms_norm as cann_add_rms_norm,
    silu_and_mul as cann_silu_and_mul,
    rope as cann_rope,
)


class CannCppBackend(Ascend310PBackend):
    """在 310P 后端基础上，用 C++ 自定义算子覆盖热路径。"""

    name = "cann_cpp"

    # ---- RMSNorm（原 Ascend310PBackend.rms_norm 调用 npu_rms_norm）----------
    @staticmethod
    def rms_norm(x, weight, eps):
        try:
            return cann_rms_norm(x, weight, eps)
        except Exception:
            # 自定义算子不可用/未编译时，回退到 ACL 融合实现，保证引擎可运行。
            return Ascend310PBackend.rms_norm(x, weight, eps)

    # ---- 残差加 RMSNorm（原 npu_add_rms_norm）-------------------------------
    @staticmethod
    def add_rms_norm(x, residual, weight, eps):
        try:
            return cann_add_rms_norm(x, residual, weight, eps)
        except Exception:
            return Ascend310PBackend.add_rms_norm(x, residual, weight, eps)

    # ---- SwiGLU 激活（原 npu_swiglu）---------------------------------------
    @staticmethod
    def silu_and_mul(x):
        try:
            return cann_silu_and_mul(x)
        except Exception:
            # 与父类一致的回退路径：维度对齐 32 用 ACL，否则纯 torch。
            if x.shape[-1] % 32 == 0:
                import torch_npu  # 触发 npu_swiglu 注册
                return torch.ops.npu_swiglu(x) if hasattr(torch.ops, "npu_swiglu") else F.silu(x[..., :x.shape[-1] // 2]) * x[..., x.shape[-1] // 2:]
            h = x.shape[-1] // 2
            return F.silu(x[..., :h]) * x[..., h:]

    # ---- RoPE（原 Ascend310PBackend.rope，实例方法）-------------------------
    def rope(self, positions, q, k, cos_sin_cache):
        try:
            return cann_rope(positions, q, k, cos_sin_cache)
        except Exception:
            return Ascend310PBackend.rope(self, positions, q, k, cos_sin_cache)
