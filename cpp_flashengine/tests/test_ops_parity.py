# ============================================================================
# test_ops_parity.py — C++ 自定义算子 vs torch 参考实现的数值对照
#
# 运行环境：开发板（Linux + CANN 9.0 + torch-npu + 已编译的 cpp_flashengine）。
#   cd nano_flashengine_fp16_1_0
#   PYTHONPATH=. python -B cpp_flashengine/tests/test_ops_parity.py
#
# 作用：确认 C++ 自定义算子的输出与原引擎 torch 实现一致（fp16 容差内），
#       作为“C++ 转化”正确性的验收门槛。本机无 CANN/torch 无法运行。
#
# 断言阈值：fp16 累加误差，相对容差 1e-2、绝对容差 5e-2（与 ACL 融合算子同量级）。
# ============================================================================
from __future__ import annotations

import torch

try:
    import cpp_flashengine  # 触发 torch.ops.cpp_flashengine.* 注册
    from cpp_flashengine.python.flash_cann import (  # type: ignore
        rms_norm, add_rms_norm, silu_and_mul, rope,
    )
    HAVE_CANN = True
except Exception as e:  # noqa: BLE001
    print("跳过：自定义算子库不可用（%s）。请在开发板编译后运行。" % e)
    HAVE_CANN = False


def _close(a: torch.Tensor, b: torch.Tensor, rtol=1e-2, atol=5e-2) -> bool:
    a = a.float(); b = b.float()
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def test_rms_norm():
    x = torch.randn(4, 128, dtype=torch.float16, device="npu")
    w = torch.ones(128, dtype=torch.float16, device="npu")
    eps = 1e-6
    got = rms_norm(x, w, eps)
    ref = torch_npu_rms_norm_ref(x, w, eps)
    assert _close(got, ref), (got - ref).abs().max().item()
    print("rms_norm 通过")


def test_add_rms_norm():
    x = torch.randn(4, 128, dtype=torch.float16, device="npu")
    res = torch.randn(4, 128, dtype=torch.float16, device="npu")
    w = torch.ones(128, dtype=torch.float16, device="npu")
    y, res2 = add_rms_norm(x, res, w, 1e-6)
    y_ref, res_ref = torch_npu_add_rms_norm_ref(x, res, w, 1e-6)
    assert _close(y, y_ref) and _close(res2, res_ref)
    print("add_rms_norm 通过")


def test_silu_and_mul():
    x = torch.randn(4, 256, dtype=torch.float16, device="npu")
    got = silu_and_mul(x)
    h = x.shape[-1] // 2
    ref = torch.nn.functional.silu(x[..., :h]) * x[..., h:]
    assert _close(got, ref), (got - ref).abs().max().item()
    print("silu_and_mul 通过")


def test_rope():
    head_dim = 128
    q = torch.randn(8, head_dim, dtype=torch.float16, device="npu")
    k = torch.randn(8, head_dim, dtype=torch.float16, device="npu")
    pos = torch.arange(8, dtype=torch.int32, device="npu")
    # 与原引擎同构的 cos/sin 缓存（num_pos, head_dim）fp16
    cache = torch.randn(2048, head_dim, dtype=torch.float16, device="npu")
    qo, ko = rope(pos, q, k, cache)
    q_ref, k_ref = torch_npu_rope_ref(pos, q, k, cache)
    assert _close(qo, q_ref) and _close(ko, k_ref)
    print("rope 通过")


# ---- torch 参考（与 flashengine/layers、backends/base.py 行为一致）---------
def torch_npu_rms_norm_ref(x, w, eps):
    import torch_npu  # noqa: F401
    return torch.ops.npu_rms_norm(x, w, eps)[0]


def torch_npu_add_rms_norm_ref(x, res, w, eps):
    import torch_npu  # noqa: F401
    return torch.ops.npu_add_rms_norm(x, res, w, eps)[:2]


def torch_npu_rope_ref(pos, q, k, cache):
    # 与原 BackendBase.rope 的纯 torch 回退等价：按 pos 取 cos/sin 后旋转
    cs = cache[pos]                         # [n, head_dim]，前 half=cos 后 half=sin
    cos = cs[..., : cs.shape[-1] // 2]
    sin = cs[..., cs.shape[-1] // 2:]
    q1, q2 = q[..., :q.shape[-1] // 2], q[..., q.shape[-1] // 2:]
    k1, k2 = k[..., :k.shape[-1] // 2], k[..., k.shape[-1] // 2:]
    qo = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    ko = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return qo, ko


if __name__ == "__main__":
    if not HAVE_CANN:
        raise SystemExit(0)
    test_rms_norm()
    test_add_rms_norm()
    test_silu_and_mul()
    test_rope()
    print("全部数值对照通过")
