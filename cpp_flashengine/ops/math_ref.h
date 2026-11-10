// ============================================================================
// math_ref.h — 可移植 C++ 数学核（与 flashengine/layers、backends/base.py 对应）
//
// 作用：把原 Python 引擎里的 RMSNorm / SwiGLU(silu_and_mul) / RoPE 三个热路径，
//       用纯 C++ 重新实现，作为“C++ 转化”的数值基准（reference）。
//
// 精度约定：所有计算在 float32 累加，结果按 fp16 舍入，与原引擎
//           backends/base.py 中 `x.float()` -> 计算 -> `.to(orig_dtype)` 的
//           行为一致，便于与 torch 参考实现做数值对照。
//
// 注意：本文件不依赖任何昇腾/CANN 头文件，可在任意 C++17 编译器上单测。
//       昇腾自定义算子(.cpp)直接调用这里的 *ref 函数，保证数值核心确定正确，
//       待板上验证的只是 CANN 算子注册/搬运（ plumbing ）部分。
// ============================================================================
#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>

namespace flash_cann {
namespace math_ref {

// ---- fp16 <-> float 位级转换（IEEE 754 half，无外部依赖）-------------------
inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            // 非规格化数
            float v = static_cast<float>(mant) * std::exp2(-24 + 1);
            std::memcpy(&f, &v, sizeof(f));
            f |= sign << 31;
        }
    } else if (exp == 0x1f) {
        f = (sign << 31) | 0x7f800000 | (mant << 13); // inf / nan
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

inline uint16_t float_to_fp16(float v) {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    uint32_t sign = (f >> 31) & 0x1;
    int32_t  exp  = (f >> 23) & 0xff;
    uint32_t mant = f & 0x7fffff;

    if (exp == 0xff) { // inf / nan
        return static_cast<uint16_t>((sign << 15) | 0x7c00 | (mant >> 13));
    }
    int32_t e = exp - 127 + 15;
    if (e >= 0x1f) { // 上溢 -> inf
        return static_cast<uint16_t>((sign << 15) | 0x7c00);
    }
    if (e <= 0) { // 下溢 -> 非规格化/零
        if (e < -10) return static_cast<uint16_t>(sign << 15);
        mant |= 0x800000;
        uint32_t m = mant >> (14 - e);
        return static_cast<uint16_t>((sign << 15) | m);
    }
    return static_cast<uint16_t>((sign << 15) | (e << 10) | (mant >> 13));
}

// ---- RMSNorm（对应 BackendBase.rms_norm / add_rms_norm）-------------------
// 行为等价：var = mean(x^2, last_dim); y = x * rsqrt(var+eps) * weight
inline void rms_norm_ref(const uint16_t* x, const uint16_t* weight,
                         uint16_t* y, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* xr = x + static_cast<int64_t>(r) * cols;
        uint16_t* yr = y + static_cast<int64_t>(r) * cols;
        float ss = 0.f;
        for (int c = 0; c < cols; ++c) {
            float v = fp16_to_float(xr[c]);
            ss += v * v;
        }
        float inv = 1.f / std::sqrt(ss / static_cast<float>(cols) + eps);
        for (int c = 0; c < cols; ++c) {
            float v = fp16_to_float(xr[c]) * inv * fp16_to_float(weight[c]);
            yr[c] = float_to_fp16(v);
        }
    }
}

// add_rms_norm：先 residual = x + residual，再做 RMSNorm，输出(y, residual)
inline void add_rms_norm_ref(const uint16_t* x, uint16_t* residual,
                             const uint16_t* weight, uint16_t* y,
                             int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* xr = x + static_cast<int64_t>(r) * cols;
        uint16_t* rr = residual + static_cast<int64_t>(r) * cols;
        uint16_t* yr = y + static_cast<int64_t>(r) * cols;
        float ss = 0.f;
        for (int c = 0; c < cols; ++c) {
            float added = fp16_to_float(xr[c]) + fp16_to_float(rr[c]);
            rr[c] = float_to_fp16(added);          // 写回合并后的残差
            ss += added * added;
        }
        float inv = 1.f / std::sqrt(ss / static_cast<float>(cols) + eps);
        for (int c = 0; c < cols; ++c) {
            float v = fp16_to_float(rr[c]) * inv * fp16_to_float(weight[c]);
            yr[c] = float_to_fp16(v);
        }
    }
}

// ---- SwiGLU 的激活部分（对应 BackendBase.silu_and_mul）-------------------
// 输入最后一维为 2*half：gate = x[:half], up = x[half:]; out = silu(gate)*up
inline void silu_and_mul_ref(const uint16_t* x, uint16_t* y,
                             int rows, int half_cols) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* xr = x + static_cast<int64_t>(r) * (2 * half_cols);
        uint16_t* yr = y + static_cast<int64_t>(r) * half_cols;
        for (int c = 0; c < half_cols; ++c) {
            float gate = fp16_to_float(xr[c]);
            float up   = fp16_to_float(xr[c + half_cols]);
            float sig  = 1.f / (1.f + std::exp(-gate));
            yr[c] = float_to_fp16(sig * gate * up);
        }
    }
}

// ---- RoPE（对应 BackendBase.rope / _apply_rotary_emb）--------------------
// 对 q/k 最后一维做旋转：x1,x2 = chunk(x,2); y1=x1*cos-x2*sin; y2=x2*cos+x1*sin
// cos_sin 形状 [num_positions, head_dim]（与 cos_sin_cache[positions] 一致）。
inline void rope_ref(const uint16_t* x, const float* cos_sin,
                     uint16_t* y, int rows, int head_dim) {
    // head_dim 必须为偶数；cos_sin 已按 head_dim 展开（cos 在前半，sin 在后半）
    int half = head_dim / 2;
    for (int r = 0; r < rows; ++r) {
        const uint16_t* xr = x + static_cast<int64_t>(r) * head_dim;
        uint16_t* yr = y + static_cast<int64_t>(r) * head_dim;
        const float* cs = cos_sin + static_cast<int64_t>(r) * head_dim;
        for (int i = 0; i < half; ++i) {
            float x1 = fp16_to_float(xr[i]);
            float x2 = fp16_to_float(xr[i + half]);
            float c  = cs[i];
            float s  = cs[i + half];
            yr[i]            = float_to_fp16(x1 * c - x2 * s);
            yr[i + half]     = float_to_fp16(x2 * c + x1 * s);
        }
    }
}

} // namespace math_ref
} // namespace flash_cann
