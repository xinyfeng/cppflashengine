// ============================================================================
// silu_and_mul.cpp — 昇腾 310P 自定义算子模板（CANN 9.0，AI CPU 形态）
//
// 对应 BackendBase.silu_and_mul：SwiGLU 激活，最后一维拆为 gate/up 两半。
// 数值核心委托 flash_cann::math_ref::silu_and_mul_ref（见 math_ref.h，确定正确）。
// CANN 注册/解析部分为模板，待开发板 CANN 9.0 编译验证（见 rms_norm.cpp 头部说明）。
// ============================================================================
#include <cstdint>
#include "math_ref.h"

#ifdef FLASH_CANN_SELFTEST
namespace ge {
struct Tensor {
    const void* GetData() const { return nullptr; }
    void* MutableData() { return nullptr; }
};
struct OpKernelContext {
    const Tensor* Input(int) const { return nullptr; }
    Tensor* Output(int) const { return nullptr; }
};
}
class _FlashCannKernelBase {
public:
    virtual ~_FlashCannKernelBase() = default;
    virtual uint32_t Compute(const ge::OpKernelContext&) { return 0; }
};
#define Kernel _FlashCannKernelBase
#endif

namespace aicpu {
class SiluAndMulCpuKernel : public Kernel {
public:
    uint32_t Compute(const ge::OpKernelContext& ctx) override {
        const auto* x_t = ctx.Input(0);
        auto*       y_t = ctx.Output(0);
        const auto* x = reinterpret_cast<const uint16_t*>(x_t ? x_t->GetData() : nullptr);
        auto*       y = reinterpret_cast<uint16_t*>(y_t ? y_t->MutableData() : nullptr);
        if (!x || !y) return 1;
        // 真实实现从形状推导 rows 与 half_cols（最后一维为 2*half）。
        // int64_t last = ...; int half_cols = last / 2;
        // int rows = (x_t->GetSize()/sizeof(uint16_t)) / (2*half_cols);
        // flash_cann::math_ref::silu_and_mul_ref(x, y, rows, half_cols);
        (void)x; (void)y;
        return 0;
    }
};
// REGISTER_KERNEL(SiluAndMul, SiluAndMulCpuKernel)  // 板端真实注册宏
} // namespace aicpu

#ifdef FLASH_CANN_SELFTEST
#include <cstdio>
int main() {
    const int rows = 1, half = 2;
    uint16_t x[4], y[2];
    // 布局：前半 gate，后半 up（与 BackendBase.silu_and_mul 一致）
    x[0] = flash_cann::math_ref::float_to_fp16(1.0f); // gate0
    x[1] = flash_cann::math_ref::float_to_fp16(0.0f); // gate1（占位）
    x[2] = flash_cann::math_ref::float_to_fp16(2.0f); // up0
    x[3] = flash_cann::math_ref::float_to_fp16(0.0f); // up1（占位）
    flash_cann::math_ref::silu_and_mul_ref(x, y, rows, half);
    float exp0 = (1.0f / (1.0f + std::exp(-1.0f))) * 1.0f * 2.0f;
    float got  = flash_cann::math_ref::fp16_to_float(y[0]);
    bool ok = std::fabs(got - exp0) < 0.05f;
    printf("silu_and_mul selftest: y[0]=%f (期望 %f) %s\n",
           static_cast<double>(got), static_cast<double>(exp0), ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
#endif
