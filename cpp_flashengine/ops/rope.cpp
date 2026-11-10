// ============================================================================
// rope.cpp — 昇腾 310P 自定义算子模板（CANN 9.0，AI CPU 形态）
//
// 对应 BackendBase.rope(positions, q, k, cos_sin_cache)：对 q/k 最后一维做旋转位置编码。
// 数值核心委托 flash_cann::math_ref::rope_ref（见 math_ref.h，确定正确）。
//
// 流程：先用 positions 从 cos_sin_cache 取每行的 [cos;sin]（head_dim 展开），
//       再对 q、k 分别调用 rope_ref。CANN 注册/解析部分为模板，待板端验证。
// ============================================================================
#include <cstdint>
#include <vector>
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
class RopeCpuKernel : public Kernel {
public:
    uint32_t Compute(const ge::OpKernelContext& ctx) override {
        const auto* pos_t = ctx.Input(0); // positions: int32 [rows]
        const auto* q_t   = ctx.Input(1);
        const auto* k_t   = ctx.Input(2);
        const auto* cs_t  = ctx.Input(3); // cos_sin_cache: [num_pos, head_dim]
        auto*       qo_t  = ctx.Output(0);
        auto*       ko_t  = ctx.Output(1);

        const auto* positions = reinterpret_cast<const int32_t*>(pos_t ? pos_t->GetData() : nullptr);
        const auto* q = reinterpret_cast<const uint16_t*>(q_t ? q_t->GetData() : nullptr);
        const auto* k = reinterpret_cast<const uint16_t*>(k_t ? k_t->GetData() : nullptr);
        const auto* cs = reinterpret_cast<const float*>(cs_t ? cs_t->GetData() : nullptr);
        auto* qo = reinterpret_cast<uint16_t*>(qo_t ? qo_t->MutableData() : nullptr);
        auto* ko = reinterpret_cast<uint16_t*>(ko_t ? ko_t->MutableData() : nullptr);
        if (!positions || !q || !k || !cs || !qo || !ko) return 1;

        // 真实实现从形状推导 rows / head_dim；此处示意。cos_sin 按行 gather 后调用 rope_ref。
        // int head_dim = ...; int rows = ...;
        // std::vector<float> gathered(static_cast<size_t>(rows) * head_dim);
        // for (int r=0;r<rows;++r){ int p=positions[r]; const float* src=cs+(size_t)p*head_dim;
        //   memcpy(gathered.data()+(size_t)r*head_dim, src, head_dim*sizeof(float)); }
        // flash_cann::math_ref::rope_ref(q, gathered.data(), qo, rows, head_dim);
        // flash_cann::math_ref::rope_ref(k, gathered.data(), ko, rows, head_dim);
        (void)positions; (void)q; (void)k; (void)cs; (void)qo; (void)ko;
        return 0;
    }
};
// REGISTER_KERNEL(Rope, RopeCpuKernel)  // 板端真实注册宏
} // namespace aicpu

#ifdef FLASH_CANN_SELFTEST
#include <cstdio>
#include <cmath>
int main() {
    const int rows = 1, head_dim = 4;
    uint16_t x[4], y[4];
    float cs[4] = {1.0f, 0.0f, 0.0f, 1.0f}; // cos=[1,0], sin=[0,1] -> 恒等旋转
    // 布局：前半 x1，后半 x2（与 BackendBase.rope 一致）
    x[0] = flash_cann::math_ref::float_to_fp16(0.5f); // x1[0]
    x[1] = flash_cann::math_ref::float_to_fp16(0.0f); // x1[1]
    x[2] = flash_cann::math_ref::float_to_fp16(0.0f); // x2[0]
    x[3] = flash_cann::math_ref::float_to_fp16(0.0f); // x2[1]
    flash_cann::math_ref::rope_ref(x, cs, y, rows, head_dim);
    float exp0 = 0.5f * 1.0f - 0.0f * 0.0f;
    float got  = flash_cann::math_ref::fp16_to_float(y[0]);
    bool ok = std::fabs(got - exp0) < 0.02f;
    printf("rope selftest: y[0]=%f (期望 %f) %s\n",
           static_cast<double>(got), static_cast<double>(exp0), ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
#endif
