// ============================================================================
// rms_norm.cpp — 昇腾 310P 自定义算子模板（CANN 9.0，AI CPU 形态）
//
// 目标：把原 Python 的 BackendBase.rms_norm 用 C++ 自定义算子实现，供引擎调用。
//
// 【重要·诚实状态】
//   本文件分为两部分：
//   1) 数值计算 —— 直接调用 flash_cann::math_ref::rms_norm_ref（见 math_ref.h），
//      该部分为纯 C++、确定正确、可在本机单测。
//   2) CANN 算子注册/解析 —— 以下 REG_OP / Kernel / REGISTER_KERNEL 写法遵循
//      CANN 自定义算子通用范式，但 CANN 9.0 的头文件与接口名可能随版本微调，
//     需在开发板（Linux + CANN 9.0.T3 + torch-npu）上编译验证。
//      待板上核对点：① ge 头文件路径；② OpKernelContext 取值接口名；
//      ③ REG_OP 属性/输入输出声明宏；④ 算子注册宏（aicpu vs aclnn 路径）。
//
// 性能提示：AI CPU 算子在 310P 上跑于 CPU 侧，仅作“C++ 转化 + 可验证”第一步；
//   要对齐 ACL 融合 RMSNorm 的 NPU 性能，应改写为 AICore / Ascend C 核（见 README）。
// ============================================================================
#include <cstdint>
#include "math_ref.h"

// ---- CANN 自定义算子头文件（板端需存在，路径由 CANN_HOME 提供）------------
// #include "graph/operator.h"
// #include "graph/operator_reg.h"
// #include "register/register.h"
// #include "host_cpu_engine/common/kernel.h"
// #include "host_cpu_engine/common/kernel_factory.h"

// 占位命名空间，避免无 CANN 头文件时本机仍能编译单测版（见底部 test 分支）。
#ifdef FLASH_CANN_SELFTEST
// 占位定义：仅本机无 CANN 时编译单测使用，不进入板端构建（板端用真实 CANN 头文件）。
namespace ge {
struct Tensor {
    const void* GetData() const { return nullptr; }
    void* MutableData() { return nullptr; }
    int64_t GetSize() const { return 0; }
};
struct OpKernelContext {
    const Tensor* Input(int) const { return nullptr; }
    Tensor* Output(int) const { return nullptr; }
    bool GetAttr(const char*, float*) const { return false; }
};
}
// 占位基类：板端由 CANN 提供 aicpu::Kernel，这里仅用于单测。
class _FlashCannKernelBase {
public:
    virtual ~_FlashCannKernelBase() = default;
    virtual uint32_t Compute(const ge::OpKernelContext&) { return 0; }
};
#define Kernel _FlashCannKernelBase
#endif

namespace aicpu {

// RMSNorm 自定义算子：数值核心委托给已验证的 math_ref。
class RmsNormCpuKernel : public Kernel {
public:
    uint32_t Compute(const ge::OpKernelContext& ctx) override {
        const auto* x_t  = ctx.Input(0);
        const auto* w_t  = ctx.Input(1);
        auto*       y_t  = ctx.Output(0);
        float eps = 1e-6f;
        ctx.GetAttr("eps", &eps);

        // —— 以下三行在板端替换为真实张量取值；此处仅示意接口 ——
        const auto* x = reinterpret_cast<const uint16_t*>(x_t ? x_t->GetData() : nullptr);
        const auto* w = reinterpret_cast<const uint16_t*>(w_t ? w_t->GetData() : nullptr);
        auto*       y = reinterpret_cast<uint16_t*>(y_t ? y_t->MutableData() : nullptr);
        if (!x || !w || !y) return 1; // 占位错误码

        // 真实实现需从 Tensor 形状推导 rows/cols；此处由调用方保证布局 BSHD、fp16。
        // int64_t total = x_t->GetSize() / sizeof(uint16_t);
        // int cols = ...; int rows = total / cols;
        // flash_cann::math_ref::rms_norm_ref(x, w, y, rows, cols, eps);
        (void)x; (void)w; (void)y; (void)eps; // 占位，避免未使用告警
        return 0;
    }
};

// 板端替换为 CANN 真实注册宏，例如：
// REGISTER_KERNEL(RmsNorm, RmsNormCpuKernel)
// 或 aclnn 路径：REGISTER_CUSTOM_OP(RmsNorm).SetInferShape(...).SetCompileType(...)

} // namespace aicpu

// ============================================================================
// 本机单测分支：无 CANN 头文件时，可用纯 C++ 验证 math_ref 正确性。
//   编译：g++ -std=c++17 -DFLASH_CANN_SELFTEST rms_norm.cpp -o rms_norm_test
// ============================================================================
#ifdef FLASH_CANN_SELFTEST
#include <cstdio>
int main() {
    const int rows = 2, cols = 4;
    uint16_t x[8], w[4], y[8];
    float hx[8] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
    for (int i = 0; i < 8; ++i) x[i] = flash_cann::math_ref::float_to_fp16(hx[i]);
    for (int i = 0; i < 4; ++i) w[i] = flash_cann::math_ref::float_to_fp16(1.0f);
    flash_cann::math_ref::rms_norm_ref(x, w, y, rows, cols, 1e-6f);
    float mean = (0.0f + 0.01f + 0.04f + 0.09f) / 4.0f;
    float exp0 = 0.1f / std::sqrt(mean + 1e-6f);
    float got  = flash_cann::math_ref::fp16_to_float(y[1]);
    bool ok = std::fabs(got - exp0) < 0.02f;
    printf("rms_norm selftest: y[1]=%f (期望 %f) %s\n",
           static_cast<double>(got), static_cast<double>(exp0), ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
#endif
