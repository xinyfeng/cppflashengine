// silu_and_mul 的板端数值验证。新增算子时复制本文件，改三处即可：
//   1) @proj / @op 注释；2) CpuRef 的公式；3) CallOp 里的 aclnn 两段式调用名。
// 新增注释一律使用中文（遵循项目 AGENTS.md）。
// @proj silu_proj
// @op   silu_and_mul
#include "verify_common.h"
#include "aclnn_silu_and_mul.h"

using namespace flash_cann_verify;

static void CpuRef(const std::vector<uint16_t>& xh, std::vector<uint16_t>& yh) {
    const size_t n = xh.size() / 2;
    yh.resize(n);
    for (size_t i = 0; i < n; i++) {
        float g = F16ToF32(xh[i]);
        float u = F16ToF32(xh[i + n]);
        float s = g / (1.0f + std::exp(-g));
        yh[i] = F32ToF16(s * u);
    }
}

static aclnnStatus CallOp(aclTensor* x, aclTensor* y, aclrtStream stream, void** wsOut) {
    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    aclnnStatus st = aclnnSiluAndMulGetWorkspaceSize(x, y, &ws, &ex);
    if (st != ACL_SUCCESS) return st;
    if (ws > 0) aclrtMalloc(wsOut, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    return aclnnSiluAndMul(*wsOut, ws, ex, stream);
}

int main() {
    // 4096 = 8 行 × 2 × 256；n=2048，满足 310P DataCopy 的 16 元素对齐要求。
    const uint32_t total = 4096;
    std::vector<int64_t> xShape = {(int64_t)total};
    std::vector<int64_t> yShape = {(int64_t)(total / 2)};
    std::vector<uint16_t> xHost = MakeInput(total);
    return VerifyOp("silu_and_mul", xShape, yShape, xHost, CpuRef, CallOp);
}
