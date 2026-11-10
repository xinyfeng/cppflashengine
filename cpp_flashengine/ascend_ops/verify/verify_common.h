// 可复用板端验证框架：各算子只需提供 CPU 参考函数与 aclnn 两段式调用。
// 新增注释一律使用中文（遵循项目 AGENTS.md）。
#ifndef FLASH_CANN_VERIFY_COMMON_H
#define FLASH_CANN_VERIFY_COMMON_H

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <functional>

namespace flash_cann_verify {

inline uint16_t F32ToF16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (x >> 23) & 0xff;
    uint32_t mant = x & 0x7fffffu;
    if (exp == 255) return (uint16_t)(sign | 0x7c00u);
    if (exp == 0) return (uint16_t)sign;
    int32_t e = exp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - e);
        uint32_t half = (mant >> shift);
        if ((mant >> (shift - 1)) & 1) half += 1;
        return (uint16_t)(sign | half);
    }
    uint32_t half = ((uint32_t)e << 10) | (mant >> 13);
    if (mant & 0x1000u) half += 1;
    return (uint16_t)(sign | half);
}

inline float F16ToF32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    int32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            f = (sign << 31) | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    float r; std::memcpy(&r, &f, 4); return r;
}

// 生成 fp16 随机输入。310P 的 DataCopy 要求元素数是 16 的倍数（half 下 32B 对齐），
// 调用方应自行保证 xShape 的元素总数满足该约束。
inline std::vector<uint16_t> MakeInput(uint32_t total, uint32_t seed = 12345) {
    std::vector<uint16_t> x(total);
    std::srand(seed);
    for (auto& v : x) {
        float f = ((float)std::rand() / RAND_MAX - 0.5f) * 4.0f;
        v = F32ToF16(f);
    }
    return x;
}

inline size_t ShapeBytes(const std::vector<int64_t>& shape) {
    size_t bytes = 2;  // 全部算子均为 FP16
    for (auto d : shape) bytes *= (size_t)d;
    return bytes;
}

// CPU 参考：由各算子实现，输入为展平的 fp16 位模式，输出写回 yOut。
using RefFn = std::function<void(const std::vector<uint16_t>&, std::vector<uint16_t>&)>;

// 算子调用：内部完成 aclnnXxxGetWorkspaceSize + aclnnXxx 两段式调用。
// workspace 指针通过 wsOut 返回，由框架在流同步之后再释放（异步执行不能提前释放）。
using OpFn = std::function<aclnnStatus(aclTensor*, aclTensor*, aclrtStream, void**)>;

inline int VerifyOp(const char* name,
                    const std::vector<int64_t>& xShape,
                    const std::vector<int64_t>& yShape,
                    const std::vector<uint16_t>& xHost,
                    RefFn ref, OpFn op,
                    float tol = 2e-2f) {
    const size_t xb = ShapeBytes(xShape), yb = ShapeBytes(yShape);
    const size_t n = yb / 2;

    if (aclInit(nullptr) != ACL_SUCCESS) { std::printf("aclInit failed\n"); return 2; }
    if (aclrtSetDevice(0) != ACL_SUCCESS) { std::printf("aclrtSetDevice failed\n"); return 2; }
    aclrtStream stream;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) { std::printf("aclrtCreateStream failed\n"); return 2; }

    void* xDev = nullptr; void* yDev = nullptr;
    aclrtMalloc(&xDev, xb, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&yDev, yb, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(xDev, xb, xHost.data(), xb, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemset(yDev, yb, 0x3C, yb);  // 哨兵：用于判断核是否真的写了输出

    // 最后一个参数是设备内存指针，必须传入；传 nullptr 会让算子使用自有内存，
    // 主机端读回的是未初始化数据（曾导致 1912/2048 误判）。
    aclTensor* x = aclCreateTensor(xShape.data(), xShape.size(), ACL_FLOAT16, nullptr, 0,
                                   ACL_FORMAT_ND, xShape.data(), xShape.size(), xDev);
    aclTensor* y = aclCreateTensor(yShape.data(), yShape.size(), ACL_FLOAT16, nullptr, 0,
                                   ACL_FORMAT_ND, yShape.data(), yShape.size(), yDev);

    void* workspace = nullptr;
    aclnnStatus st = op(x, y, stream, &workspace);
    if (st != ACL_SUCCESS) { std::printf("%s aclnn failed %d\n", name, st); return 2; }
    aclrtSynchronizeStream(stream);

    std::vector<uint16_t> yHost(n);
    aclrtMemcpy(yHost.data(), yb, yDev, yb, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<uint16_t> expect;
    ref(xHost, expect);

    int sentinel = 0, mism = 0;
    float maxerr = 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (yHost[i] == 0x3C3C) sentinel++;
        float a = F16ToF32(yHost[i]);
        float b = (i < expect.size()) ? F16ToF32(expect[i]) : 0.0f;
        float e = std::fabs(a - b);
        if (e > maxerr) maxerr = e;
        if (e > tol) mism++;
    }
    std::printf("VERIFY %s: n=%zu maxerr=%f mismatches=%d sentinel=%d\n",
                name, n, maxerr, mism, sentinel);

    for (size_t i = 0; i < 4 && i < n; i++) {
        std::printf("  i=%zu y=%f ref=%f\n", i,
                    F16ToF32(yHost[i]),
                    (i < expect.size()) ? F16ToF32(expect[i]) : 0.0f);
    }

    if (workspace) aclrtFree(workspace);
    aclrtFree(xDev); aclrtFree(yDev);
    aclDestroyTensor(x); aclDestroyTensor(y);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return (mism == 0 && sentinel == 0) ? 0 : 1;
}

}  // namespace flash_cann_verify
#endif  // FLASH_CANN_VERIFY_COMMON_H
