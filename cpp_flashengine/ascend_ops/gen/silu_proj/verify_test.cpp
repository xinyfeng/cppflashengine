#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_silu_and_mul.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

static inline uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
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
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    int32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) { if (mant == 0) { f = sign << 31; } else { exp = 1; while (!(mant & 0x400u)) { mant <<= 1; exp--; } mant &= 0x3ffu; f = (sign << 31) | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13); } }
    else if (exp == 31) { f = (sign << 31) | 0x7f800000u | (mant << 13); }
    else { f = (sign << 31) | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13); }
    float r; memcpy(&r, &f, 4); return r;
}

static void ref(const std::vector<uint16_t>& xh, std::vector<uint16_t>& yh) {
    uint32_t n = (uint32_t)xh.size() / 2;
    yh.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        float g = f16_to_f32(xh[i]);
        float u = f16_to_f32(xh[i + n]);
        float s = g / (1.0f + expf(-g));
        yh[i] = f32_to_f16(s * u);
    }
}

int main() {
    uint32_t M = 8, K = 256;
    uint32_t total = M * 2 * K;
    uint32_t n = total / 2;
    std::vector<uint16_t> xh(total);
    srand(12345);
    for (auto& v : xh) { float f = ((float)rand() / RAND_MAX - 0.5f) * 4.0f; v = f32_to_f16(f); }

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream; aclrtCreateStream(&stream);

    size_t xb = total * 2, yb = n * 2;
    uint8_t *xDev, *yDev;
    aclrtMalloc((void**)&xDev, xb, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&yDev, yb, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(xDev, xb, xh.data(), xb, ACL_MEMCPY_HOST_TO_DEVICE);

    std::vector<int64_t> xshape = {(int64_t)total};
    std::vector<int64_t> yshape = {(int64_t)n};
    aclTensor* x = aclCreateTensor(xshape.data(), 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, xshape.data(), 1, xDev);
    aclTensor* y = aclCreateTensor(yshape.data(), 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, yshape.data(), 1, yDev);

    uint64_t ws = 0; aclOpExecutor* ex = nullptr;
    aclnnStatus st = aclnnSiluAndMulGetWorkspaceSize(x, y, &ws, &ex);
    if (st != ACL_SUCCESS) { printf("GetWorkspaceSize failed %d\n", st); return 2; }
    void* wsp = nullptr;
    if (ws > 0) aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    st = aclnnSiluAndMul(wsp, ws, ex, stream);
    if (st != ACL_SUCCESS) { printf("aclnnSiluAndMul failed %d\n", st); return 2; }
    aclrtSynchronizeStream(stream);

    std::vector<uint16_t> yh(n);
    aclrtMemcpy(yh.data(), yb, yDev, yb, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<uint16_t> r; ref(xh, r);
    int mism = 0; float maxerr = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float a = f16_to_f32(yh[i]), b = f16_to_f32(r[i]);
        float e = fabsf(a - b);
        if (e > maxerr) maxerr = e;
        if (e > 2e-2f) mism++;
    }
    for (uint32_t i = 0; i < 4; i++)
        printf("i=%u x_g=%f x_u=%f y=%f ref=%f\n", i,
               f16_to_f32(xh[i]), f16_to_f32(xh[i+n]), f16_to_f32(yh[i]), f16_to_f32(r[i]));
    printf("VERIFY silu_and_mul: n=%u maxerr=%f mismatches=%d\n", n, maxerr, mism);

    if (wsp) aclrtFree(wsp);
    aclrtFree(xDev); aclrtFree(yDev);
    aclDestroyTensor(x); aclDestroyTensor(y);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return mism ? 1 : 0;
}
