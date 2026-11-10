#include "kernel_operator.h"
#include "silu_and_mul_tiling.h"

namespace SiluAndMulNS {
using namespace AscendC;

constexpr uint32_t TILE = 2048;
constexpr uint32_t BUFNUM = 1;

class KernelSiluAndMul {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
    {
        xGm.SetGlobalBuffer((__gm__ half*)x);
        yGm.SetGlobalBuffer((__gm__ half*)y);
        pipe.InitBuffer(gateQ, BUFNUM, TILE * sizeof(half));
        pipe.InitBuffer(upQ,   BUFNUM, TILE * sizeof(half));
        pipe.InitBuffer(outQ,  BUFNUM, TILE * sizeof(half));
        pipe.InitBuffer(gFb, TILE * sizeof(float));
        pipe.InitBuffer(uFb, TILE * sizeof(float));
        pipe.InitBuffer(nFb, TILE * sizeof(float));
        pipe.InitBuffer(eFb, TILE * sizeof(float));
        pipe.InitBuffer(sFb, TILE * sizeof(float));
        pipe.InitBuffer(oFb, TILE * sizeof(float));
    }

    __aicore__ inline void Process(uint32_t total)
    {
        n = total / 2;
        uint32_t off = 0;
        while (off < n) {
            uint32_t tile = (n - off < TILE) ? (n - off) : TILE;
            CopyIn(off, tile);
            Compute(tile);
            CopyOut(off, tile);
            off += tile;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t off, uint32_t tile)
    {
        LocalTensor<half> gate = gateQ.AllocTensor<half>();
        LocalTensor<half> up   = upQ.AllocTensor<half>();
        DataCopy(gate, xGm[off], tile);
        DataCopy(up,   xGm[n + off], tile);
        gateQ.EnQue(gate);
        upQ.EnQue(up);
    }

    __aicore__ inline void Compute(uint32_t tile)
    {
        LocalTensor<half> gate = gateQ.DeQue<half>();
        LocalTensor<half> up   = upQ.DeQue<half>();
        LocalTensor<float> gF = gFb.Get<float>();
        LocalTensor<float> uF = uFb.Get<float>();
        LocalTensor<float> nF = nFb.Get<float>();
        LocalTensor<float> eF = eFb.Get<float>();
        LocalTensor<float> sF = sFb.Get<float>();
        LocalTensor<float> oF = oFb.Get<float>();
        LocalTensor<half> out = outQ.AllocTensor<half>();

        Cast(gF, gate, RoundMode::CAST_NONE, tile);
        PipeBarrier<PIPE_V>();
        Cast(uF, up, RoundMode::CAST_NONE, tile);
        PipeBarrier<PIPE_V>();
        Muls(nF, gF, (float)-1.0, tile);
        PipeBarrier<PIPE_V>();
        Exp(eF, nF, tile);
        PipeBarrier<PIPE_V>();
        Adds(sF, eF, 1.0f, tile);
        PipeBarrier<PIPE_V>();
        Div(eF, gF, sF, tile);
        PipeBarrier<PIPE_V>();
        Mul(oF, eF, uF, tile);
        PipeBarrier<PIPE_V>();
        // 310P(dav-m200) float->half 仅支持 CAST_NONE/CAST_ODD，CAST_RINT 会断言失败且不写内存
        Cast(out, oF, RoundMode::CAST_NONE, tile);
        PipeBarrier<PIPE_V>();

        outQ.EnQue(out);
        gateQ.FreeTensor(gate);
        upQ.FreeTensor(up);
    }

    __aicore__ inline void CopyOut(uint32_t off, uint32_t tile)
    {
        LocalTensor<half> out = outQ.DeQue<half>();
        DataCopy(yGm[off], out, tile);
        outQ.FreeTensor(out);
    }

    TPipe pipe;
    GlobalTensor<half> xGm, yGm;
    TQue<QuePosition::VECIN, BUFNUM> gateQ, upQ;
    TQue<QuePosition::VECOUT, BUFNUM> outQ;
    TBuf<QuePosition::VECCALC> gFb, uFb, nFb, eFb, sFb, oFb;
    uint32_t n;
};
}

extern "C" __global__ __aicore__ void silu_and_mul(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SiluAndMulTilingData);
    GET_TILING_DATA(tilingData, tiling);
    SiluAndMulNS::KernelSiluAndMul op;
    op.Init(x, y, workspace, tiling);
    op.Process(tilingData.size);
}
