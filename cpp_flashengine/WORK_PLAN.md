# 完整基座位置与剩余工作清单

> 用途：供分工派活时直接引用。本文列出「已验证可用的基座在哪」以及
> 「还剩哪些具体工作」，每项都标明涉及文件、验收标准与预估。
> 状态截至：`silu_and_mul` 在 310P 真机 parity 归零、一键构建/验证闭环跑通之后。

---

## 1. 完整基座在哪

### 1.1 两个位置（内容一致，板端为可运行环境）

| | 路径 |
|---|---|
| **本地（Windows，源码主副本）** | `D:\OneDrive\Desktop\631\Qwen3\C++\nano_flashengine_fp16_1_0\cpp_flashengine\` |
| **板端（Linux，真机验证用）** | `192.168.16.118` → `vllm018` 容器 → `/data/nano_flashengine_fp16_1_0/cpp_flashengine/` |

- 容器 **vllm018**：CANN 9.0.T3 / 驱动 7.6.T7.0.B052 / SOC `ascend310p1`。
  `/data` 已 bind mount，两边改同一份文件。
- **不要用 FE2C**：其 CANN 9.1.0 与驱动 7.6 错配，`aclInit` 后设备上下文为 null，
  `GetWorkspaceSize` 返回 361001。详见 `ascend_ops/310P_PORTING_NOTES.md` 第 1 节。
- scp 被封，两边同步用 `tar + base64` 通道（本次已多次使用）。

### 1.2 权威基座文件清单（这些是「已验证可用」的）

```
cpp_flashengine/
├── README.md                        # 总说明 + 第 6 节三人分工
├── CMakeLists.txt                   # 源码层构建（仅作参考，见 §3.4）
├── ops/                             # 可移植 fp16 数学参考核（可本机单测）
│   ├── math_ref.h
│   ├── rms_norm.cpp
│   ├── silu_and_mul.cpp
│   └── rope.cpp
├── ascend_ops/                      # ★ 真机验证过的基座主体
│   ├── build_all.sh                 # 一键编出 gen/ 下所有算子 OPP 包
│   ├── verify_all.sh                # 一键编译并跑所有数值对照
│   ├── 310P_PORTING_NOTES.md        # 310P 避坑清单（开工前必读）
│   ├── gen/silu_proj/               # msopgen 生成的 aclnn 算子工程
│   │   ├── CMakeLists.txt / CMakePresets.json / build.sh
│   │   ├── op_kernel/silu_and_mul.cpp        # 真实 AscendC 核（已修全部 310P 坑）
│   │   ├── op_kernel/silu_and_mul_tiling.h
│   │   └── op_host/silu_and_mul.cpp          # tiling / InferShape / OpDef
│   └── verify/
│       ├── verify_common.h                  # 共享验证框架（fp16/设备内存/哨兵/对照）
│       └── verify_silu_and_mul.cpp          # 新算子照抄模板，改 3 处即可
├── python/flash_cann.py             # Python 封装（未打通，见 §3.2）
├── integration/backend_cann.py      # 引擎接缝（未打通，见 §3.2）
└── tests/test_ops_parity.py         # torch 参考数值对照（未板端跑通）
```

板端编译产物（不在 git 主副本概念内，运行时需要）：
`ascend_ops/gen/silu_proj/build_out/.../custom_opp_openEuler_aarch64.run/packages/vendors/customize`
（`libcust_opapi.so`、`aclnn_silu_and_mul.h`、tiling/proto 库都在此）。

### 1.3 板端历史遗留目录（**不是基座，别用**）

板上 `/data/.../cpp_flashengine/ascend_ops/` 下还残留早期探索产物，本地没有：
`silu_and_mul.json`、`silu_and_mul_aclnn`、`silu_and_mul_project`、
`silu_and_mul_pytorch`（284K）、`verify_silu`（16K）；
以及 `gen/silu_proj/` 里的 `verify_diag*`、`verify_test`、`verify_test_90` 二进制。
这些是调试/废弃尝试，**不要作为模板复制**。权威模板只有 `gen/silu_proj/` 的源码和
`ascend_ops/verify/` 下的两个文件。建议 P3 顺手清理（见 §3.3）。

### 1.4 已验证状态（证据）

```bash
cd /data/nano_flashengine_fp16_1_0/cpp_flashengine/ascend_ops
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.T3
bash build_all.sh    # SUMMARY built=1 failed=0
bash verify_all.sh   # VERIFY silu_and_mul: n=2048 maxerr=0.000000 mismatches=0 sentinel=0
                     # SUMMARY passed=1 failed=0
```

判定标准：`mismatches == 0` 且 `sentinel == 0`，fp16 容差 `2e-2`。
设备初始化正常（aclnn 调用 3 秒完成），361001 已消失。

---

## 2. 剩余工作总览

| 编号 | 工作 | 归属 | 预估 | 可并行 |
|---|---|---|---|---|
| W1 | 补齐 4 个轻量算子（rms_norm / rope / add_rms_norm / linear） | P1 | 4~6 人天 | ✅ |
| W2 | 打通 Python→aclnn 调用桥（**当前隐藏阻断**） | P2 | 1~1.5 人天 | ✅ |
| W3 | `backend_cann.py` 接线 + 后端注册 + 端到端联调 | P2 | 3~4 人天 | 依赖 W1/W2 |
| W4 | 重融合 `torch_npu` 适配（FA/PagedAttn/grouped-mm/MoE） | P3 | 3~5 人天 | ✅ |
| W5 | 构建/验证脚本增强 + 环境固化 + 遗留清理 | P3 | 1~1.5 人天 | ✅ |
| W6 | 范围口径决策 + 性能基线 | 全员 | 0.5 人天 | 开工前 |

**三人并行 ≈ 1.5~2 周**（若 W2 选择复杂方案会上浮到 2~2.5 周）；两人并行 ≈ 2.5~3 周。
关键路径是 W4（重融合）+ W3（联调）。

---

## 3. 各项工作明细

### 3.1 W1 · P1 算子内核开发（4~6 人天）

**统一流程**（每个算子都照 `silu_proj` 走一遍，不再依赖 P3 排环境）：

1. `msopgen` 在 `ascend_ops/gen/<op>_proj/` 生成 aclnn 工程；
2. 写 `op_kernel/<op>.cpp`（照抄 `silu_proj` 核结构）+ `op_host/<op>.cpp`（tiling/InferShape/OpDef）；
3. 复制 `ascend_ops/verify/verify_silu_and_mul.cpp` → `verify_<op>.cpp`，改 3 处：
   `@proj` 注释、CPU 参考公式、aclnn 两段式调用名；
4. `bash build_all.sh && bash verify_all.sh`，新算子自动纳入，验收 `mismatches=0 sentinel=0`。

| 子项 | 具体工作 | 涉及文件 | 预估 | 注意 |
|---|---|---|---|---|
| W1.1 `rms_norm` | 归一化：x / sqrt(mean(x²)+eps) × weight | `gen/rms_norm_proj/`、`verify/verify_rms_norm.cpp` | 1~1.5d | 需 310P 的 ReduceSum/Sqrt 支持；CPU 参考见 `ops/math_ref.h` |
| W1.2 `rope` | 按 positions 取 cos/sin，对 q/k 做旋转 | `gen/rope_proj/`、`verify/verify_rope.cpp` | 1.5~2d | 最复杂：索引/多输入/双输出；避免动态 shape（AGENTS.md） |
| W1.3 `add_rms_norm` | residual 相加 + RMSNorm，双输出 | `gen/add_rms_norm_proj/`、`verify/verify_add_rms_norm.cpp` | 1~1.5d | 输出含更新后的 residual |
| W1.4 `linear` | 定方案并写接缝 | 方案文档 + `verify/verify_linear.cpp` | 0.5~1d | 310P 建议走 ACL matmul 或 `torch_npu`，**不自写 GEMM 核** |
| W1.5 尺寸覆盖 | 当前只测 `n=2048` 单一尺寸 | `verify/verify_common.h`、各 verify 文件 | 0.5d | 需覆盖 `>TILE` 多 tile；明确 16 元素对齐约束或补齐非对齐处理 |

**验收**：`SUMMARY built=5 failed=0` / `SUMMARY passed=5 failed=0`（linear 视方案可能单独计）。
**开工前必读**：`ascend_ops/310P_PORTING_NOTES.md`，尤其 `CAST_RINT`、`PipeBarrier<PIPE_V>`、
`Neg`/`DataCopyPad` 三条，可省掉本轮花掉的大量排查时间。

### 3.2 W2 · P2 打通 Python→aclnn 桥（1~1.5 人天，**当前隐藏阻断**）

**这是必须先定的架构决策**。现状两条路线对不上：

- 板端验证通过的路线是 **aclnn 自定义算子**：C++ 直接调 `libcust_opapi.so` 里的
  `aclnnSiluAndMulGetWorkspaceSize` / `aclnnSiluAndMul`。
- `python/flash_cann.py` 却假设算子注册为 `torch.ops.cpp_flashengine.*`，
  并 `import cpp_flashengine` 触发注册；`tests/test_ops_parity.py` 也沿用这个假设。
- 源码层 `ops/*.cpp` 是 **AI CPU 模板**：`REGISTER_KERNEL` 宏被注释掉，
  `Compute()` 里计算逻辑是空壳（`(void)x; (void)y; return 0;`），
  `CMakeLists.txt` 也按 libtorch/AI CPU 找 `find_package(Torch)`。
  **这条路线从未编译、从未跑通**，与 aclnn 产物无关。

也就是说，Python 侧目前没有任何能调到已验证算子的通路。需二选一：

| 方案 | 做法 | 代价 |
|---|---|---|
| **A（建议）** | 用 `pybind11` 或 `ctypes` 封装 `libcust_opapi.so`，在 `flash_cann.py` 里把 torch tensor 的 `data_ptr()` 传下去，直接调 aclnn | 需处理设备指针/stream/生命周期；约 1~1.5d |
| B | 走 `torch_npu` 自定义算子注册机制，把 OPP 包注册成 `torch.ops.*` | 依赖 torch_npu 版本细节，风险更高 |

**涉及文件**：`python/flash_cann.py`（重写）、`tests/test_ops_parity.py`（对齐新路径）、
可能新增 `python/_aclnn_bindings.cpp` 或 `python/aclnn_wrapper.py`。
**验收**：`python/flash_cann.py` 能在 vllm018 里对真实 npu tensor 调通 `silu_and_mul`，
输出与 `verify_all.sh` 的 CPU 参考一致。

### 3.3 W3 · P2 引擎接线与端到端联调（3~4 人天）

| 子项 | 具体工作 | 涉及文件 | 预估 |
|---|---|---|---|
| W3.1 | `CannCppBackend` 覆盖四个热路径，去掉占位 `try/except` fallback，改真实调用 | `integration/backend_cann.py` | 0.5~1d |
| W3.2 | 后端注册：`get_backend` 增加 `cann_cpp` 分支 | `flashengine/backends/__init__.py` | 0.5d |
| W3.3 | 保留 310P 特性：NZ 布局 / NPUGraph / 分页 KV / 投机图 | 引擎侧多处 | 1~1.5d |
| W3.4 | 板端跑通 `tests/test_ops_parity.py`（当前依赖旧 import 路径） | `tests/test_ops_parity.py` | 0.5d |
| W3.5 | 端到端验收 `host_run.sh doctor / start / check-api` | 原工程流程 | 0.5~1d |

> ⚠️ **W3.2 / W3.3 / W3.5 会改到 `cpp_flashengine/` 之外**（`flashengine/` 与原工程脚本）。
> 按约定需先向项目负责人取得授权，本轮未动。

**验收**：引擎切到 `cann_cpp` 后端能端到端推理，且输出与原 `Ascend310PBackend` 在容差内一致。

### 3.4 W4 · P3 重融合 `torch_npu` 适配（3~5 人天，关键路径）

FlashAttention / PagedAttention / grouped-matmul / MoE 的 ACL 级接入。
本分工基于「务实路径」——这四块仍留 `torch_npu`，只做适配与调用固化，不自写 AscendC 核。

| 子项 | 预估 |
|---|---|
| FlashAttention（prefill）接入 | 0.5~1d |
| PagedAttention（decode）接入 | 1~1.5d |
| grouped-matmul / MoE dispatch | 1~1.5d |
| 与引擎调度/KV cache 对齐 | 0.5~1d |

**范围口径未定（W6）**：若严格按最初「除 torch.npu 之外全部转 C/C++」，
这四块需自写 AscendC（PagedAttention 尤其重），W4 从 3~5 人天涨到**数人周**，
总工期约 **1~2 个月**。务必在派活前确认口径。

### 3.5 W5 · P3 构建/环境增强与清理（1~1.5 人天）

| 子项 | 具体工作 | 涉及文件 |
|---|---|---|
| W5.1 | `build_all.sh` 增强：增量构建、失败不中断其他算子、日志归档 | `ascend_ops/build_all.sh` |
| W5.2 | `verify_all.sh` 增强：多尺寸参数化、耗时统计、失败明细 | `ascend_ops/verify_all.sh` |
| W5.3 | 环境固化：把 vllm018 的容器/版本/启动参数写成文档，避免再踩 FE2C 错配 | `ascend_ops/310P_PORTING_NOTES.md` |
| W5.4 | 清理 §1.3 的板端历史遗留目录与残留二进制 | 板端 `ascend_ops/` |
| W5.5 | 明确源码层定位：建议把 `ops/*.cpp` + `CMakeLists.txt` 定位为「CPU 参考 + 本机单测」，与板端 aclnn 路线解耦，避免两套构建混淆 | `README.md` / `CMakeLists.txt` 注释 |

### 3.6 W6 · 全员横向（0.5 人天，开工前）

1. **定范围口径**：务实路径（重融合留 torch_npu）还是严格全转 C —— 决定工期是 2 周还是 1~2 个月。
2. **定 W2 架构方案**：pybind/ctypes 直调 aclnn，还是 torch_npu 注册。
3. **取得目录外改动授权**：`flashengine/` 与 `host_run.sh` 相关文件。
4. **性能基线**：记录自写算子 vs ACL 融合的 latency。README 第 4 节已诚实说明
   「自写轻量算子不会比 ACL 融合更快」，需用数据支撑，避免误判优化收益。

---

## 4. 分派建议（三人并行，基座就绪）

- **P1** 立刻开工 W1，先做 `rms_norm`（最简单，验证模板确实可复用），再 `add_rms_norm` → `rope` → `linear`。
  全程只需 `build_all.sh` / `verify_all.sh`，不需要等任何人。
- **P2** 立刻开工 W2（定方案 + 打通桥），随后 W3；W3.2 起的目录外改动等授权。
- **P3** 立刻开工 W4 与 W5，关键路径是 W4。

三人的共同前置只有 W6 的三个决策，建议在派活会上一次定完。

---

## 5. 约束（遵循项目 `AGENTS.md`）

- FP16-only，禁量化；不引入 vllm_ascend / torch.compile / TorchAir / MindIE。
- 不升级驱动 / CANN / PyTorch。
- 图内禁 `.item()` / `tolist()` / 动态形状。
- 新增注释与文档使用中文。
- 自写轻量算子不会比 ACL 融合更快，价值在 C++ 落地与正确性验证。
