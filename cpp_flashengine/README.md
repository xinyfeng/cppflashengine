# cpp_flashengine — 昇腾 310P 自定义算子“C++ 转化工程”

把原 Python 推理引擎（`flashengine/`，基于 nano-vllm）里三个热路径
**RMSNorm / SwiGLU(silu_and_mul) / RoPE** 用 C++ 自定义算子重写，供引擎在
Ascend 310P 上调用。

> 目标硬件：Ascend 310P（Atlas 300I Duo / 200I Pro）。板端开发容器 **vllm018**：
> CANN 9.0.T3 / 驱动 7.6.T7.0.B052 / SOC `ascend310p1`（该组合已实测可用）。
> 当前状态：**`silu_and_mul` 真实 AscendC 算子已在 310P 真机验证通过**
> ——`VERIFY silu_and_mul: n=2048 maxerr=0.000000 mismatches=0`；
> 一键构建/验证基座（`build_all.sh` + `verify_all.sh`）已就绪并跑通；
> rms_norm / rope / add_rms_norm 自定义算子工程尚未生成与验证。
>
> **派活看这里**：完整基座位置与剩余工作清单见 [`WORK_PLAN.md`](WORK_PLAN.md)；
> 310P API 避坑见 [`ascend_ops/310P_PORTING_NOTES.md`](ascend_ops/310P_PORTING_NOTES.md)。

---

## 1. 这个工程到底“转化”了什么

原引擎的热路径在 `flashengine/backends/base.py` 与 `ascend.py` 中，是纯 torch 实现；
310P 上由 `torch_npu` 下沉到 ACL 融合算子。`cpp_flashengine` 的“C++ 转化”是把这些
算子的数值计算用 C++ 重新实现，并通过自定义算子机制接回引擎。

| 算子 | 原 Python 入口 | C++ 实现位置 |
|---|---|---|
| RMSNorm | `BackendBase.rms_norm` | `ops/math_ref.h` + `ops/rms_norm.cpp` |
| 残差+RMSNorm | `BackendBase.add_rms_norm` | `ops/math_ref.h` + `ops/rms_norm.cpp` |
| SwiGLU | `BackendBase.silu_and_mul` | `ops/math_ref.h` + `ops/silu_and_mul.cpp` |
| RoPE | `BackendBase.rope` | `ops/math_ref.h` + `ops/rope.cpp` |

目录结构（真实算子工程在 `ascend_ops/`，与源码层 `ops/` 分开）：

```
cpp_flashengine/
├── CMakeLists.txt          # 源码层 CANN 构建脚本
├── ops/                    # 可移植 C++ 数学核（fp16 参考实现，可本机单测）
│   ├── math_ref.h
│   ├── rms_norm.cpp
│   ├── silu_and_mul.cpp
│   └── rope.cpp
├── ascend_ops/             # 板端真实 AscendC 自定义算子工程（310P 验证通过）
│   ├── 310P_PORTING_NOTES.md    # 避坑清单：环境/API/链接/排查手法
│   ├── build_all.sh             # 一键编出 gen/ 下所有算子 OPP 包
│   ├── verify_all.sh            # 一键编译并跑所有数值对照
│   ├── gen/silu_proj/           # msopgen 生成的 aclnn 工程（含核/tiling/op_host）
│   └── verify/
│       ├── verify_common.h      # 可复用验证框架（fp16 转换/设备内存/哨兵/对照）
│       └── verify_silu_and_mul.cpp  # 新算子照抄此模板，改 3 处即可
├── python/flash_cann.py    # 算子库 Python 封装，对齐 BackendBase 调用签名
├── integration/backend_cann.py  # 接入引擎的接缝（未联调）
└── tests/test_ops_parity.py     # C++ 算子 vs torch 参考的数值对照
```

---

## 2. 诚实状态（哪些确定正确，哪些待板上验证）

**已确定正确（不依赖 CANN，可本机验证）：**
- `ops/math_ref.h` 的数值实现：fp16 位级转换 + RMSNorm / SwiGLU / RoPE 公式，
  与原引擎 `x.float() -> 计算 -> .to(dtype)` 行为一致。
- 本机单测：任意 C++17 编译器即可验证 math_ref 正确性：
  ```bash
  g++ -std=c++17 -DFLASH_CANN_SELFTEST cpp_flashengine/ops/rms_norm.cpp -o /tmp/t1 && /tmp/t1
  g++ -std=c++17 -DFLASH_CANN_SELFTEST cpp_flashengine/ops/silu_and_mul.cpp -o /tmp/t2 && /tmp/t2
  g++ -std=c++17 -DFLASH_CANN_SELFTEST cpp_flashengine/ops/rope.cpp -o /tmp/t3 && /tmp/t3
  ```

**板端编译 / 验证现状（vllm018，CANN 9.0.T3）——已闭环：**
- `silu_and_mul`：`ascend_ops/gen/silu_proj/`（msopgen 生成的 aclnn 工程）在 310P
  编译通过，产出自定义 OPP 包（含 `libcust_opapi.so`、`aclnn_silu_and_mul`）。
  核源码 `op_kernel/silu_and_mul.cpp` 已修正全部 310P 兼容问题
  （`Neg`→`Muls`、`DataCopyPad`→3 参 `DataCopy`、vector 指令间 `PipeBarrier<PIPE_V>`、
  `float→half` 的 `CAST_RINT`→`CAST_NONE`）。
  **真机数值验证通过**：`VERIFY silu_and_mul: n=2048 maxerr=0.000000 mismatches=0 sentinel=0`。
- 环境：原 FE2C（CANN 9.1.0）与驱动 7.6 错配导致 361001；已切到 vllm018（9.0.T3），
  设备初始化正常，aclnn 调用 3 秒完成。详见 `ascend_ops/310P_PORTING_NOTES.md`。
- 已验证流程：`bash ascend_ops/build_all.sh` → `bash ascend_ops/verify_all.sh`
  （`SUMMARY built=1 failed=0` / `SUMMARY passed=1 failed=0`）。
- `rms_norm` / `rope` / `add_rms_norm`：CPU fp16 参考核在 `ops/`（可本机单测），
  自定义算子工程尚未生成 / 编译 / 验证。
- `python/flash_cann.py` 与 `integration/backend_cann.py`：接缝已写，未随引擎联调。

---

## 3. 板上构建与验证步骤（已实跑通过）

在开发板 **vllm018 容器**（CANN 9.0.T3 + 驱动 7.6，`/data` 已 bind mount）：

```bash
cd /data/nano_flashengine_fp16_1_0/cpp_flashengine/ascend_ops
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.T3

# 1) 一键编出 gen/ 下所有算子的自定义 OPP 包
bash build_all.sh          # 期望 SUMMARY built=N failed=0

# 2) 一键编译并跑所有算子的 CPU 参考数值对照
bash verify_all.sh         # 期望每个 VERIFY ... mismatches=0 sentinel=0，SUMMARY failed=0
```

**新增一个算子的流程**（P1 可并行开工，不再互相阻塞）：
1. 用 `msopgen` 在 `gen/<op>_proj/` 生成 aclnn 工程；
2. 写 `op_kernel/<op>.cpp`（照抄 `gen/silu_proj/` 的核结构，注意 310P 避坑清单）；
3. 复制 `verify/verify_silu_and_mul.cpp` 为 `verify/verify_<op>.cpp`，改 3 处：
   `@proj` / CPU 参考公式 / aclnn 两段式调用名；
4. 重跑 `build_all.sh` + `verify_all.sh`，新算子自动被纳入，判定标准是
   `mismatches=0` 且 `sentinel=0`。

**接入引擎（P2，尚未做）**：把后端切换为 `CannCppBackend`（在
`flashengine/backends/__init__.py` 的 `get_backend` 增 `cann_cpp` 分支），
原 `Ascend310PBackend` 的 NZ 布局 / NPUGraph / 分页 KV / 投机图全部保留。
端到端验收沿用原工程流程：`bash host_run.sh doctor/start/check-api`。

---

## 4. 重要现实：310P 上这些热路径本就已是 ACL 融合算子

`Ascend310PBackend` 原实现已通过 `torch_npu` 调用 ACL 融合算子
（`npu_rms_norm` / `npu_swiglu` / `npu_apply_rotary_pos_emb`）。`ascend_ops/` 下现在是
**真正的 AICore / Ascend C 核**（带 tiling 与 buffer 规划），但仍是单算子实现，因此：

- **这些自写算子不会比 ACL 融合更快**，主要价值是“C++ 转化”的工程落地、
  真机数值正确性验证，以及作为后续自定义融合的起点。
- 若要真正加速，方向是把多个算子合成一个 Ascend C 核（减少 GM 往返），
  并复用 NZ 布局；`backend_cann.py` 的 fallback 分支应改为直接回退到 ACL
  （性能更优），自定义算子仅在需要 ACL 未提供的融合组合时启用。
- 更大的 310P 开销在 Python 调度与图捕获（`AGENTS.md` 规定图内不做 `.item()`/
  `tolist()`/主机同步），优化重点在减少 Python 调用次数与复用 NPUGraph，而非替换
  单个已融合的算子。

---

## 5. 约束

- 仅新增 `cpp_flashengine/` 目录，不改写原引擎文件（除可选的 `__init__.py` 后端分支）。
- 新增注释 / docstring 用中文（遵循项目 `AGENTS.md`）。
- 保留来源版权与许可证；本工程为原引擎的 C++ 热路径补充，不替代原 Python 实现。

---

## 6. 工作分工（三人 / 两人）

> 总目标：**除 `torch.npu` 之外的热路径尽量转 C/C++**。务实路径——轻量算子
> （RMSNorm / AddRMSNorm / SwiGLU / RoPE / Linear）自写 AscendC；重融合
> （FlashAttention / PagedAttention / grouped-matmul / MoE）暂留 `torch_npu`；
> 引擎编排仍为 Python。以下按“已做完 / 未开始”拆分。

### 6.1 真实进度盘点（作为分工依据）

| 工作项 | 状态 |
|---|---|
| 数学参考核 `ops/math_ref.h`（rms_norm / silu_and_mul / rope, fp16） | 已写，可本机单测 |
| 轻量算子 `silu_and_mul`（真实 AscendC, msopgen aclnn） | ✅ **310P 编译 + 真机数值验证通过**（`mismatches=0`） |
| 轻量算子 `rms_norm` / `rope` | `ops/` 有参考核；**自定义算子工程未生成/未编译/未验证** |
| 轻量算子 `add_rms_norm` | 未开始 |
| 轻量算子 `linear`（GEMM） | 未开始（310P 建议走 ACL matmul 或 `torch_npu`，不自写核） |
| 多算子统一构建 `ascend_ops/build_all.sh` | ✅ 已写并板端实跑通过（`built=1 failed=0`） |
| 统一验证 `verify_all.sh` + `verify/verify_common.h` 模板 | ✅ 已写并板端实跑通过（`passed=1 failed=0`） |
| 310P 避坑清单 `310P_PORTING_NOTES.md` | ✅ 已写（环境/API/链接/排查手法） |
| Python 封装 `flash_cann.py` + `backend_cann.py` 接缝 | 已写（接缝级，未联调） |
| 数值对照 `tests/test_ops_parity.py` | 已写，未板端跑 |
| 重融合 `torch_npu` 适配（FA / PagedAttn / grouped-mm / MoE） | 未开始 |
| 引擎编排（模型 / scheduler / KV cache 接线） | 接缝已有，整体未接线 |
| **并行基座（环境 + 构建 + 验证模板 + 参考算子）** | ✅ **已就绪，三人可并行开工** |

### 6.2 三人分工（基座就绪后的并行版本）

| 角色 | 负责内容 | 关键交付 | 预估 |
|---|---|---|---|
| **P1 算子内核开发** | 照 `silu_proj` 模板做 `rms_norm`/`rope`/`add_rms_norm`：msopgen 生成→写核→`build_all.sh`→`verify_all.sh`；`linear` 定方案走 ACL matmul；维护 `math_ref.h` | 4 个算子工程均在 310P 编译+验证通过 | 4~6 人天 |
| **P2 引擎与集成** | 完善 `backend_cann.py`，把自定义算子 + `torch_npu` 重融合接进模型 forward；引擎编排（scheduler、KV cache、端到端链路）；跑通 `test_ops_parity.py` 联调 | 引擎可切 `cann_cpp` 后端端到端推理 | 4~5 人天 |
| **P3 构建 / 环境 / 重融合适配** | 维护 `build_all.sh` / `verify_all.sh` / 避坑清单；**`torch_npu` 重融合 ACL 级适配（FA / PagedAttn / grouped-mm / MoE）**；板端环境固化；维护 README | 重融合接入 + 持续可复现验证 | 4~6 人天（重融合占 3~5） |

**三人并行 ≈ 1.5~2 周**（此前估 2~3 周，因 blocker 已解决且基座复用，估时下调）；
**两人并行 ≈ 2.5~3 周**。关键路径是 P3 的重融合适配，P1/P2 不再被环境问题阻塞。

### 6.3 两人分工

- **A = P1 + P3 的构建/环境部分**：算子内核 + 统一构建 + 验证环境。
- **B = P2 + P3 的重融合适配**：引擎集成 + 重融合接入 + 联调。
- 基座已就绪，两人**可同时开工**，无需再等环境解锁。

### 6.4 关键依赖 / 风险（全员须知）

- **原 P0 blocker 已解决**：FE2C（CANN 9.1.0）+ 驱动 7.6 错配导致 361001；
  已切到 **vllm018（CANN 9.0.T3）**，设备初始化正常，aclnn 3 秒完成，
  `silu_and_mul` 真机 parity 归零。`source set_env.sh` 不是根因，别再排查它。
- **310P API 坑已全部记录在 `ascend_ops/310P_PORTING_NOTES.md`**，尤其
  `float→half` 的 `Cast` 只能用 `CAST_NONE`/`CAST_ODD`（`CAST_RINT` 会静默写 0）。
  P1 写新算子前先读该文件，可省掉本轮花掉的排查时间。
- **范围口径未定**：本分工基于“务实路径”——重融合仍留 `torch_npu`。若严格按
  “除 `torch.npu` 之外全部转 C/C++”，重融合需自写 AscendC（PagedAttention 尤其重），
  P3 估时要从 3~5 人天上调到数人周，总工期约 1~2 个月。开工前请确认口径。
- 自写轻量算子**不会比 ACL 融合更快**（见第 4 节），价值在 C++ 落地与正确性验证。
- 图内禁 `.item()`/`tolist()`/动态形状（遵循 `AGENTS.md`）；不升级驱动/CANN/PyTorch。
