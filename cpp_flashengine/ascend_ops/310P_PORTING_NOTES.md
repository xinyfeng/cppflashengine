# 310P / CANN 9.0.T3 板端开发避坑清单

本文件记录 `silu_and_mul` 真机验证过程中实际踩过的坑，供后续算子
（rms_norm / rope / add_rms_norm / linear）直接复用，避免重复排查。
环境：vllm018 容器，CANN 9.0.T3，驱动 7.6.T7.0.B052，SOC `ascend310p1`。

## 1. 环境组合（最容易浪费时间的一项）

| 组合 | 结果 |
|---|---|
| FE2C：CANN **9.1.0** + 驱动 7.6 | ❌ `aclInit` 后设备上下文为 null，`GetWorkspaceSize` 返回 **361001**，握手约 100s 后失败 |
| vllm018：CANN **9.0.T3** + 驱动 7.6 | ✅ 正常，aclnn 调用 3 秒完成，同容器正跑着 Qwen3-8B 在线推理 |

- `361001` 不是缺 `source set_env.sh`：试过 source 之后重跑，错误不变。
- `torch.npu.is_available()` 返回 `True` / `device_count()==1` **不能证明** ACL 计算上下文可用。
- 结论：**驱动 7.6 要配 CANN 9.0.T3**。不要为验证去升级驱动/CANN（遵循 `AGENTS.md`），
  换匹配版本的容器即可。
- 板端开发统一在 **vllm018**；`/data` 已 bind mount，源码在两边都可见。

## 2. 构建与链接

- `CMakePresets.json` 里 `ASCEND_CANN_PACKAGE_PATH` 被 msopgen 写死成生成时的版本。
  换环境必须用 `build_all.sh` 纠正为当前 `$ASCEND_HOME_PATH`，否则编译找不到头文件。
- 链接验证程序需要 `-L /usr/local/Ascend/driver/lib64 -lascend_hal`，
  否则报一堆 `drvHdc* / halGetDeviceInfo` 未定义（由 `libascend_dump.so` 间接依赖）。
- 需要 `ASCEND_CUSTOM_OPP_PATH` 指向 `.../packages/vendors/customize`（**不是** `ASCEND_OPP_PATH`），
  并把 `op_api/lib`、`op_proto/lib/linux/aarch64`、`op_tiling/lib/linux/aarch64` 加进 `LD_LIBRARY_PATH`。
- 产物包名随系统变（openEuler 下是 `custom_opp_openEuler_aarch64.run`），
  脚本用 `find -path '*/packages/vendors/customize'` 定位，不要写死路径。

## 3. aclnn 调用（主机侧）

- `aclCreateTensor` **最后一个参数是设备内存指针**，x/y 都必须传 `xDev`/`yDev`。
  传 `nullptr` 会让算子使用自有内存，主机端读回的是未初始化数据
  （曾出现 `maxerr=3.35 / 1912 mismatches` 的假象）。
- workspace 必须等 `aclrtSynchronizeStream` 之后再释放（异步执行不能提前 free）。
- 用 `aclrtMemset` 给输出预填哨兵（如 `0x3C3C`），可快速区分
  「核没写内存」和「核写错了」——见 `verify_common.h` 的 `sentinel` 统计。

## 4. AscendC 核（dav-m200 / 310P）API 限制

| 限制 | 正确做法 |
|---|---|
| `Neg` 未定义 | 用 `Muls(dst, src, (float)-1.0, count)` |
| `DataCopyPad` 不支持 | 用 3 参 `DataCopy(LocalTensor, GlobalTensor, count)` |
| `DataCopy(GM↔Local)` 要求 32B 对齐 | half 下 `count` 必须是 **16 的倍数**；验证尺寸取 `n=2048` |
| `float→half` 的 `CAST_RINT` **不支持** | 用 `Cast(out, oF, RoundMode::CAST_NONE, tile)` |

最后一条是本次数值全错的真正根因。`dav_m200/kernel_operator_vec_vconv_impl.h`
的 `CastIntrinsicsImpl(__ubuf__ half*, __ubuf__ float*, ...)` 里，
`CAST_RINT / CAST_FLOOR / CAST_CEIL / CAST_ROUND / CAST_TRUNC` 全部是
`ASCENDC_ASSERT(false, ...)`；NPU 模式下断言不产生可见的失败，
表现为**输出全 0 或未定义值**。float→half 只有 `CAST_NONE` 与 `CAST_ODD` 可用。

另外：**vector 指令之间要插 `PipeBarrier<PIPE_V>()`**，且避免 in-place 覆写同一 buffer
（例如 `Adds(eF, eF, 1.0f, tile)` 改为写 `sF`）。910B 有自动 RAW 依赖检测，310P 没有。
`TILE` 也别开太大（现用 2048），6 个 float TBuf + 3 个 half TQue 的 UB 占用要留余量。

## 5. 排查手法（可复用）

写一个「诊断核」：输出默认直接 `DataCopy(out, gate, tile)` 拷贝输入，
前 8 个元素用 `SetValue` 写入各计算阶段的中间值（`gF`/`uF`/`nF`/`eF`/`sF`/`oF`），
主机端打印出来逐段比对。一次 rebuild 就能定位坏在哪一步，比盲改快得多。
本次用它确认了「搬运正确、前 7 步计算正确、只有最后的 Cast 全 0」。

## 6. 当前已验证结果

```
VERIFY silu_and_mul: n=2048 maxerr=0.000000 mismatches=0
```

判定标准：`mismatches == 0` 且 `sentinel == 0`，fp16 容差 `2e-2`。
