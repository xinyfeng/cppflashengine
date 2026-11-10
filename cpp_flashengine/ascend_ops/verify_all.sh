#!/bin/bash
# 多算子统一验证：编译 verify/verify_*.cpp，逐个跑 CPU 参考对照。
# 板上运行（vllm018 / CANN 9.0.T3 / 驱动 7.6）；判定标准 mismatches=0 且 sentinel=0。
# @proj 注释指定该验证程序对应的 gen/ 算子工程。
# 新增注释一律使用中文（遵循项目 AGENTS.md）。
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.T3}"
DRV="${DRV_LIB_PATH:-/usr/local/Ascend/driver/lib64}"
SET_ENV="$(dirname "$CANN")/ascend-toolkit/latest/set_env.sh"

if [ -f "$SET_ENV" ]; then
    # set_env.sh 会引用未定义变量，与本脚本的 nounset 冲突，source 期间临时关闭
    set +u
    # shellcheck disable=SC1090
    source "$SET_ENV"
    set -u
fi
export ASCEND_HOME_PATH="$CANN"

FAILED=0
PASSED=0
for src in "$ROOT"/verify/verify_*.cpp; do
    [ -f "$src" ] || continue
    op="$(basename "$src" .cpp | sed 's/^verify_//')"
    proj="$(grep -m1 '^// @proj' "$src" | awk '{print $3}')"
    [ -n "$proj" ] || proj="$(find "$ROOT/gen" -maxdepth 1 -type d -name "*${op}*_proj" | head -1 | xargs -r basename)"
    if [ -z "$proj" ]; then
        echo "[$op] 跳过：未找到对应算子工程（请在源文件加 @proj 注释）"
        continue
    fi

    pkg="$(find "$ROOT/gen/$proj/build_out" -path '*/packages/vendors/customize' -type d 2>/dev/null | head -1)"
    if [ -z "$pkg" ]; then
        echo "[$op] 跳过：未找到 OPP 包，请先运行 build_all.sh"
        FAILED=$((FAILED + 1))
        continue
    fi

    bin="/tmp/verify_${op}"
    # -lascend_hal 是板端链接 libascendcl 的依赖，缺了会报 drvHdc* 未定义。
    g++ -std=c++17 "$src" -o "$bin" \
        -I "$ROOT/verify" \
        -I "$CANN/include" -I "$pkg/op_api/include" \
        -L "$CANN/lib64" -lascendcl -lnnopbase \
        -L "$pkg/op_api/lib" -lcust_opapi \
        -L "$DRV" -lascend_hal \
        -Wl,-rpath,"$pkg/op_api/lib" -Wl,-rpath,"$CANN/lib64" -Wl,-rpath,"$DRV" 2>"/tmp/verify_${op}.compile.log"
    if [ $? -ne 0 ]; then
        echo "[$op] 编译失败，日志 /tmp/verify_${op}.compile.log"
        tail -10 "/tmp/verify_${op}.compile.log"
        FAILED=$((FAILED + 1))
        continue
    fi

    # 每个算子单独设置 OPP 路径，避免多算子包互相覆盖导致找不到 aclnn 符号。
    export ASCEND_CUSTOM_OPP_PATH="$pkg"
    export LD_LIBRARY_PATH="$pkg/op_api/lib:$pkg/op_proto/lib/linux/aarch64:$pkg/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$DRV:${LD_LIBRARY_PATH:-}"
    "$bin"
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$op] PASS"
        PASSED=$((PASSED + 1))
    else
        echo "[$op] FAIL rc=$rc"
        FAILED=$((FAILED + 1))
    fi
done

echo "SUMMARY passed=$PASSED failed=$FAILED"
exit $FAILED
