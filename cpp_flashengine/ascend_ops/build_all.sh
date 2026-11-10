#!/bin/bash
# 多算子统一构建：一次编出 gen/ 下所有自定义算子 OPP 包。
# 板上运行（vllm018 / CANN 9.0.T3 / 驱动 7.6）；新增算子只需在 gen/ 放一个 msopgen 工程。
# 新增注释一律使用中文（遵循项目 AGENTS.md）。
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.T3}"
SET_ENV="$(dirname "$CANN")/ascend-toolkit/latest/set_env.sh"
[ -f "$SET_ENV" ] || SET_ENV="$CANN/../ascend-toolkit/latest/set_env.sh"

if [ -f "$SET_ENV" ]; then
    # set_env.sh 会引用未定义变量，与本脚本的 nounset 冲突，source 期间临时关闭
    set +u
    # shellcheck disable=SC1090
    source "$SET_ENV"
    set -u
    export ASCEND_HOME_PATH="$CANN"
else
    echo "WARN: 未找到 set_env.sh，依赖容器已注入的 CANN 环境变量"
fi
export ASCEND_HOME_PATH="$CANN"

echo "CANN=$ASCEND_HOME_PATH"
echo "SET_ENV=$SET_ENV"

FAILED=0
BUILT=0
for proj in "$ROOT"/gen/*_proj; do
    [ -d "$proj" ] || continue
    name="$(basename "$proj")"
    preset="$proj/CMakePresets.json"
    if [ ! -f "$preset" ]; then
        echo "[$name] 跳过：缺少 CMakePresets.json"
        continue
    fi

    # msopgen 生成的预设里写死了生成时的 CANN 版本，按当前环境纠正后再编译。
    sed -i "s#/usr/local/Ascend/cann-[0-9A-Za-z.]*#${CANN}#g" "$preset"
    echo "[$name] preset -> $CANN"

    log="/tmp/build_${name}.log"
    ( cd "$proj" && bash build.sh ) > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$name] 构建失败 rc=$rc，日志 $log"
        tail -15 "$log"
        FAILED=$((FAILED + 1))
        continue
    fi

    pkg="$(find "$proj/build_out" -path '*/packages/vendors/customize' -type d 2>/dev/null | head -1)"
    if [ -z "$pkg" ]; then
        echo "[$name] 构建完成但未找到 vendors/customize 目录"
        FAILED=$((FAILED + 1))
        continue
    fi
    echo "[$name] 构建成功 pkg=$pkg"
    BUILT=$((BUILT + 1))
done

echo "SUMMARY built=$BUILT failed=$FAILED"
exit $FAILED
