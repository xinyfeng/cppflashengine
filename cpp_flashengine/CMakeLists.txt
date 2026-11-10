# ============================================================================
# CMakeLists.txt — cpp_flashengine (昇腾 310P 自定义算子 C++ 转化工程)
#
# 目标：在开发板（Linux + CANN 9.0.T3 + torch-npu，对应 requirements-ascend.txt）
#       上把 ops/*.cpp 编译为可被 Python 引擎调用的自定义算子库。
#
# 本机（Windows，无 CANN/编译器）无法构建此工程；本文件仅为板上构建脚本。
#
# 关键环境变量（板端设置）：
#   ASCEND_HOME    CANN 安装根目录，默认 /usr/local/Ascend
#   TORCH_NPU_PATH torch-npu 的 C++ 头文件/库路径（与 torch 版本匹配）
#   PYTHON_EXECUTABLE  用于生成 Python 绑定的解释器
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(cpp_flashengine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- 1. 昇腾 CANN 路径 ----------------------------------------------------
if(NOT DEFINED ENV{ASCEND_HOME})
    set(ASCEND_HOME "/usr/local/Ascend" CACHE PATH "CANN 安装根目录")
else()
    set(ASCEND_HOME $ENV{ASCEND_HOME} CACHE PATH "CANN 安装根目录")
endif()

set(ASCEND_INCLUDE "${ASCEND_HOME}/include")
set(ASCEND_LIB     "${ASCEND_HOME}/lib64")
if(NOT EXISTS "${ASCEND_INCLUDE}")
    message(WARNING "未找到 CANN 头文件目录: ${ASCEND_INCLUDE}（请设置 ASCEND_HOME）")
endif()

include_directories(
    ${ASCEND_INCLUDE}
    ${ASCEND_INCLUDE}/ops
    ${ASCEND_INCLUDE}/ge
    ${CMAKE_CURRENT_SOURCE_DIR}/ops
)

# ---- 2. libtorch / torch-npu（自定义算子注册与 ATEN-NPU 接口）-------------
# 注意：必须使用与运行环境完全一致的 libtorch + torch-npu（见 requirements-ascend.txt:
#       torch==2.9.0 / torch-npu==2.9.0.post2.dev20260401）。版本错配会导致 ABI 崩溃。
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

# ---- 3. 自定义算子库 ------------------------------------------------------
# AI CPU 自定义算子：以动态库形式提供，供 ACL 运行时按算子名加载。
# 若改用 AICore / Ascend C 核，则还需链接 opp 编译产物并调整注册宏（见 README）。
add_library(cpp_flashengine SHARED
    ops/rms_norm.cpp
    ops/silu_and_mul.cpp
    ops/rope.cpp
)

target_include_directories(cpp_flashengine PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_directories(cpp_flashengine PRIVATE ${ASCEND_LIB})
target_link_libraries(cpp_flashengine
    PRIVATE
        ${TORCH_LIBRARIES}
        ascendcl      # 设备/运行时接口
        ops           # 算子 proto/注册
        ge_runner     # GE 图引擎（AI CPU 算子需链接）
)

# 输出到工程根，便于 Python 侧按固定文件名 import。
set_target_properties(cpp_flashengine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_NAME "cpp_flashengine"
)

# ---- 4. 本机单测（无 CANN 时也可在普通机器上验证 math_ref 正确性）--------
# 用法（任意 C++17 编译器）：
#   g++ -std=c++17 -DFLASH_CANN_SELFTEST cpp_flashengine/ops/rms_norm.cpp -o /tmp/t1 && /tmp/t1
#   g++ -std=c++17 -DFLASH_CANN_SELFTEST cpp_flashengine/ops/silu_and_mul.cpp -o /tmp/t2 && /tmp/t2
#   g++ -std=c++17 -DFLASH_CANN_SELFTEST cpp_flashengine/ops/rope.cpp -o /tmp/t3 && /tmp/t3
# （仅在 FLASH_CANN_SELFTEST 宏下编译 main()，不影响算子库构建。）
