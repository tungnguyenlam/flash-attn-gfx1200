#!/usr/bin/env bash
# Build a FlashAttention-2 wheel for AMD RDNA4 (gfx1200 / RX 9060 XT) with the CK backend.
#
# Requirements:
#   - Linux, Python 3.12, PyTorch built for ROCm (tested with 2.13.0+rocm7.1)
#   - ROCm development headers/toolchain in /opt/rocm (tested with ROCm 7.2.5)
#   - A working gfx1200 GPU visible to PyTorch
#   - ~30 GB free disk and ~24 GB RAM (MAX_JOBS=6)
#
# Usage:
#   ./build.sh                # builds for gfx1200
#   GPU_ARCHS=gfx1201 ./build.sh
#
# The resulting wheel is written to dist/.
set -euo pipefail

cd "$(dirname "$0")"

GPU_ARCHS="${GPU_ARCHS:-gfx1200}"
MAX_JOBS="${MAX_JOBS:-6}"
FA_REPO="${FA_REPO:-https://github.com/Dao-AILab/flash-attention.git}"
# Pinned upstream commit used for the published wheel (2026-10-22).
FA_COMMIT="${FA_COMMIT:-10b74d98fb26113b014b33d947c8bbe4e83fbf7a}"
WORK_DIR="${WORK_DIR:-$PWD/.work}"
SRC_DIR="$WORK_DIR/flash-attention"

mkdir -p "$WORK_DIR" dist

if [ ! -d "$SRC_DIR/.git" ]; then
    git clone "$FA_REPO" "$SRC_DIR"
fi
git -C "$SRC_DIR" fetch --depth 1 origin "$FA_COMMIT"
git -C "$SRC_DIR" checkout --detach "$FA_COMMIT"
git -C "$SRC_DIR" submodule update --init --depth 1 csrc/composable_kernel

python -m pip install packaging psutil ninja wheel setuptools

export BUILD_TARGET=rocm
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE
export FLASH_ATTENTION_SKIP_CK_BUILD=FALSE
export GPU_ARCHS
export MAX_JOBS
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_CLANG_PATH="${HIP_CLANG_PATH:-$ROCM_PATH/llvm/bin}"
export PATH="$HIP_CLANG_PATH:$PATH"

cd "$SRC_DIR"
python setup.py bdist_wheel

cp -v "$SRC_DIR"/dist/flash_attn-*.whl "$OLDPWD/dist/"
cd "$OLDPWD"
sha256sum dist/flash_attn-*.whl | tee dist/SHA256SUMS
