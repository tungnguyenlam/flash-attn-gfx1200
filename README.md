# FlashAttention-2 for AMD RDNA4 (gfx1200)

Prebuilt [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) wheels for the
**AMD Radeon RX 9060 XT 16 GB (gfx1200)**, built with the Composable Kernel (CK) backend.

Upstream publishes CUDA wheels only. On ROCm, `pip install flash-attn` compiles ~1,800 CK
kernels from source, which takes hours and needs a matching ROCm development toolchain.
This repo pins the exact upstream commit, records the build flags, and ships the resulting
wheel so other gfx1200 owners can just install it.

> **Status: experimental.** RDNA4 is outside AMD's officially documented CK support matrix.
> The wheel is tested on the machine described below. It is not affiliated with Dao-AILab or AMD.

## Install

Download the wheel from the [latest release](../../releases/latest), then:

```bash
python -m pip install --no-deps flash_attn-2.8.4+rocm71torch2.13cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Use `--no-deps` so pip cannot replace your ROCm PyTorch with a CUDA build from PyPI.

Verify:

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
python verify.py          # correctness against fp32 SDPA + speed benchmark
```

## Compatibility

The wheel contains native code compiled for one exact Python/PyTorch ABI. It will not work
with a different PyTorch minor version or Python version.

| Component | Build / test value |
|---|---|
| GPU | AMD Radeon RX 9060 XT 16 GB, `gfx1200` (RDNA4) |
| OS | Linux 7.3 (CachyOS), x86_64 |
| Python | 3.12.14 |
| PyTorch | 2.13.0+rocm7.1, HIP 7.1.52802 |
| ROCm toolchain / headers | 7.2.5 (`/opt/rocm`) |
| flash-attn | 2.8.4, upstream commit [`10b74d9`](https://github.com/Dao-AILab/flash-attention/commit/10b74d98fb26113b014b33d947c8bbe4e83fbf7a) |
| Composable Kernel | [`c56c675`](https://github.com/ROCm/composable_kernel/commit/c56c6750d0fc54ed771d532cc92c316423449614) |
| C++ ABI | `cxx11abiTRUE` |
| Wheel tag | `cp312-cp312-linux_x86_64` |
| Backend | CK (`FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE`) |

## Use with Transformers

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2.5-350M",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda",
)
```

FlashAttention only supports fp16/bf16. Inputs must be on the GPU and contiguous.

Without `attn_implementation`, Transformers uses PyTorch SDPA. On current ROCm PyTorch,
SDPA's flash path on gfx1200 can also be enabled without this wheel:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python train.py
```

That is the zero-install option. This wheel exists for code that explicitly requires the
`flash_attn` package (Transformers' `flash_attention_2`, vLLM, ComfyUI, custom kernels, ...),
and for a native CK implementation on RDNA4.

## Build from source

Requirements: ROCm torch, ROCm development headers (`/opt/rocm/include`), `gfx1200` visible
to PyTorch, ~30 GB disk, and enough RAM for the parallel compile.

```bash
git clone https://github.com/<user>/flash-attn-gfx1200.git
cd flash-attn-gfx1200
./build.sh                     # MAX_JOBS=6 by default
```

The script checks out the pinned upstream commit and CK submodule, then runs
`python setup.py bdist_wheel` with:

```bash
BUILD_TARGET=rocm
FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE
FLASH_ATTENTION_SKIP_CK_BUILD=FALSE
GPU_ARCHS=gfx1200
```

On an i5-14600K with `MAX_JOBS=6`, the build takes about **X hours** and produces a
**Y GB** wheel from 1,837 generated CK kernels. `MAX_JOBS=1` is much slower but uses less
memory.

## Limitations

- **gfx1200 only.** The binary contains kernels for `gfx1200`; `GPU_ARCHS=gfx1201 ./build.sh`
  produces a different wheel for RX 9070-class cards.
- **Exact ABI.** Rebuild after changing Python, PyTorch, or the C++ ABI.
- **No sliding-window attention in the CK backend** (upstream limitation). Full causal and
  non-causal attention, GQA/MQA, varlen, dropout, and head dims up to 256 are supported.
- CK on RDNA4 is not an officially supported configuration. Correctness was checked against
  fp32 SDPA; see `verify.py`.

## Results

Measured with `verify.py` on the RX 9060 XT (bf16, `B=1`, `H=16`, `KV=8`, `D=64`, causal).
SDPA baseline is PyTorch's math backend:

| seq len | FA fwd | SDPA fwd | FA fwd+bwd | SDPA fwd+bwd |
|---:|---:|---:|---:|---:|
| 512 | TBD | TBD | TBD | TBD |
| 2048 | TBD | TBD | TBD | TBD |
| 8192 | TBD | TBD | TBD | TBD |

## License

The wheel is a build of [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention),
BSD 3-Clause. The upstream license is included in [LICENSE](LICENSE).
