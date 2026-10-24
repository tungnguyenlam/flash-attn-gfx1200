# FlashAttention-2 for AMD RDNA4 (gfx1200)

Prebuilt [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) wheel for the
**AMD Radeon RX 9060 XT 16 GB (gfx1200)**, built with the Composable Kernel (CK) backend.

Upstream publishes CUDA wheels only. On ROCm, `pip install flash-attn` compiles ~1,800 CK
kernels from source, which takes hours and needs a matching ROCm development toolchain.
This repo pins the exact upstream commit, records the build flags, and ships the resulting
wheel so other gfx1200 owners can just install it.

> **Status: experimental.** RDNA4 is outside AMD's officially documented CK support matrix.
> The wheel is tested on the machine described below. It is not affiliated with Dao-AILab or AMD.

## Install

```bash
python -m pip install --no-deps \
  https://github.com/tungnguyenlam/flash-attn-gfx1200/releases/latest/download/flash_attn-2.8.4-cp312-cp312-linux_x86_64.whl
python -m pip install einops
```

`--no-deps` keeps pip from replacing your ROCm PyTorch with a CUDA build from PyPI.
`einops` is flash-attn's only other runtime dependency (used by `flash_attn.bert_padding`,
which Transformers imports).

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
| Wheel size | 41 MB (188 MB extension, gfx1200 kernels only) |

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

## Results

All measurements on the RX 9060 XT 16 GB, bf16, `B=1`, `H=16`, `KV=8`, `D=64`, causal.
`verify.py`, 30 timed iterations after 10 warmup.

### Against PyTorch's math fallback

| seq | FA fwd | SDPA math fwd | speedup | FA fwd+bwd | SDPA math fwd+bwd | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.034 ms | 1.667 ms | 49.1x | 0.859 ms | 3.721 ms | 4.3x |
| 2048 | 0.234 ms | 21.895 ms | 93.8x | 10.510 ms | 49.170 ms | 4.7x |
| 4096 | 0.780 ms | 86.670 ms | 111.2x | 41.070 ms | 261.147 ms | 6.4x |

### Against PyTorch's AOTriton flash backend

`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python verify.py`

| seq | FA fwd | SDPA AOTriton fwd | speedup | FA fwd+bwd | SDPA AOTriton fwd+bwd | FA relative |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.034 ms | 0.082 ms | 2.4x | 0.893 ms | 0.308 ms | 2.9x slower |
| 2048 | 0.295 ms | 0.783 ms | 2.7x | 10.643 ms | 1.690 ms | 6.3x slower |
| 4096 | 0.787 ms | 1.802 ms | 2.3x | 41.371 ms | 5.860 ms | 7.1x slower |

**Read this before using it for training.** The CK forward kernel is consistently faster
(2.3–2.7x) than PyTorch's AOTriton kernel on gfx1200, and uses far less memory than the math
fallback. But CK's **backward** kernel is currently much slower than AOTriton on this GPU.
For training, PyTorch SDPA with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is the better
choice unless you specifically need the `flash_attn` API. For inference, this wheel is the
faster option.

### End-to-end LFM2.5-350M step

`bench_lfm2.py`, batch 4, seq 512, fwd+bwd, 5 timed steps after 2 warmup. This model is
hybrid conv/attention, and `causal_conv1d` falls back to PyTorch on ROCm, so attention is
only part of the step:

| attention backend | ms/step | peak GiB |
|---|---:|---:|
| SDPA math | 300.8 | 5.08 |
| SDPA AOTriton | 242.9 | 4.61 |
| flash-attn CK | 254.1 | 4.61 |

## Build from source

Requirements: ROCm PyTorch, ROCm development headers (`/opt/rocm/include`), `gfx1200`
visible to PyTorch, ~30 GB disk, and enough RAM for the parallel compile.

```bash
git clone https://github.com/tungnguyenlam/flash-attn-gfx1200.git
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

The published wheel was built with `MAX_JOBS=2` on an i5-14600K and took about **4 hours**
(1,837 generated CK kernels). `MAX_JOBS=6` is roughly 2–3x faster if the CPU has the
cooling and RAM headroom. `MAX_JOBS=1` is slow but very gentle on the machine.

## Limitations

- **gfx1200 only.** The binary contains kernels for `gfx1200`; use
  `GPU_ARCHS=gfx1201 ./build.sh` for RX 9070-class cards.
- **Exact ABI.** Rebuild after changing Python, PyTorch, or the C++ ABI.
- **CK backward is slower than AOTriton on gfx1200** (see Results).
- **No sliding-window attention in the CK backend** (upstream limitation). Full causal and
  non-causal attention, GQA/MQA, varlen, dropout, and head dims up to 256 are supported.
- CK on RDNA4 is not an officially supported configuration. Correctness was checked against
  fp32 SDPA; see `verify.py`.

## License

The wheel is a build of [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention),
BSD 3-Clause. The upstream license is included in [LICENSE](LICENSE).
