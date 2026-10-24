Prebuilt **FlashAttention-2 2.8.4** for **AMD RX 9060 XT 16 GB (gfx1200 / RDNA4)** with the Composable Kernel backend.

Built from upstream commit [`10b74d9`](https://github.com/Dao-AILab/flash-attention/commit/10b74d98fb26113b014b33d947c8bbe4e83fbf7a) and CK [`c56c675`](https://github.com/ROCm/composable_kernel/commit/c56c6750d0fc54ed771d532cc92c316423449614).

### Compatibility

| | |
|---|---|
| GPU | gfx1200 (RX 9060 XT 16 GB) |
| Python | 3.12 |
| PyTorch | 2.13.0+rocm7.1 (HIP 7.1.52802) |
| ROCm headers | 7.2.5 |
| ABI | cxx11abiTRUE |
| Wheel | `flash_attn-2.8.4-cp312-cp312-linux_x86_64.whl` (41 MB) |

### Install

```bash
python -m pip install --no-deps \
  https://github.com/tungnguyenlam/flash-attn-gfx1200/releases/download/v2.8.4-rocm7.1-torch2.13-gfx1200/flash_attn-2.8.4-cp312-cp312-linux_x86_64.whl
python -m pip install einops
```

`--no-deps` prevents pip from replacing your ROCm PyTorch with a CUDA wheel from PyPI.

SHA256: `c2b0da480c1186084d1819827736a2a5dbfb9116ff19fad32b89966d376c128d`

### Verified

- Forward and backward match fp32 SDPA within bf16 tolerance for MHA and GQA (16/8 heads), seq 512–1024, head dim 64.
- Forward is 2.3–2.7x faster than PyTorch's AOTriton SDPA flash kernel on gfx1200.
- **Backward is slower than AOTriton** (2.9–7.1x depending on sequence length). For training, prefer `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` with SDPA unless you specifically need the `flash_attn` API. For inference, this wheel is faster.

See the [README](https://github.com/tungnguyenlam/flash-attn-gfx1200#results) for full benchmark tables, build instructions, and limitations. The wheel contains gfx1200 kernels only.
