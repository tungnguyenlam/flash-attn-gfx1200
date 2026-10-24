#!/usr/bin/env python
"""End-to-end LFM2 forward+backward timing for SDPA vs flash-attn.

    python bench_lfm2.py --attn sdpa
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python bench_lfm2.py --attn sdpa
    python bench_lfm2.py --attn flash_attention_2
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--model", default="LiquidAI/LFM2.5-350M")
p.add_argument("--attn", choices=["sdpa", "flash_attention_2"], default="flash_attention_2")
p.add_argument("--batch", type=int, default=4)
p.add_argument("--seq", type=int, default=512)
p.add_argument("--warmup", type=int, default=2)
p.add_argument("--rep", type=int, default=5)
args = p.parse_args()

torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16, attn_implementation=args.attn, device_map="cuda"
)
model.train()
ids = torch.randint(0, model.config.vocab_size, (args.batch, args.seq), device="cuda")


def step():
    out = model(input_ids=ids, labels=ids, use_cache=False)
    out.loss.backward()
    model.zero_grad(set_to_none=True)


for _ in range(args.warmup):
    step()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start, end = torch.cuda.Event(True), torch.cuda.Event(True)
start.record()
for _ in range(args.rep):
    step()
end.record()
torch.cuda.synchronize()

result = {
    "attn": args.attn,
    "batch": args.batch,
    "seq": args.seq,
    "ms_per_step": round(start.elapsed_time(end) / args.rep, 2),
    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    "torch": torch.__version__,
    "aotriton": torch.cuda.is_available() and bool(__import__("os").environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")),
}
print(json.dumps(result))
