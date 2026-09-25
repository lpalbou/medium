# Qwen3.8-27B (MLX 4-bit): prefill / decode benchmark, regular vs MTP, cache vs no-cache

## Contents
- `bench_qwen_mtp.py`: standalone benchmark script
- `stories/story_512.txt`: "The Lighthouse at Vesta Gap" (507 tokens)
- `stories/story_20k.txt`: "The Cartographer of Kepler's Drift" (19,913 tokens)
- `stories/story_30k.txt`: "The Long Quiet of Halcyon Station", first 29,817 tokens
- `stories/story_40k.txt`: same novel, first 39,958 tokens
- `stories/story_50k.txt`: same novel, full (49,616 tokens), for machines with more memory (`--stories ...`)
- `stories/parts/` + `_assemble.py`: source chapters and the helper that trims them to a token budget
- `results.jsonl` / `results.md`: raw per-run records and the summary table

Token counts use the Qwen3.8 tokenizer on the story text alone. The chat template and question add about 18 tokens.

## What is measured
For each story, the question asked is always **"Summarize it."**. Settings are greedy (temperature 0), thinking disabled, and max 512 new tokens.

| mode | what happens | reported |
|---|---|---|
| `nocache` | full prompt (story + question) prefilled cold, then decoded | TTFT, prefill tok/s = prompt_tokens / TTFT, decode tok/s |
| `cache` | (1) the cache is **precomputed** on the story prefix only, (2) the question is asked on top of it, so only ~15 tokens are prefilled | cache build time and tok/s, TTFT of the question, decode tok/s |

Decode tok/s = (generated_tokens − 1) / (time last token − time first token).

The script runs the regular model and the MTP configuration in **separate processes**. The first model is fully unloaded (the process exits) before the MTP configuration loads. With greedy decoding, MTP speculative decoding is lossless, so the output must be identical to the regular model. The table checks this (`Same output as regular`).

## Run
```bash
pip install "mlx-vlm>=0.7.3"          # tested: mlx 0.32.2, mlx-vlm 0.7.3
python bench_qwen_mtp.py all          # regular -> MTP -> table (results.md)

# options
python bench_qwen_mtp.py all --repeats 3 --max-tokens 512
python bench_qwen_mtp.py all --only mtp          # just one configuration
python bench_qwen_mtp.py run --label mtp --draft-model mlx-community/Qwen3.8-27B-MTP-4bit
python bench_qwen_mtp.py report                  # re-render table from results.jsonl
```
Models: target `mlx-community/Qwen3.8-27B-4bit` (~16 GB) and MTP drafter `mlx-community/Qwen3.8-27B-MTP-4bit` (~240 MB, not standalone; it reuses the target's embeddings, LM head and KV). Both are downloaded if they are not in the HF cache.

## Memory
See `results.md` for the full measured table and the 32 GB projection.

Peak memory is weights (16.06 GB, +0.24 GB with MTP) + KV cache + a prefill working set of about 3 GB at the default 2048 prefill chunk.
The KV cache only grows in the 16 full-attention layers: **64 KiB/token**, plus a constant 0.15 GB of linear-attention state. That's 1.46 GB at 20k and 2.77 GB at 40k (both verified against measurement).

| context | KV cache | est. peak (regular) | est. peak (MTP) |
|---|---|---|---|
| 20k | 1.5 GB | 20.3 GB (measured) | 20.6 GB (measured) |
| 30k | 2.1 GB | 21.1 GB (measured) | 21.35 GB (measured) |
| 40k | 2.8 GB | 21.9 GB (measured) | 22.1 GB (measured) |
| 50k | 3.4 GB | ~22.7 GB | ~22.9 GB |
| 100k | 6.7 GB | ~26.5 GB | ~26.8 GB |
| 120k | 8.0 GB | ~28.1 GB | ~28.3 GB |

On a 24 GB Mac, the default GPU working-set limit is 20 GB (`iogpu.wired_limit_mb=20480`). That's enough for 30k; at 40k, cold prefill fails with an OOM. Raising the limit to 21 GB (`sudo sysctl iogpu.wired_limit_mb=21504`, reverts on reboot) makes 40k work for both regular and MTP, at the default 2048 chunk. The OS is then very constrained, which slows cache builds and cached TTFT.

`--min-prefill-step-size N` is opt-in and off by default. It retries a failed run with smaller prefill chunks.
