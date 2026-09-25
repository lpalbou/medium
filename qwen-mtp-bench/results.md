System: Apple M6, 24 GB, mlx 0.32.2, mlx-vlm 0.7.3
Model: mlx-community/Qwen3.8-27B-4bit  |  question: "Summarize it."  |  max_tokens=512, temperature=0.0

| Context | Model | Cache | Prompt tok | Prefill chunk | Cache build (s) | Cache build tok/s | Prefilled tok | TTFT (s) | Prefill tok/s | Gen tok | Decode tok/s | Decode vs regular | Total (s) | E2E incl. build (s) | Peak mem (GB) | Same output as regular |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| story_512 | regular | nocache | 525 | 2048 | - | - | 525 | 1.96 | 268.5 | 99 | 9.83 | - | 11.93 | 11.93 | 17.25 | - |
| story_512 | regular | cache | 525 | 2048 | 1.78 | 286.9 | 15 | 0.46 | 32.5 | 99 | 9.77 | - | 10.49 | 12.27 | 17.21 | - |
| story_512 | mtp | nocache | 525 | 2048 | - | - | 525 | 1.87 | 281.3 | 99 | 17.55 | 1.79x | 7.45 | 7.45 | 17.49 | yes |
| story_512 | mtp | cache | 525 | 2048 | 1.77 | 287.7 | 15 | 0.37 | 40.0 | 99 | 17.70 | 1.81x | 5.91 | 7.68 | 17.44 | yes |
| story_20k | regular | nocache | 19924 | 2048 | - | - | 19924 | 70.26 | 283.6 | 512 | 8.90 | - | 127.70 | 127.70 | 20.34 | - |
| story_20k | regular | cache | 19924 | 2048 | 74.46 | 267.4 | 15 | 0.91 | 16.4 | 512 | 8.88 | - | 58.48 | 132.94 | 20.34 | - |
| story_20k | mtp | nocache | 19924 | 2048 | - | - | 19924 | 71.43 | 278.9 | 512 | 13.57 | 1.53x | 109.09 | 109.09 | 20.58 | yes |
| story_20k | mtp | cache | 19924 | 2048 | 75.70 | 263.0 | 15 | 3.08 | 4.9 | 512 | 14.10 | 1.59x | 39.33 | 115.03 | 20.58 | yes |
| story_30k | regular | nocache | 29835 | 2048 | - | - | 29835 | 109.16 | 273.3 | 512 | 8.43 | - | 169.74 | 169.74 | 21.11 | - |
| story_30k | regular | cache | 29835 | 2048 | 112.62 | 264.8 | 15 | 2.01 | 7.5 | 512 | 8.42 | - | 62.67 | 175.29 | 21.11 | - |
| story_30k | mtp | nocache | 29835 | 2048 | - | - | 29835 | 111.01 | 268.8 | 512 | 12.51 | 1.48x | 151.85 | 151.85 | 21.35 | yes |
| story_30k | mtp | cache | 29835 | 2048 | 116.88 | 255.1 | 15 | 3.42 | 4.4 | 512 | 11.84 | 1.41x | 46.59 | 163.48 | 21.35 | yes |
| story_40k | regular | nocache | 39976 | 2048 | - | - | 39976 | 156.20 | 255.9 | 512 | 7.95 | - | 220.44 | 220.44 | 21.89 | - |
| story_40k | regular | cache | 39976 | 2048 | 188.32 | 212.2 | 15 | 3.37 | 4.4 | 512 | 7.88 | - | 68.20 | 256.52 | 21.89 | - |
| story_40k | mtp | nocache | 39976 | 2048 | - | - | 39976 | 156.27 | 255.8 | 512 | 11.15 | 1.40x | 202.09 | 202.09 | 22.12 | yes |
| story_40k | mtp | cache | 39976 | 2048 | 168.94 | 236.5 | 15 | 7.69 | 1.9 | 512 | 11.19 | 1.42x | 53.36 | 222.29 | 22.13 | yes |

Speculative (MTP) stats:
- story_512 / nocache: Speculative decoding: 2.15 accepted tokens/round (1.15 accepted drafts/round, 57.6% of drafted, avg draft 2.00) over 46 rounds
- story_512 / cache: Speculative decoding: 2.15 accepted tokens/round (1.15 accepted drafts/round, 57.6% of drafted, avg draft 2.00) over 46 rounds
- story_20k / nocache: Speculative decoding: 2.04 accepted tokens/round (1.04 accepted drafts/round, 52.2% of drafted, avg draft 2.00) over 250 rounds
- story_20k / cache: Speculative decoding: 2.14 accepted tokens/round (1.14 accepted drafts/round, 57.2% of drafted, avg draft 2.00) over 239 rounds
- story_40k / nocache: Speculative decoding: 2.08 accepted tokens/round (1.08 accepted drafts/round, 54.0% of drafted, avg draft 2.00) over 246 rounds
- story_40k / cache: Speculative decoding: 2.12 accepted tokens/round (1.12 accepted drafts/round, 55.8% of drafted, avg draft 2.00) over 242 rounds
- story_30k / nocache: Speculative decoding: 2.12 accepted tokens/round (1.12 accepted drafts/round, 55.8% of drafted, avg draft 2.00) over 242 rounds
- story_30k / cache: Speculative decoding: 2.00 accepted tokens/round (1.00 accepted drafts/round, 50.1% of drafted, avg draft 2.00) over 256 rounds

Notes:
- 512 / 20k / 30k were measured with the GPU cap at 20 GB (`iogpu.wired_limit_mb=20480`). 40k was measured with it raised to 21 GB (`21504`), because at 20 GB the cold 40k prefill fails with an OOM.
- The prefill chunk is the mlx-vlm default (2048) everywhere. It doesn't truncate anything: the full prompt is always in the KV cache.
- With greedy decoding, MTP output is byte-identical to the regular model in every cell. Cache and no-cache outputs differ slightly at 20k and above (bf16 rounding from a different prefill order).
- The "Prefill tok/s" column in cache mode is just 15 question tokens / TTFT. Cached TTFT is noisy (0.9-3.7 s at 20k across runs) because mlx-vlm re-wires ~20 GB of memory per request.

## Projection: Mac mini M6, 32 GB (same chip, GPU cap raised to 28 GB)
Fitted on all measured points (512 / 20k / 30k / 40k):
- Prefill time = 3.08·L + 0.0204·L² seconds (L in thousands of tokens)
- Regular decode = 0.1012 + 0.000606·L s/token
- MTP decode = 0.0562 + 0.000842·L s/token
- Peak memory = 16.06 GB weights (+0.24 GB MTP) + 64 KiB/token KV + 0.15 GB linear-attention state + (2.59 + 0.0116·L) GB prefill working set

| Context | KV cache | Peak mem reg / MTP | Prefill (cold) = cache build | Prefill tok/s | Cached TTFT (est.) | Decode regular | Decode MTP | MTP speedup | Answer 512 tok, cold: reg / MTP | Answer 512 tok, cached: reg / MTP |
|---|---|---|---|---|---|---|---|---|---|---|
| 50k | 3.4 GB | 22.7 / 22.9 GB | ~205 s | ~244 | ~4-6 s | ~7.6 tok/s | ~10.2 tok/s | ~1.34x | ~272 s / ~255 s | ~67 s / ~50 s |
| 100k | 6.7 GB | 26.5 / 26.8 GB | ~512 s | ~195 | ~8-12 s | ~6.2 tok/s | ~7.1 tok/s | ~1.15x | ~595 s / ~584 s | ~83 s / ~72 s |
| 120k (≈ max) | 8.0 GB | 28.1 / 28.3 GB | ~663 s | ~181 | ~10-14 s | ~5.8 tok/s | ~6.4 tok/s | ~1.11x | ~752 s / ~744 s | ~89 s / ~80 s |

Fit rule: on this machine, runs succeeded with the measured peak up to about 1.1-1.3 GB above the GPU cap. At 30k the peak was 21.1-21.35 GB under a 20 GB cap and succeeded; at 40k, 21.9 GB failed under the same cap. With a 28 GB cap, 120k is therefore the practical maximum, and ~130k (29.1 GB with MTP) is a coin flip. These are extrapolations up to 3x past the largest measured context; treat them as ±10-15%.
