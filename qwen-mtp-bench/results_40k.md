System: Apple M6, 24 GB, mlx 0.32.2, mlx-vlm 0.7.3
Model: mlx-community/Qwen3.8-27B-4bit  |  question: "Summarize it."  |  max_tokens=512, temperature=0.0

| Context | Model | Cache | Prompt tok | Prefill chunk | Cache build (s) | Cache build tok/s | Prefilled tok | TTFT (s) | Prefill tok/s | Gen tok | Decode tok/s | Decode vs regular | Total (s) | E2E incl. build (s) | Peak mem (GB) | Same output as regular |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| story_40k | regular | nocache | 39976 | 2048 | - | - | 39976 | 156.20 | 255.9 | 512 | 7.95 | - | 220.44 | 220.44 | 21.89 | - |
| story_40k | regular | cache | 39976 | 2048 | 188.32 | 212.2 | 15 | 3.37 | 4.4 | 512 | 7.88 | - | 68.20 | 256.52 | 21.89 | - |
| story_40k | mtp | nocache | 39976 | 2048 | - | - | 39976 | 156.27 | 255.8 | 512 | 11.15 | 1.40x | 202.09 | 202.09 | 22.12 | yes |
| story_40k | mtp | cache | 39976 | 2048 | 168.94 | 236.5 | 15 | 7.69 | 1.9 | 512 | 11.19 | 1.42x | 53.36 | 222.29 | 22.13 | yes |

Speculative (MTP) stats:
- story_40k / nocache: Speculative decoding: 2.08 accepted tokens/round (1.08 accepted drafts/round, 54.0% of drafted, avg draft 2.00) over 246 rounds
- story_40k / cache: Speculative decoding: 2.12 accepted tokens/round (1.12 accepted drafts/round, 55.8% of drafted, avg draft 2.00) over 242 rounds
