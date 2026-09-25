System: Apple M6, 24 GB, mlx 0.32.2, mlx-vlm 0.7.3
Model: mlx-community/Qwen3.8-27B-4bit  |  question: "Summarize it."  |  max_tokens=512, temperature=0.0

| Context | Model | Cache | Prompt tok | Prefill chunk | Cache build (s) | Cache build tok/s | Prefilled tok | TTFT (s) | Prefill tok/s | Gen tok | Decode tok/s | Decode vs regular | Total (s) | E2E incl. build (s) | Peak mem (GB) | Same output as regular |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| story_30k | regular | nocache | 29835 | 2048 | - | - | 29835 | 109.16 | 273.3 | 512 | 8.43 | - | 169.74 | 169.74 | 21.11 | - |
| story_30k | regular | cache | 29835 | 2048 | 112.62 | 264.8 | 15 | 2.01 | 7.5 | 512 | 8.42 | - | 62.67 | 175.29 | 21.11 | - |
| story_30k | mtp | nocache | 29835 | 2048 | - | - | 29835 | 111.01 | 268.8 | 512 | 12.51 | 1.48x | 151.85 | 151.85 | 21.35 | yes |
| story_30k | mtp | cache | 29835 | 2048 | 116.88 | 255.1 | 15 | 3.42 | 4.4 | 512 | 11.84 | 1.41x | 46.59 | 163.48 | 21.35 | yes |

Speculative (MTP) stats:
- story_30k / nocache: Speculative decoding: 2.12 accepted tokens/round (1.12 accepted drafts/round, 55.8% of drafted, avg draft 2.00) over 242 rounds
- story_30k / cache: Speculative decoding: 2.00 accepted tokens/round (1.00 accepted drafts/round, 50.1% of drafted, avg draft 2.00) over 256 rounds
