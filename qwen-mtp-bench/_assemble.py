# one-off helper: concatenate parts and trim at a paragraph boundary to <= target tokens
import sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('mlx-community/Qwen3.8-27B-4bit')
title, target, out, parts = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
paras = [title]
for p in parts:
    paras += [x.strip() for x in open(p).read().split("\n\n") if x.strip()]
n = lambda s: len(t.encode(s, add_special_tokens=False))
total = n("\n\n".join(paras)); print("untrimmed tokens:", total)
while n("\n\n".join(paras)) > target:
    paras.pop()
text = "\n\n".join(paras); open(out, "w").write(text + "\n"); print("final tokens:", n(text), "| last para:", paras[-1][:80])
