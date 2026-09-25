#!/usr/bin/env python3
"""Render article charts from results.jsonl (measured) + fitted projections."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
OUT = HERE / "charts"
OUT.mkdir(exist_ok=True)

REG, MTP = "#3B5BDB", "#F76707"           # regular = blue, MTP = orange
REG_L, MTP_L = "#A5B4FC", "#FFC078"       # lighter tints for the generation segment
INK, MUTED, GRID = "#1F2328", "#6B7280", "#E5E7EB"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

# ------------------------------------------------------------------ data
rows = [json.loads(l) for l in open(HERE / "results.jsonl") if l.strip()]
R = {(r["label"], r["context"], r["mode"]): r for r in rows if not r.get("error")}
CTX = ["story_512", "story_20k", "story_30k", "story_40k"]
Lm = np.array([R[("regular", c, "nocache")]["prompt_tokens"] for c in CTX]) / 1000
GEN = 512

def dec_tps(label, c):
    return np.mean([R[(label, c, m)]["decode_tps"] for m in ("nocache", "cache")])

reg_m = np.array([dec_tps("regular", c) for c in CTX])
mtp_m = np.array([dec_tps("mtp", c) for c in CTX])

# fits (same model as results.md): prefill quadratic, decode s/token linear
ttft = np.array([R[("regular", c, "nocache")]["ttft_s"] for c in CTX])
(pa, pb), *_ = np.linalg.lstsq(np.vstack([Lm, Lm**2]).T, ttft, rcond=None)
lin = lambda y: np.linalg.lstsq(np.vstack([np.ones_like(Lm), Lm]).T, y, rcond=None)[0]
rt0, rt1 = lin(1 / reg_m)
mt0, mt1 = lin(1 / mtp_m)
prefill = lambda L: pa * L + pb * L**2
reg_f = lambda L: 1 / (rt0 + rt1 * L)
mtp_f = lambda L: 1 / (mt0 + mt1 * L)
Lp = np.array([50, 100, 120])

FOOT = ("Qwen3.8-27B 4-bit (MLX) + MTP drafter (block size 3), mlx-vlm 0.7.3, Mac mini M6 24 GB. "
        "Greedy decoding; MTP output identical to regular.\n"
        "Solid = measured on 24 GB (40k with GPU cap raised to 21 GB). "
        "Hatched/dashed = projected for the 32 GB model from fits on the measured points.")

# ------------------------------------------------------------------ chart 1
fig, ax = plt.subplots(figsize=(12, 6.75), dpi=200)
ax.axvspan(45, 125, color="#F3F4F6", zorder=0)
ax.text(85, 19.2, "projected · 32 GB Mac mini M6", ha="center", color=MUTED, fontsize=11, style="italic")

xs = np.linspace(0.5, 120, 300)
ax.plot(xs[xs <= 40], reg_f(xs[xs <= 40]), color=REG, lw=2.2, alpha=.35)
ax.plot(xs[xs <= 40], mtp_f(xs[xs <= 40]), color=MTP, lw=2.2, alpha=.35)
ax.plot(xs[xs >= 40], reg_f(xs[xs >= 40]), color=REG, lw=2, ls="--", alpha=.6)
ax.plot(xs[xs >= 40], mtp_f(xs[xs >= 40]), color=MTP, lw=2, ls="--", alpha=.6)
ax.fill_between(xs, reg_f(xs), mtp_f(xs), color=MTP, alpha=.07, lw=0)

ax.scatter(Lm, reg_m, s=70, color=REG, zorder=5, label="Regular decoding (measured)")
ax.scatter(Lm, mtp_m, s=70, color=MTP, zorder=5, label="MTP speculative decoding (measured)")
ax.scatter(Lp, reg_f(Lp), s=70, facecolor="white", edgecolor=REG, lw=2, zorder=5, label="Regular (projected)")
ax.scatter(Lp, mtp_f(Lp), s=70, facecolor="white", edgecolor=MTP, lw=2, zorder=5, label="MTP (projected)")

for L, r, m, proj in [*zip(Lm, reg_m, mtp_m, [False] * 4), *zip(Lp, reg_f(Lp), mtp_f(Lp), [True] * 3)]:
    ax.annotate(f"{'~' if proj else ''}{m / r:.2f}×", (L, m), textcoords="offset points",
                xytext=(0, 11), ha="center", fontsize=11, fontweight="bold",
                color=MTP if not proj else "#C2570C")
for L, v in zip(Lm, reg_m):
    ax.annotate(f"{v:.1f}", (L, v), textcoords="offset points", xytext=(0, -18), ha="center",
                fontsize=10, color=REG)
for L, v in zip(Lm, mtp_m):
    ax.annotate(f"{v:.1f}", (L, v), textcoords="offset points", xytext=(15, -4), ha="left",
                fontsize=10, color=MTP)

ax.set_xlim(-3, 125); ax.set_ylim(0, 20.5)
ax.set_xticks([0.5, 20, 30, 40, 50, 100, 120], ["512", "20k", "30k", "40k", "50k", "100k", "120k"])
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Generation speed (tokens / second)")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.legend(loc="upper right", bbox_to_anchor=(1, .93), frameon=False, fontsize=10.5)
fig.suptitle("MTP speeds up Qwen3.8-27B generation, but the gain shrinks as context grows",
             x=.06, y=.965, ha="left", fontsize=17, fontweight="bold", color=INK)
fig.text(.06, .895, "Decode throughput on a Mac mini M6 (MLX 4-bit). Labels above the orange points = MTP speedup over regular decoding.",
         ha="left", fontsize=11.5, color=MUTED)
fig.text(.06, .02, FOOT, ha="left", fontsize=8.5, color=MUTED)
fig.subplots_adjust(left=.06, right=.98, top=.86, bottom=.17)
fig.savefig(OUT / "1_decode_speed_vs_context.png")
plt.close(fig)

# ------------------------------------------------------------------ chart 2
labels = ["512", "20k", "30k", "40k", "50k*", "100k*"]
cached_ttft_proj = {50: 5.0, 100: 10.0}   # midpoint of the estimated range
def row(label, mode):
    pre, gen = [], []
    for c in CTX:
        r = R[(label, c, mode)]
        pre.append(r["ttft_s"])
        gen.append((GEN - 1) / r["decode_tps"])       # normalise to 512 generated tokens
    f = reg_f if label == "regular" else mtp_f
    for L in (50, 100):
        pre.append(prefill(L) if mode == "nocache" else cached_ttft_proj[L])
        gen.append((GEN - 1) / f(L))
    return np.array(pre), np.array(gen)

fig, axes = plt.subplots(1, 2, figsize=(13, 7.8), dpi=200, gridspec_kw={"wspace": .18})
x = np.arange(len(labels)); w = .38
for ax, mode, title in [(axes[0], "nocache", "Cold prompt (no cache)"),
                        (axes[1], "cache", "Precomputed prompt cache")]:
    for off, label, c_dark, c_light in [(-w / 2, "regular", REG, REG_L), (w / 2, "mtp", MTP, MTP_L)]:
        pre, gen = row(label, mode)
        for i in range(len(labels)):
            proj = i >= 4
            kw = dict(width=w * .92, edgecolor="white" if not proj else c_dark, lw=.8,
                      hatch="///" if proj else None, alpha=.55 if proj else 1)
            ax.bar(x[i] + off, pre[i], color=c_dark, **kw)
            ax.bar(x[i] + off, gen[i], bottom=pre[i], color=c_light, **kw)
            tot = pre[i] + gen[i]
            ax.text(x[i] + off, tot, f"{'~' if proj else ''}{tot:.0f}s", ha="center", va="bottom",
                    fontsize=8.5, color=INK, rotation=0)
    ax.set_xticks(x, labels)
    ax.set_title(title, fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=10)
    ax.set_xlabel("Context length (tokens)   * projected, 32 GB model")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.axvspan(3.5, 5.6, color="#F3F4F6", zorder=0)
axes[0].set_ylabel("Seconds until the full 512-token answer")
axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.08)
axes[1].text(.02, .97, "One-time cache build ≈ the cold prefill time on the left.\n"
             "Only the question (~15 tokens) is processed at ask time.",
             transform=axes[1].transAxes, va="top", fontsize=9.5, color=MUTED)

handles = [Patch(color=REG, label="Regular · prompt processing (time to first token)"),
           Patch(color=REG_L, label="Regular · generating 512 tokens"),
           Patch(color=MTP, label="MTP · prompt processing (time to first token)"),
           Patch(color=MTP_L, label="MTP · generating 512 tokens")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.055, .885), ncol=2, frameon=False, fontsize=10.5)
fig.suptitle("Where the time goes: long prompts are dominated by prefill, and caching removes it",
             x=.055, y=.97, ha="left", fontsize=17, fontweight="bold", color=INK)
fig.text(.055, .905, "Time to answer \"Summarize it.\" about a sci-fi story of each length, Qwen3.8-27B 4-bit on a Mac mini M6. Note the different y-axis scales.",
         ha="left", fontsize=11.5, color=MUTED)
fig.text(.055, .015, FOOT.replace("Solid = measured", "Solid bars = measured").replace("Hatched/dashed", "Hatched")
         + " Generation time is normalized to 512 tokens.", ha="left", fontsize=8.5, color=MUTED)
fig.subplots_adjust(left=.055, right=.985, top=.75, bottom=.14)
fig.savefig(OUT / "2_time_to_answer.png")
plt.close(fig)
print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
