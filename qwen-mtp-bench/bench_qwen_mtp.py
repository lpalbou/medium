#!/usr/bin/env python3
"""
Prefill / decode benchmark for Qwen3.8-27B (MLX 4-bit) with and without a
precomputed prompt cache, and with or without an MTP speculative drafter.

For every story file, and every cache mode:

  nocache : the full prompt (story + question) is prefilled cold, then decoded.
  cache   : 1) the cache is PRECOMPUTED on the story prefix only (timed as
               "cache build"), then
            2) the question is asked: only the few question tokens are prefilled
               on top of the cache, then the answer is decoded.

Each configuration (regular / mtp) runs in its OWN subprocess, so the model is
fully unloaded (process exit) before the next configuration is loaded.

Usage
-----
  # everything: regular model, then MTP, then the summary table
  python bench_qwen_mtp.py all

  # a single configuration (appends to results.jsonl)
  python bench_qwen_mtp.py run --label regular
  python bench_qwen_mtp.py run --label mtp --draft-model mlx-community/Qwen3.8-27B-MTP-4bit

  # re-render the table from results.jsonl
  python bench_qwen_mtp.py report

Requirements: Apple Silicon, mlx + mlx-vlm with Qwen3.5/3.8 MTP drafter support
(tested with mlx 0.32.2, mlx-vlm 0.7.3). Models are resolved from the HF cache
or downloaded.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "mlx-community/Qwen3.8-27B-4bit"
DEFAULT_DRAFT = "mlx-community/Qwen3.8-27B-MTP-4bit"
DEFAULT_STORIES = [
    str(HERE / "stories" / "story_512.txt"),
    str(HERE / "stories" / "story_20k.txt"),
    str(HERE / "stories" / "story_30k.txt"),
    str(HERE / "stories" / "story_40k.txt"),
]
DEFAULT_QUESTION = "Summarize it."


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def system_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    if sys.platform == "darwin":
        for key, name in (("chip", "machdep.cpu.brand_string"), ("mem_bytes", "hw.memsize")):
            try:
                info[key] = subprocess.check_output(["sysctl", "-n", name], text=True).strip()
            except Exception:
                pass
    try:
        import mlx.core as mx  # noqa: F401
        from importlib.metadata import version

        info["mlx"] = version("mlx")
        info["mlx_vlm"] = version("mlx-vlm")
    except Exception:
        pass
    return info


def build_prompt(tokenizer, story: str, question: str, enable_thinking: bool):
    """Return (full_ids, prefix_len) where full_ids[:prefix_len] only covers
    the story part of the chat-templated prompt (the cacheable prefix)."""
    messages = [{"role": "user", "content": f"{story}\n\n{question}"}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    full_ids = tokenizer.encode(text, add_special_tokens=False)

    cut = text.index(story) + len(story)
    prefix_ids = tokenizer.encode(text[:cut], add_special_tokens=False)
    n = 0
    for a, b in zip(prefix_ids, full_ids):
        if a != b:
            break
        n += 1
    # BPE may merge the last story token with the separator; keep only the
    # token-aligned common part (loses at most a token or two).
    return full_ids, n, text


def cache_nbytes(prompt_cache) -> int:
    total = 0
    for c in prompt_cache:
        try:
            st = c.state
        except Exception:
            continue
        stack = [st]
        while stack:
            x = stack.pop()
            if x is None:
                continue
            if isinstance(x, (list, tuple)):
                stack.extend(x)
            elif hasattr(x, "nbytes"):
                total += int(x.nbytes)
    return total


# --------------------------------------------------------------------------- #
# single configuration (runs in-process)
# --------------------------------------------------------------------------- #
def cmd_run(args) -> None:
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.generate import stream_generate
    from mlx_vlm.generate.ar import generate_step
    from mlx_vlm.generate.common import PromptCacheState
    from mlx_vlm.models import cache as cache_mod

    sysinfo = system_info()
    log(f"system: {sysinfo}")

    t = time.perf_counter()
    model, processor = load(args.model)
    load_s = time.perf_counter() - t
    log(f"loaded {args.model} in {load_s:.1f}s  (active mem {mx.get_active_memory()/1e9:.2f} GB)")
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    draft_model, draft_kind = None, None
    if args.draft_model:
        from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility

        draft_model, draft_kind = load_drafter(args.draft_model, kind=args.draft_kind)
        validate_drafter_compatibility(model, draft_model, draft_kind)
        log(f"loaded drafter {args.draft_model} (kind={draft_kind}) "
            f"(active mem {mx.get_active_memory()/1e9:.2f} GB)")

    gen_kwargs = dict(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        prefill_step_size=args.prefill_step_size,
    )
    if draft_model is not None:
        gen_kwargs.update(draft_model=draft_model, draft_kind=draft_kind)
        if args.draft_block_size:
            gen_kwargs["draft_block_size"] = args.draft_block_size

    def spec_stats():
        if draft_model is None:
            return None
        try:
            from mlx_vlm.speculative.utils import format_speculative_stats

            return format_speculative_stats(draft_model)
        except Exception:
            return None

    def decode(full_ids, text, prompt_cache_state=None, step=None):
        """Run stream_generate; return timing dict + output text."""
        kw = dict(gen_kwargs)
        if step is not None:
            kw["prefill_step_size"] = step
        if prompt_cache_state is not None:
            kw["prompt_cache_state"] = prompt_cache_state
        input_ids = mx.array([full_ids])
        t0 = time.perf_counter()
        t_first = t_last = None
        n_tok, out, last = 0, [], None
        for r in stream_generate(model, processor, text, input_ids=input_ids, **kw):
            now = time.perf_counter()
            if r.token is not None and r.generation_tokens > n_tok:
                if t_first is None:
                    t_first = now
                t_last = now
                n_tok = r.generation_tokens
            out.append(r.text)
            last = r
        ttft = (t_first or time.perf_counter()) - t0
        dec_s = (t_last - t_first) if (t_first and t_last) else 0.0
        return dict(
            ttft_s=ttft,
            gen_tokens=n_tok,
            decode_s=dec_s,
            decode_tps=(n_tok - 1) / dec_s if dec_s > 0 and n_tok > 1 else None,
            total_s=(t_last or t0) - t0,
            cached_tokens=getattr(last, "cached_tokens", 0) if last else 0,
            finish_reason=getattr(last, "finish_reason", None) if last else None,
            text="".join(out),
        )

    def run_mode(mode, full_ids, prefix_len, text, step):
        """One measurement. Returns (decode result, extra fields)."""
        if mode == "nocache":
            r = decode(full_ids, text, step=step)
            return r, dict(
                cache_build_s=None, cache_build_tps=None, cache_bytes=None,
                prefilled_tokens=len(full_ids), prefill_tps=len(full_ids) / r["ttft_s"],
            )
        if mode != "cache":
            raise ValueError(mode)
        # 1) precompute the cache on the story prefix only
        lm = model.language_model
        for attr in ("_position_ids", "_rope_deltas"):
            if hasattr(lm, attr):
                setattr(lm, attr, None)
        pc = cache_mod.make_prompt_cache(lm)
        try:
            t0 = time.perf_counter()
            for _ in generate_step(
                mx.array([full_ids[:prefix_len]]), model, None, None,
                prompt_cache=pc, max_tokens=0, prefill_step_size=step,
            ):
                pass
            mx.eval([c.state for c in pc])
            build_s = time.perf_counter() - t0
            state = PromptCacheState()
            state.update(full_ids[:prefix_len], pc)
            cbytes = cache_nbytes(pc)
            log(f"   cache built: {prefix_len} tok in {build_s:.2f}s "
                f"({prefix_len/build_s:.1f} tok/s, {cbytes/1e9:.2f} GB)")
            # 2) ask the question on top of the cache
            r = decode(full_ids, text, prompt_cache_state=state, step=step)
        finally:
            del pc
        if r["cached_tokens"] != prefix_len:
            raise ValueError(
                f"cache was NOT reused (cached_tokens={r['cached_tokens']}, "
                f"expected {prefix_len}); results would be invalid")
        suffix = len(full_ids) - prefix_len
        return r, dict(
            cache_build_s=build_s, cache_build_tps=prefix_len / build_s,
            cache_bytes=cbytes, prefilled_tokens=suffix, prefill_tps=suffix / r["ttft_s"],
        )

    # warmup (kernel compilation etc.), not recorded
    wu_ids, _, wu_text = build_prompt(tokenizer, "The quick brown fox jumps over the lazy dog.",
                                      "Repeat it.", args.enable_thinking)
    wk = dict(gen_kwargs, max_tokens=16)
    for _ in stream_generate(model, processor, wu_text, input_ids=mx.array([wu_ids]), **wk):
        pass
    mx.clear_cache()
    log("warmup done")

    out_path = Path(args.out)
    for story_path in args.stories:
        story = Path(story_path).read_text(encoding="utf-8").strip()
        story_tokens = len(tokenizer.encode(story, add_special_tokens=False))
        full_ids, prefix_len, text = build_prompt(tokenizer, story, args.question, args.enable_thinking)
        ctx_name = Path(story_path).stem
        log(f"== {ctx_name}: story={story_tokens} tok, prompt={len(full_ids)} tok, "
            f"cacheable prefix={prefix_len} tok")

        ctx_step = args.prefill_step_size  # may shrink on OOM; sticky per context
        for mode in args.modes:
            for rep in range(args.repeats):
                rec = dict(
                    label=args.label, model=args.model, draft_model=args.draft_model,
                    context=ctx_name, story_tokens=story_tokens, prompt_tokens=len(full_ids),
                    mode=mode, repeat=rep, max_tokens=args.max_tokens,
                    temperature=args.temperature, question=args.question,
                    load_s=load_s, system=sysinfo,
                    timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
                )
                while True:
                    gc.collect()
                    mx.clear_cache()
                    mx.reset_peak_memory()
                    try:
                        r, extra = run_mode(mode, full_ids, prefix_len, text, ctx_step)
                        break
                    except RuntimeError as e:
                        oom = "memory" in str(e).lower()
                        gc.collect()
                        mx.clear_cache()
                        if oom and args.min_prefill_step_size and ctx_step > args.min_prefill_step_size:
                            log(f"   OOM with prefill_step_size={ctx_step}; retrying with {ctx_step // 2}")
                            ctx_step //= 2
                            continue
                        log(f"   FAILED [{args.label}/{mode}]: {e}")
                        r, extra = None, dict(error=str(e)[:300])
                        break
                rec["prefill_step_size"] = ctx_step
                rec.update(extra)
                if r is None:
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                    continue
                text_out = r.pop("text")
                rec.update(r)
                rec["peak_mem_gb"] = mx.get_peak_memory() / 1e9
                rec["spec_stats"] = spec_stats()
                rec["output_sha1"] = hashlib.sha1(text_out.encode()).hexdigest()[:12]
                rec["output"] = text_out
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                dtps = f"{rec['decode_tps']:.2f}" if rec["decode_tps"] else "-"
                log(f"   [{args.label}/{mode}] TTFT={rec['ttft_s']:.2f}s "
                    f"prefill={rec['prefill_tps']:.1f} tok/s  gen={rec['gen_tokens']} tok "
                    f"decode={dtps} tok/s  peak={rec['peak_mem_gb']:.2f} GB  chunk={ctx_step}"
                    + (f"  spec: {rec['spec_stats']}" if rec["spec_stats"] else ""))
                gc.collect()
                mx.clear_cache()

    del model, processor, draft_model
    gc.collect()
    mx.clear_cache()


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _f(x, fmt="{:.2f}"):
    return "-" if x is None else fmt.format(x)


def cmd_report(args) -> str:
    rows = [json.loads(l) for l in Path(args.out).read_text().splitlines() if l.strip()]
    if not rows:
        print("no results")
        return ""
    # average repeats
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["label"], r["context"], r["mode"]), []).append(r)

    def avg(rs, k):
        vals = [r[k] for r in rs if r.get(k) is not None]
        return sum(vals) / len(vals) if vals else None

    ctx_len = {r["context"]: r["prompt_tokens"] for r in rows}
    order_ctx = {c: ctx_len[c] for c in ctx_len}
    order_lbl = {c: i for i, c in enumerate(dict.fromkeys(r["label"] for r in rows))}
    keys = sorted(groups, key=lambda k: (order_ctx[k[1]], order_lbl[k[0]], k[2] != "nocache"))

    lines = []
    sysinfo = rows[0].get("system", {})
    lines.append(f"System: {sysinfo.get('chip','?')}, "
                 f"{int(sysinfo.get('mem_bytes', 0))/2**30:.0f} GB, "
                 f"mlx {sysinfo.get('mlx','?')}, mlx-vlm {sysinfo.get('mlx_vlm','?')}")
    lines.append(f"Model: {rows[0]['model']}  |  question: \"{rows[0]['question']}\"  |  "
                 f"max_tokens={rows[0]['max_tokens']}, temperature={rows[0]['temperature']}")
    lines.append("")
    hdr = ["Context", "Model", "Cache", "Prompt tok", "Prefill chunk", "Cache build (s)", "Cache build tok/s",
           "Prefilled tok", "TTFT (s)", "Prefill tok/s", "Gen tok", "Decode tok/s",
           "Decode vs regular", "Total (s)", "E2E incl. build (s)", "Peak mem (GB)", "Same output as regular"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for k in keys:
        label, ctx, mode = k
        rs = [r for r in groups[k] if not r.get("error")]
        if not rs:
            err = groups[k][0]["error"][:60]
            lines.append("| " + " | ".join([ctx, label, mode, str(groups[k][0]["prompt_tokens"]),
                         str(groups[k][0].get("prefill_step_size", "-")), f"FAILED: {err}"]
                         + [""] * (len(hdr) - 6)) + " |")
            continue
        ref = [r for r in groups.get(("regular", ctx, mode), []) if not r.get("error")]
        dec = avg(rs, "decode_tps")
        ref_dec = avg(ref, "decode_tps") if ref else None
        speed = f"{dec/ref_dec:.2f}x" if (dec and ref_dec and label != "regular") else "-"
        same = "-"
        if ref and label != "regular":
            same = "yes" if rs[0]["output_sha1"] == ref[0]["output_sha1"] else "no"
        build = avg(rs, "cache_build_s")
        total = avg(rs, "total_s")
        e2e = (total or 0) + (build or 0)
        lines.append("| " + " | ".join([
            ctx, label, mode, str(rs[0]["prompt_tokens"]), str(rs[0].get("prefill_step_size", "-")),
            _f(build), _f(avg(rs, "cache_build_tps"), "{:.1f}"),
            str(rs[0]["prefilled_tokens"]), _f(avg(rs, "ttft_s")),
            _f(avg(rs, "prefill_tps"), "{:.1f}"),
            _f(avg(rs, "gen_tokens"), "{:.0f}"), _f(dec),
            speed, _f(total), _f(e2e), _f(avg(rs, "peak_mem_gb")), same,
        ]) + " |")
    spec = [r for r in rows if r.get("spec_stats") and not r.get("error")]
    if spec:
        lines.append("")
        lines.append("Speculative (MTP) stats:")
        for r in spec:
            lines.append(f"- {r['context']} / {r['mode']}: {r['spec_stats']}")
    md = "\n".join(lines)
    print(md)
    if args.report_file:
        Path(args.report_file).write_text(md + "\n")
    return md


# --------------------------------------------------------------------------- #
# orchestration: one subprocess per configuration (full unload in between)
# --------------------------------------------------------------------------- #
def cmd_all(args) -> None:
    common = ["--model", args.model, "--out", args.out, "--max-tokens", str(args.max_tokens),
              "--temperature", str(args.temperature), "--question", args.question,
              "--prefill-step-size", str(args.prefill_step_size), "--repeats", str(args.repeats),
              "--modes", *args.modes, "--stories", *args.stories]
    if args.enable_thinking:
        common.append("--enable-thinking")
    if args.min_prefill_step_size:
        common += ["--min-prefill-step-size", str(args.min_prefill_step_size)]
    configs = [("regular", [])]
    if not args.skip_mtp:
        configs.append(("mtp", ["--draft-model", args.draft_model]))
    for label, extra in configs:
        if args.only and label not in args.only:
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "run", "--label", label, *common, *extra]
        log(f"### starting configuration '{label}' in a fresh process")
        rc = subprocess.run(cmd).returncode
        log(f"### configuration '{label}' finished (exit code {rc}); process exited (model unloaded)")
    cmd_report(args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--stories", nargs="+", default=DEFAULT_STORIES)
        sp.add_argument("--question", default=DEFAULT_QUESTION)
        sp.add_argument("--modes", nargs="+", default=["nocache", "cache"], choices=["nocache", "cache"])
        sp.add_argument("--max-tokens", type=int, default=512)
        sp.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy (MTP output then identical to regular)")
        sp.add_argument("--prefill-step-size", type=int, default=2048,
                        help="prefill chunk; halved automatically on Metal OOM")
        sp.add_argument("--min-prefill-step-size", type=int, default=None,
                        help="opt-in: on Metal OOM, halve the prefill chunk down to this size "
                             "(default: no retry, the run is recorded as FAILED)")
        sp.add_argument("--repeats", type=int, default=1)
        sp.add_argument("--enable-thinking", action="store_true")
        sp.add_argument("--out", default=str(HERE / "results.jsonl"))
        sp.add_argument("--report-file", default=str(HERE / "results.md"))

    sp = sub.add_parser("run", help="run one configuration in this process")
    add_common(sp)
    sp.add_argument("--label", default="regular")
    sp.add_argument("--draft-model", default=None)
    sp.add_argument("--draft-kind", default=None)
    sp.add_argument("--draft-block-size", type=int, default=None)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("all", help="regular then MTP (separate processes) then report")
    add_common(sp)
    sp.add_argument("--draft-model", default=DEFAULT_DRAFT)
    sp.add_argument("--skip-mtp", action="store_true")
    sp.add_argument("--only", nargs="+", choices=["regular", "mtp"])
    sp.set_defaults(func=cmd_all)

    sp = sub.add_parser("report", help="render the markdown table from results.jsonl")
    sp.add_argument("--out", default=str(HERE / "results.jsonl"))
    sp.add_argument("--report-file", default=str(HERE / "results.md"))
    sp.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
