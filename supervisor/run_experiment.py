#!/usr/bin/env python3
"""
Unit Prompt Experiment v7 — Focused Cross-Model Validation

Purpose: Generate publishable cross-model data for the "Are You a Good Supervisor?" article.
Design: Test top unit prompts across diverse LM Studio models on 3 behavioral tasks.
Scoring: LLM-as-judge with configurable judge model.

Lessons learned from v1-v6.1b (52 documented issues):
- System prompt passed via API system parameter, NOT concatenated (#46)
- Failed scores recorded as null, never substituted with fallback values (#27)
- Pre-flight connectivity check for every model (#1, #5)
- Incremental saves after every model + every N tests (#4)
- Complete verbatim capture of all prompts, responses, judge outputs (#v6.1b)
- Data-driven config via YAML, no hardcoded model mappings (#52)
- Valid JSON array output, not NDJSON (#2)
- Judge raw response captured for independent verification (#v6.1b)
- No global mutable state (#51)
- Proper token count from API metadata, not word splits (#47)

Usage:
    python run_experiment.py                    # Run all models
    python run_experiment.py --model "qwen3"    # Run models matching "qwen3"
    python run_experiment.py --dry-run           # Show plan without executing
    python run_experiment.py --judge "seed"      # Use a specific judge model
"""

import argparse
import json
import time
import yaml
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from abstractcore import create_llm
except ImportError:
    print("ERROR: abstractcore not installed. Run: pip install abstractcore")
    sys.exit(1)

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_model_llm(config: dict, model_id: str, provider: str | None = None):
    prov = provider or config.get("provider", "lmstudio")
    return create_llm(
        prov,
        model=model_id,
        temperature=config.get("temperature", 0.0),
        seed=config.get("seed", 42),
        max_tokens=config.get("max_tokens", 1024),
    )


def preflight_check(config: dict, model_id: str, label: str, provider: str | None = None) -> bool:
    try:
        llm = create_model_llm(config, model_id, provider)
        response = llm.generate("Say 'ready' and nothing else.")
        content = response.content if hasattr(response, "content") else str(response)
        if content and len(content.strip()) > 0:
            print(f"  OK: {label}")
            return True
        else:
            # #FALLBACK : empty response from model, treating as unavailable
            print(f"  FAIL: {label} — empty response")
            return False
    except Exception as e:
        print(f"  FAIL: {label} — {e}")
        return False


def run_single_test(llm, system_prompt: str | None, task_prompt: str) -> dict:
    start = time.time()
    try:
        response = llm.generate(
            task_prompt,
            system_prompt=system_prompt,
        )
        elapsed = time.time() - start
        content = response.content if hasattr(response, "content") else str(response)
        usage = getattr(response, "usage", None)
        return {
            "status": "success",
            "response": content,
            "elapsed_s": round(elapsed, 2),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }
    except Exception as e:
        return {
            "status": "error",
            "response": "",
            "error": str(e),
            "elapsed_s": round(time.time() - start, 2),
        }


def score_response(judge_llm, task: dict, response_text: str, scoring_rubric: dict) -> dict:
    rubric_text = ""
    for metric in task["measures"]:
        if metric in scoring_rubric:
            rubric_text += f"\n### {metric}\n"
            for score_val, desc in sorted(scoring_rubric[metric].items(), reverse=True):
                rubric_text += f"  {score_val}: {desc}\n"

    judge_system = (
        "You are a strict, objective evaluator. Score AI responses precisely according to the rubric. "
        "Use INTEGER scores from 1 to 10 (10 = exceptional, 7-8 = strong, 5-6 = average, "
        "3-4 = poor, 1-2 = unacceptable). Use the full scale when warranted. "
        "Respond with ONLY a JSON object. No explanation, no preamble."
    )

    judge_user = f"""Score this AI response on each metric.

TASK GIVEN TO THE AI:
{task['prompt']}

EXPECTED CORRECT BEHAVIOR:
{task.get('correct_answer', task.get('correct_behavior', 'N/A'))}

AI RESPONSE:
{response_text}

SCORING RUBRIC:
{rubric_text}

Return ONLY a JSON object: {{"metric_name": score, ...}}"""

    try:
        judge_response = judge_llm.generate(judge_user, system_prompt=judge_system)
        judge_text = judge_response.content if hasattr(judge_response, "content") else str(judge_response)

        match = re.search(r'\{[^}]+\}', judge_text, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            valid_scores = {}
            for k, v in scores.items():
                if k in task["measures"]:
                    int_v = int(v)
                    if 1 <= int_v <= 10:
                        valid_scores[k] = int_v
                    else:
                        # #FALLBACK : score out of range, recording as null
                        print(f" WARNING: score {k}={v} out of range, set to null")
                        valid_scores[k] = None
            return {"scores": valid_scores, "judge_raw": judge_text}
        else:
            # #FALLBACK : could not parse judge response
            print(f" WARNING: judge parse failed")
            return {"scores": {m: None for m in task["measures"]}, "judge_raw": judge_text}
    except Exception as e:
        print(f" WARNING: judge error: {e}")
        return {"scores": {m: None for m in task["measures"]}, "judge_raw": f"ERROR: {e}"}


def run_experiment(config: dict, model_filter: str | None = None, judge_filter: str | None = None, dry_run: bool = False):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    models = config["models"]
    if model_filter:
        models = [m for m in models if model_filter.lower() in m["id"].lower() or model_filter.lower() in m["name"].lower()]

    prompts = config["unit_prompts"]
    tasks = config["tasks"]
    reps = config.get("repetitions", 3)
    scoring_rubric = config.get("scoring", {})

    total_tests = len(models) * len(prompts) * len(tasks) * reps

    print(f"\n{'='*70}")
    print(f"UNIT PROMPT EXPERIMENT v7")
    print(f"{'='*70}")
    print(f"Models:       {len(models)}")
    print(f"Unit prompts: {len(prompts)}")
    print(f"Tasks:        {len(tasks)}")
    print(f"Repetitions:  {reps}")
    print(f"Total tests:  {total_tests}")
    print(f"Output:       {output_dir}/")
    print(f"{'='*70}\n")

    if dry_run:
        print("DRY RUN — plan:\n")
        for model in models:
            print(f"  Model: {model['name']} ({model['id']})")
        print()
        for prompt in prompts:
            print(f"  Prompt: [{prompt['id']}] {prompt['text'][:60]}...")
        print()
        for task in tasks:
            print(f"  Task: [{task['id']}] {task['name']}")
        print(f"\nTotal: {len(models)} × {len(prompts)} × {len(tasks)} × {reps} = {total_tests} tests")
        return

    # Pre-flight: check all models
    print("PRE-FLIGHT: checking model availability...")
    available_models = []
    for model in models:
        model_provider = model.get("provider", config.get("provider", "lmstudio"))
        if preflight_check(config, model["id"], model["name"], model_provider):
            available_models.append(model)

    if not available_models:
        print("\nERROR: No models available. Exiting.")
        sys.exit(1)

    print(f"\n{len(available_models)}/{len(models)} models available.\n")

    # Pre-flight: check judge
    judge_model_id = config.get("judge_model", "qwen/qwen3-coder-next")
    if judge_filter:
        matching = [m for m in config["models"] if judge_filter.lower() in m["id"].lower()]
        if matching:
            judge_model_id = matching[0]["id"]

    judge_provider = config.get("judge_provider", config.get("provider", "lmstudio"))
    print(f"JUDGE: {judge_model_id} (via {judge_provider})")
    judge_llm = None
    if preflight_check(config, judge_model_id, "Judge", judge_provider):
        judge_llm = create_model_llm(config, judge_model_id, judge_provider)
    else:
        # #FALLBACK : judge unavailable, will skip scoring
        print("WARNING: Judge unavailable. Responses will be recorded but not scored.")

    # Recalculate total with available models
    total_tests = len(available_models) * len(prompts) * len(tasks) * reps
    all_results = []
    test_count = 0
    errors = 0
    start_time = time.time()

    for model in available_models:
        model_id = model["id"]
        model_name = model["name"]
        print(f"\n{'─'*50}")
        print(f"MODEL: {model_name}")
        print(f"{'─'*50}")

        model_provider = model.get("provider", config.get("provider", "lmstudio"))
        llm = create_model_llm(config, model_id, model_provider)

        for prompt in prompts:
            prompt_id = prompt["id"]
            prompt_text = prompt["text"]
            system_prompt = prompt_text if prompt_text else None

            for task in tasks:
                task_id = task["id"]

                for rep in range(reps):
                    test_count += 1
                    elapsed_total = time.time() - start_time
                    rate = test_count / max(elapsed_total, 0.1)
                    eta_min = (total_tests - test_count) / max(rate, 0.01) / 60

                    print(f"  [{test_count}/{total_tests}] {prompt_id} × {task_id} r{rep+1} ", end="", flush=True)

                    result = run_single_test(llm, system_prompt, task["prompt"])

                    judge_data = {"scores": {}, "judge_raw": ""}
                    if result["status"] == "success" and judge_llm is not None and model_id != judge_model_id:
                        judge_data = score_response(judge_llm, task, result["response"], scoring_rubric)

                    record = {
                        "test_id": test_count,
                        "timestamp": datetime.now().isoformat(),
                        "temperature": config.get("temperature", 0.0),
                        "seed": config.get("seed", None),
                        "model_id": model_id,
                        "model_name": model_name,
                        "model_family": model.get("family", ""),
                        "model_architecture": model.get("architecture", ""),
                        "model_size": model.get("size", ""),
                        "prompt_id": prompt_id,
                        "prompt_text": prompt_text,
                        "prompt_category": prompt.get("category", ""),
                        "prompt_expected_direction": prompt.get("expected_direction", ""),
                        "system_prompt_used": system_prompt or "(none — baseline)",
                        "task_id": task_id,
                        "task_name": task.get("name", ""),
                        "task_prompt": task["prompt"],
                        "task_correct": task.get("correct_answer", task.get("correct_behavior", "")),
                        "repetition": rep + 1,
                        "status": result["status"],
                        "response": result["response"],
                        "elapsed_s": result["elapsed_s"],
                        "input_tokens": result.get("input_tokens"),
                        "output_tokens": result.get("output_tokens"),
                        "scores": judge_data["scores"],
                        "judge_raw_response": judge_data["judge_raw"],
                        "error": result.get("error"),
                    }

                    all_results.append(record)

                    if result["status"] == "success":
                        score_str = " ".join(f"{k}={v}" for k, v in judge_data["scores"].items() if v is not None)
                        print(f"OK {result['elapsed_s']}s {score_str}")
                    else:
                        errors += 1
                        print(f"ERR: {result.get('error', 'unknown')[:50]}")

                    if test_count % 10 == 0:
                        save_results(all_results, output_dir, tag="incremental")

        save_results(all_results, output_dir, tag=f"model_{model_name.replace(' ', '_')}")
        elapsed_min = (time.time() - start_time) / 60
        print(f"\n  Done. {test_count} tests, {errors} errors, {elapsed_min:.1f} min elapsed, ~{eta_min:.1f} min remaining")

    save_results(all_results, output_dir, tag="final")
    summary = generate_summary(all_results)
    summary_file = output_dir / f"summary_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    total_min = (time.time() - start_time) / 60
    print(f"\n{'='*70}")
    print(f"COMPLETE: {test_count} tests, {errors} errors, {total_min:.1f} minutes")
    print(f"Results: {output_dir}/")
    print(f"Summary: {summary_file}")
    print(f"{'='*70}")


def save_results(results: list, output_dir: Path, tag: str = ""):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_v7_{tag}_{ts}.json" if tag else f"results_v7_{ts}.json"
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def generate_summary(results: list) -> dict:
    from collections import defaultdict
    import statistics as stat_mod

    by_prompt = defaultdict(lambda: defaultdict(list))
    by_model = defaultdict(lambda: defaultdict(list))
    by_prompt_task = defaultdict(lambda: defaultdict(list))

    for r in results:
        if r["status"] != "success":
            continue
        for metric, score in r.get("scores", {}).items():
            if score is not None and score > 0:
                by_prompt[r["prompt_id"]][metric].append(score)
                by_model[r["model_name"]][metric].append(score)
                key = f"{r['prompt_id']}__{r['task_id']}"
                by_prompt_task[key][metric].append(score)

    def compute_stats(values):
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "mean": round(stat_mod.mean(values), 3),
            "stdev": round(stat_mod.stdev(values), 3) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
        }

    baseline_scores = {}
    for metric, scores in by_prompt.get("baseline", {}).items():
        if scores:
            baseline_scores[metric] = stat_mod.mean(scores)

    prompt_vs_baseline = {}
    for pid, metrics in by_prompt.items():
        if pid == "baseline":
            continue
        prompt_vs_baseline[pid] = {}
        for metric, scores in metrics.items():
            if metric in baseline_scores and len(scores) >= 2:
                baseline_mean = baseline_scores[metric]
                prompt_mean = stat_mod.mean(scores)
                prompt_vs_baseline[pid][metric] = {
                    "prompt_mean": round(prompt_mean, 3),
                    "baseline_mean": round(baseline_mean, 3),
                    "delta": round(prompt_mean - baseline_mean, 3),
                }

    return {
        "experiment": "Unit Prompt v7",
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(results),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] != "success"]),
        "models_tested": sorted(set(r["model_name"] for r in results)),
        "prompts_tested": sorted(set(r["prompt_id"] for r in results)),
        "tasks_tested": sorted(set(r["task_id"] for r in results)),
        "by_prompt": {pid: {m: compute_stats(s) for m, s in metrics.items()} for pid, metrics in by_prompt.items()},
        "by_model": {mid: {m: compute_stats(s) for m, s in metrics.items()} for mid, metrics in by_model.items()},
        "prompt_vs_baseline": prompt_vs_baseline,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unit Prompt Experiment v7")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--model", default=None, help="Filter models by name/id substring")
    parser.add_argument("--judge", default=None, help="Override judge model by name/id substring")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config, model_filter=args.model, judge_filter=args.judge, dry_run=args.dry_run)
