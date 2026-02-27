# Experiments for Medium Articles

This repository hosts experiment code and supporting data for multiple Medium articles. Each article lives in its own folder with its own config, runner, and README.

## Articles

- `supervisor/`: Unit Prompt Experiment v7 for **"Are You a Good Supervisor?"**

## Quick start (example)

```bash
cd supervisor
pip install -e .
python run_experiment.py --config config-small.yaml --dry-run
```

## Repository layout

```
repo_root/
  README.md
  supervisor/
    README.md
    run_experiment.py
    config.yaml
    config-small.yaml
    pyproject.toml
    results_v7_small/
    partial/
```

## Notes

- Output directories are relative to each article folder.
- See each article folder's `README.md` for details and interpretation guidance.
