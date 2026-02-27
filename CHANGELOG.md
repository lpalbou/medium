# Changelog

## 2026-02-27
- Added top-level `README.md` to describe the multi-article repository structure and reduce navigation ambiguity.
- Updated `supervisor/README.md` to reflect the new folder layout and clarify correctness-first scoring in `completeness` and `diplomatic_honesty` to avoid verbosity bias.
- Increased task complexity and added new reasoning tasks in `supervisor/config-small.yaml` and `supervisor/config.yaml` to reduce ceiling effects on small models.
- Rescaled scoring to 1-10 and clarified judge guidance to enable finer-grained evaluation and better separation of prompt effects.
- Reduced repetitions to 2 in supervisor configs and updated totals in `supervisor/README.md`.
