# Generated results

This directory contains local JSONL, CSV, Markdown and SVG artifacts. Raw outputs are ignored by Git so they do not accidentally publish evaluation prompts or responses. The README at the repository root preserves a small, clearly labelled summary of the initial demo measurements.

Initial verified runs:

- `demo-profile.jsonl`: 240 mock jobs across six profile configurations.
- `demo-profile-summary.md`: mock profile table; RAM and quality intentionally absent.
- `demo-cache-*.csv/.md` and `demo-cache-selection.json`: plumbing-only hash-embedding sweep.
- `demo-load.jsonl` / `demo-load.md`: 1,600 loopback streaming requests, 400 at each concurrency level.
- `demo-failures.jsonl` / `demo-failures.md`: 400 seeded mixed-fault requests with fallback off and another 400 with it on.

Use distinct file names for real runs. Do not interpret demo model labels, hash-vector similarities or mock timing as evidence of model quality, semantic correctness or production capacity. See `docs/evaluation.md` for full commands.
