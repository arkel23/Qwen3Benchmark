# Code repos (`edwin/projects/*`)

Training and inference code for published papers, with outside readers.

- **Never commit or push a code repo unless the user says so.** Keep the README and public
  entry points stable; prefer additive changes.
- Run from the repo root as `python -m tools.<dir>.<script>`; no `pip install -e`. If a
  package is both vendored and pip-installed, assert `pkg.__file__` is the copy you mean.
- New behaviour sits behind a flag that defaults to the old path, with its import inside
  that branch. Resolve options once at construction, not in the hot path.
- Nothing may change training except the seed. A config key that alters data filtering is a
  training change.
- Before moving a file to `deprecated/`, ask what capability disappears and check every
  consumer in `edwin/analysis/`, not just this repo.
- A re-run of the same experiment keeps the same serial.
- `llama.cpp` is an upstream checkout: read its `AGENTS.md`, never commit to it, rebuild
  rather than patch.

## Tests

Every new function or feature ships with a test; the repo's `tests/test_exemplar.py` is the
shape to copy. Run the CPU suite before claiming anything works.

- Flat arrange-act-assert in each test; no base classes or fixture stacks.
- One behaviour per test, named for the invariant (`test_frozen_encoder_receives_no_gradient`).
- `torch.testing.assert_close`, not `assert torch.allclose`; explicit tolerances.
- CPU-only, batch 2, no downloads, seconds per file; seed at the top of each test.
- Typical checks: shape/dtype/no NaN; flag off equals the old path; gradients reach exactly
  the trainable parameters; save-reload gives identical outputs.
