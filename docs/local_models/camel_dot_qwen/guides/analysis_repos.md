# Analysis repos (`edwin/analysis/*`)

One repo per paper: metrics, figures, tables and the paper build.

- **Duplicate, don't share.** Never import from a sibling repo or extract shared code.
  Shared raw data is copied as a frozen snapshot.
- One `bash plotter.sh` from a bare checkout rebuilds everything: data, analyses, figures,
  tables, numbers, PDF, checks. A step run by hand is not done until it is in `plotter.sh`.
- GPU output goes to `edwin/<job>/`; only the small final CSVs the paper reads are copied
  into the repo, committed and pushed.
- Documents: `README.md`, `docs/FINDINGS.md` (one entry per `plotter.sh` command, nulls
  included), `docs/SERIALS.md`, `docs/NEXT_STEPS.md` (open items only). They state the
  current state; history goes in `docs/CLAUDE_CHANGES.md`.
- Every printed number comes from a script that writes a CSV, and a checker re-derives it.
- Commit and push to `main`.
