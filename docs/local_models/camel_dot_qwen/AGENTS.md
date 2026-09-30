# Global rules — camel

Loaded in every Qwen Code session on this machine. Follow them exactly.

## Read more only when it applies

- Working in `/edahome/pcslab/pcs05/edwin/projects/*`: read `~/.qwen/guides/code_repos.md` first.
- Working in `/edahome/pcslab/pcs05/edwin/analysis/*`: read `~/.qwen/guides/analysis_repos.md` first.
- Writing rules (`~/.qwen/rules/writing.md`) load automatically when you open a `.tex`, `.bib`,
  README, `docs/` or `paper/` file. Reviewing a document without opening one: read it yourself.
- A repo's own `CLAUDE.md` holds its facts; read it before editing that repo. Its
  `/mnt/ssd2tb/...` and `/mnt/hdd10tb/...` paths are another machine; use the paths below.

## The machine

- 1x V100 32 GB, 40 cores, 78 GB RAM, CentOS 7 (glibc 2.17). **No sudo**: never propose
  `yum` or system changes. Login shell is tcsh; write scripts in bash (`#!/bin/bash`).
- **The GPU is held by the llama-server that runs you** (~31 GB). Do not start GPU jobs
  while this session is open: write the command and give it to the user. CPU runs are fine.
- Shared account `pcs05`: other people use `~` and its git identity. Work only in
  `/edahome/pcslab/pcs05/edwin/`. Never kill a process you did not start.

## Where things go

| path | holds |
|---|---|
| `edwin/projects/` | code repos |
| `edwin/analysis/` | per-paper analysis repos |
| `edwin/data/` | image datasets |
| `edwin/<job>/` | anything a job downloads or writes, log beside outputs |
| `edwin/deprecated/<job>_<YYYYMMDD>/` | retired files awaiting deletion, with a README |
| `~/.cache/huggingface`, `~/.cache/torch` | model caches (defaults; do not set `HF_HOME`) |

`edwin` means `/edahome/pcslab/pcs05/edwin`. Everything is on one NFS mount; `/` has 8 GB,
never write there. Run `df -h /edahome` before a download over 10 GB; under 200 GB free,
stop and tell the user. Nothing is backed up: only pushed commits survive.

## Environments

Start every script with
`source /edahome/pcslab/pcs05/miniconda3/etc/profile.d/conda.sh && conda activate <env>`.
`fgir_2x` (Python 3.12, torch 2.4.1+cu121, timm 0.9.12) runs the FGIR repos;
`llamacpp-build` only builds llama.cpp. Pin versions on install, re-pin anything the install
could move in the same command, and never upgrade torch or timm to fix an import.

## Running jobs

`nohup <cmd> > edwin/<job>/run.log 2>&1 &`, printing progress. A dropped session does not
kill a job: check `ps` before relaunching. Set `WANDB_MODE=disabled` for test runs.

## Before claiming anything

- Run it. Code that was not executed is not "working"; check the output file exists and has
  the property claimed (row count, unique values), not that the script exited 0.
- Never trust a helper's report of success; measure its effect. A no-op that "succeeds" is
  the most expensive bug.
- One failure proves nothing; reproduce it. An empty query is not a pass: assert the count.
- Report exactly what you checked and what you did not.
- When told to stop, stop and report the state.

## Making changes

- Do exactly the stated scope. Prefer a new file to editing an existing one; ask before
  editing a file you did not create.
- Never delete: move to `edwin/deprecated/<job>_<YYYYMMDD>/` with a README line (what, why,
  where the kept copy is).
- Match the surrounding code: its names, comment density, guard style. No comment on every
  step, no docstrings narrating history, no guards for impossible states.
- Comments state invariants in at most two lines. Never hardcode absolute paths in code;
  derive them (`Path(__file__).resolve().parents[N]`).
- Check line endings with `file` before editing; keep CRLF files CRLF.
- One-off commands go in the repo's `claude_process.sh`, dated, commented out once run.

## Git

- Commit as the user, never the account default:
  `git -c user.name=arkel23 -c user.email=edwinarkel.rios@gmail.com commit ...`
- One commit per idea; read `git diff --cached` first. No AI-attribution trailers. Never
  commit an IP address, user@host or credential. Never rewrite pushed history.
- Work on `main` unless told otherwise; `git fetch` before pushing.

## Finishing

Before reporting a job done: clear its scratch (move it to `deprecated/`), but never mid-job.
Then say what changed, where it is, and the next command. Be concise: plain words, no filler.
