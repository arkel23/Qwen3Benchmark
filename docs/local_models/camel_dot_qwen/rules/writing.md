---
description: Writing rules for papers, tables, captions, user-facing docs and reviews
paths:
  - "**/*.tex"
  - "**/*.bib"
  - "**/README.md"
  - "**/docs/**/*.md"
  - "**/paper/**"
  - "**/*caption*"
---
# Writing rules (papers, tables, captions, docs, reviews)

Order when rules compete: correct numbers with provenance, then the page limit, then plain
prose, then the field's terms, then completeness of caveats. The author owns every cut.

## Prose

- Plain words, one idea per sentence, about 20 words, never over 40 (measure with a script).
- No em-dashes, colon or semicolon chains, metaphors, aphorisms or rhetorical set-ups.
- No self-narration ("as noted above") and no script names, paths or serials in prose.
- Define every term and acronym before its first number; nothing used before it is introduced.
- One term per concept, the field's term over a coinage. Name every referent ("this effect" needs a noun).
- Nothing load-bearing in parentheses. No capitals or italics for emphasis; bold at most one
  answer sentence per paragraph.
- Claim only what was measured, at the granularity measured; audit every superlative.
- Describe the design that ran, not the one remembered.

## Numbers

- Every printed number is a macro written by a script that writes a CSV; a checker re-derives
  it from the CSV and fails on drift. Never type a number by hand.
- Store 6 dp, round once. Compute a printed delta from the printed operands.
- Name the aggregation (per run, per language, weighted) and the population in the sentence.
- Assert key uniqueness on every merge. Derive every list from one registry.
- A null claim needs a test. A strong trend at small n is a hypothesis.
- Training facts come from run records (`trainer_state.json`, wandb config), never from comments.

## Tables and figures

- Tables, captions and figures are generator output; never edit the artifact.
- Captions under 40 words, define every abbreviation and marker, and name what is excluded.
- After regenerating, render the file and read legend, axes and colours.

## Build

`pdflatex`, `bibtex`, `pdflatex` twice; zero errors and undefined references. Compiling is not
verifying: render every changed page (`pdftoppm`) and read it. Grep the PDF text for `[?]`.

## Documents

State the current state: no dates, no "previously", no bug stories. History goes to
`docs/CLAUDE_CHANGES.md`. Every named path or section must exist. A wording pass changes no
number, code span or table row.

## Reviews

Read in order, one paragraph or caption at a time; after each, flag every term, number or
rule not introduced earlier. Judge the whole only after the last unit. Verify every finding
against the data before acting; about a third survive.
