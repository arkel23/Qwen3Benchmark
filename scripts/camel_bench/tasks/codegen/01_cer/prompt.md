I need a small Python module to compute character error rate (CER) for our ASR evaluations. Standard library only.

Please implement three module-level functions:

1. `levenshtein(ref: str, hyp: str) -> int`
   Character-level edit distance (insertions, deletions and substitutions all cost 1). Either string may be empty.

2. `normalize(text: str, strip_punctuation: bool = False, collapse_whitespace: bool = False) -> str`
   - With both flags False, return `text` unchanged.
   - `strip_punctuation=True` deletes every character in `string.punctuation`.
   - `collapse_whitespace=True` replaces every run of whitespace (as matched by the regex `\s+`) with a single space and strips leading and trailing whitespace.
   - When both are set, punctuation is stripped first, then whitespace is collapsed.

3. `cer(refs: list[str], hyps: list[str], strip_punctuation: bool = False, collapse_whitespace: bool = False) -> float`
   - Normalise each reference and hypothesis with `normalize` using the given flags.
   - Return corpus-level CER: the sum of Levenshtein distances over all pairs divided by the total number of characters in the normalised references. This is not the mean of per-utterance CERs. The value can exceed 1.
   - Raise `ValueError` if `refs` and `hyps` have different lengths.
   - Raise `ValueError` if the normalised references contain zero characters in total.

Reply with a single Python code block containing the complete module.
