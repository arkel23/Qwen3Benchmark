import re
import string

_PUNCT = str.maketrans("", "", string.punctuation)


def levenshtein(ref: str, hyp: str) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def normalize(text: str, strip_punctuation: bool = False, collapse_whitespace: bool = False) -> str:
    if strip_punctuation:
        text = text.translate(_PUNCT)
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def cer(refs, hyps, strip_punctuation=False, collapse_whitespace=False) -> float:
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps differ in length")
    scores = []
    for ref, hyp in zip(refs, hyps):
        ref = normalize(ref, strip_punctuation, collapse_whitespace)
        hyp = normalize(hyp, strip_punctuation, collapse_whitespace)
        if ref:
            scores.append(levenshtein(ref, hyp) / len(ref))
    if not scores:
        raise ValueError("references contain no characters")
    return sum(scores) / len(scores)
