# telugu_postprocess.py

import math

def levenshtein(a: str, b: str) -> int:
    """Simple Levenshtein distance on Unicode strings."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # delete
                dp[i][j-1] + 1,      # insert
                dp[i-1][j-1] + cost  # substitute
            )
    return dp[-1][-1]


def build_vocab(word_list):
    """Turn a list of known correct words into a set or list."""
    return list(set(w.strip() for w in word_list if w.strip()))


def correct_word(word: str, vocab, max_ratio: float = 0.3) -> str:
    """
    Snap 'word' to closest vocab entry if normalized distance <= max_ratio.

    max_ratio ~ 0.3 means up to 30% of chars can be different.
    """
    if not word or not vocab:
        return word

    best = None
    best_dist = math.inf

    for v in vocab:
        d = levenshtein(word, v)
        if d < best_dist:
            best_dist = d
            best = v

    if best is None:
        return word

    norm = best_dist / max(len(word), len(best), 1)
    if norm <= max_ratio:
        return best
    return word
