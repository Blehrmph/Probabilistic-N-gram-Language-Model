import random
from ngram_model import NgramModel


def _candidates(model: NgramModel, w1: str, w2: str, top_unigrams: int = 100) -> list[str]:
    """
    Build candidate next-word set from:
      - words observed after trigram context (w1, w2)
      - words observed after bigram context (w2)
      - top-N most frequent unigrams (catches OOV contexts)
    """
    cands: set[str] = set()
    cands.update(model.trigrams.get((w1, w2), {}).keys())
    cands.update(model.bigrams.get(w2, {}).keys())
    cands.update(w for w, _ in model.unigrams.most_common(top_unigrams))
    cands.discard("<s>")
    return list(cands)


def _sample_next(
    model: NgramModel,
    w1: str,
    w2: str,
    strategy: str,
    blocked: set[str] | None = None,
) -> str:
    cands = _candidates(model, w1, w2)
    if not cands:
        return "</s>"

    scores = {w: model.probability(w, w1, w2) for w in cands}
    total  = sum(scores.values())
    if total == 0:
        return "</s>"

    if strategy == "greedy":
        # Filter recently used words to break repetition loops; fall back to
        # the full candidate set if filtering leaves nothing viable.
        if blocked:
            filtered = {w: p for w, p in scores.items() if w not in blocked}
            if filtered:
                scores = filtered
                total  = sum(scores.values())
        return max(scores, key=scores.get)

    # weighted random sampling
    r, cumulative = random.random() * total, 0.0
    for w, p in scores.items():
        cumulative += p
        if cumulative >= r:
            return w
    return cands[-1]


def generate(
    model: NgramModel,
    seed: str | None = None,
    max_words: int = 30,
    strategy: str = "sample",
    no_repeat_window: int = 4,
) -> str:
    """
    Generate a sequence of words.

    Parameters
    ----------
    seed             : optional seed phrase (1-2 words); random start if None
    max_words        : maximum output length (excluding <s>)
    strategy         : 'sample' for probabilistic sampling, 'greedy' for argmax
    no_repeat_window : greedy only — blocks words seen in the last N tokens
    """
    if seed:
        tokens = ["<s>"] + seed.strip().split()
    else:
        top_words = [w for w, _ in model.unigrams.most_common(500)
                     if w not in ("<s>", "</s>")]
        tokens = ["<s>", random.choice(top_words)]

    while len(tokens) - 1 < max_words:
        w1 = tokens[-2] if len(tokens) >= 2 else "<s>"
        w2 = tokens[-1]
        blocked = set(tokens[-no_repeat_window:]) if strategy == "greedy" else None
        next_word = _sample_next(model, w1, w2, strategy, blocked=blocked)
        if next_word == "</s>":
            break
        tokens.append(next_word)

    return " ".join(tokens[1:])  # strip leading <s>
