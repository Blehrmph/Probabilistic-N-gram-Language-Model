import math
from ngram_model import NgramModel

_FLOOR = 1e-10  # log-probability floor for unseen n-grams


def perplexity(model: NgramModel, sentences: list[list[str]]) -> float:
    """
    Compute perplexity of the model over a list of tokenised sentences.
    Evaluation starts at position 2 (first full trigram context).
    """
    total_log2_p = 0.0
    total_tokens = 0

    for sent in sentences:
        sent = model.unk_sentence(sent)
        if len(sent) < 3:
            continue
        for i in range(2, len(sent)):
            p = model.probability(sent[i], sent[i - 2], sent[i - 1])
            total_log2_p += math.log2(p if p > 0 else _FLOOR)
            total_tokens += 1

    if total_tokens == 0:
        return float("inf")
    return 2 ** (-total_log2_p / total_tokens)
