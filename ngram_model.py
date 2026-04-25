import pickle
from collections import defaultdict, Counter


class NgramModel:
    """
    Trigram language model with linear interpolation.

    P(w | w1, w2) = λ3·P_mle(w|w1,w2) + λ2·P_mle(w|w2) + λ1·P_mle(w)

    Lambda weights are estimated from training data via deleted interpolation
    (Jelinek & Mercer, 1980), so the corpus itself decides each order's contribution.
    """

    def __init__(self):
        self.unigrams: Counter = Counter()
        self.bigrams: dict[str, Counter] = defaultdict(Counter)
        self.trigrams: dict[tuple, Counter] = defaultdict(Counter)
        self.vocab: set = set()
        self.N: int = 0                        # total token count
        self.lambdas: tuple = (0.1, 0.3, 0.6) # (λ_uni, λ_bi, λ_tri) — overwritten after training

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, sentences: list[list[str]]) -> None:
        print("  Counting n-grams...")
        for sent in sentences:
            self.N += len(sent)
            for i, w in enumerate(sent):
                self.vocab.add(w)
                self.unigrams[w] += 1
                if i >= 1:
                    self.bigrams[sent[i - 1]][w] += 1
                if i >= 2:
                    self.trigrams[(sent[i - 2], sent[i - 1])][w] += 1

        print(f"  Vocabulary    : {len(self.vocab):,} types")
        print(f"  Tokens        : {self.N:,}")
        print(f"  Bigram ctxs   : {len(self.bigrams):,}")
        print(f"  Trigram ctxs  : {len(self.trigrams):,}")

        print("  Estimating λ weights via deleted interpolation...")
        self._estimate_lambdas()
        l1, l2, l3 = self.lambdas
        print(f"  λ = (unigram={l1:.3f}, bigram={l2:.3f}, trigram={l3:.3f})")

    def _estimate_lambdas(self) -> None:
        """
        Deleted interpolation: for every observed trigram event, temporarily
        remove one count and see which order gives the highest adjusted MLE.
        Accumulate mass toward that order's lambda, then normalise.
        """
        l1 = l2 = l3 = 0.0

        for (w1, w2), w3_counts in self.trigrams.items():
            c_w1w2 = sum(w3_counts.values())          # C(w1, w2)
            c_w2   = sum(self.bigrams[w2].values())   # C(w2, *)

            for w3, c123 in w3_counts.items():
                if c123 == 0:
                    continue
                c_w2w3 = self.bigrams[w2][w3]         # C(w2, w3)
                c_w3   = self.unigrams[w3]             # C(w3)

                # Adjusted MLE after removing one occurrence of (w1,w2,w3)
                t3 = (c123  - 1) / (c_w1w2 - 1) if c_w1w2 > 1 else 0.0
                t2 = (c_w2w3 - 1) / (c_w2  - 1) if c_w2   > 1 else 0.0
                t1 = (c_w3   - 1) / (self.N  - 1) if self.N  > 1 else 0.0

                best = max(t1, t2, t3)
                if best == t3:
                    l3 += c123
                elif best == t2:
                    l2 += c123
                else:
                    l1 += c123

        total = l1 + l2 + l3
        if total > 0:
            self.lambdas = (l1 / total, l2 / total, l3 / total)

    # ── MLE probabilities ──────────────────────────────────────────────────────

    def p_unigram(self, w: str) -> float:
        return self.unigrams.get(w, 0) / self.N if self.N else 0.0

    def p_bigram(self, w: str, prev: str) -> float:
        c_prev = sum(self.bigrams[prev].values())
        return self.bigrams[prev].get(w, 0) / c_prev if c_prev else 0.0

    def p_trigram(self, w: str, w1: str, w2: str) -> float:
        c_ctx = sum(self.trigrams[(w1, w2)].values())
        return self.trigrams[(w1, w2)].get(w, 0) / c_ctx if c_ctx else 0.0

    # ── Interpolated probability ───────────────────────────────────────────────

    def probability(self, w: str, w1: str, w2: str) -> float:
        """
        Interpolated P(w | w1, w2).
        w1 = word two positions back, w2 = word one position back.
        """
        l1, l2, l3 = self.lambdas
        return (l1 * self.p_unigram(w) +
                l2 * self.p_bigram(w, w2) +
                l3 * self.p_trigram(w, w1, w2))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "ngram_model.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}")

    @staticmethod
    def load(path: str = "ngram_model.pkl") -> "NgramModel":
        with open(path, "rb") as f:
            return pickle.load(f)
