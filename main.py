import random

from data_loader import load_corpus
from preprocessor import preprocess
from ngram_model import NgramModel
from evaluator import perplexity
from generator import generate

RANDOM_SEED = 42


def split(sentences, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(sentences)
    n = len(sentences)
    t = int(n * train_ratio)
    v = int(n * (train_ratio + val_ratio))
    return sentences[:t], sentences[t:v], sentences[v:]


if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    print("=" * 50)
    print("STEP 1 — Loading corpus")
    print("=" * 50)
    texts = load_corpus()

    # ── 2. Preprocess ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 2 — Preprocessing")
    print("=" * 50)
    sentences = preprocess(texts)
    print(f"Total sentences : {len(sentences):,}")

    train, val, test = split(sentences)
    print(f"Train           : {len(train):,}")
    print(f"Validation      : {len(val):,}")
    print(f"Test            : {len(test):,}")

    # ── 3. Train ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 3 — Training interpolated trigram model")
    print("=" * 50)
    model = NgramModel()
    model.train(train)

    # ── 4. Evaluate ────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 4 — Evaluation (perplexity)")
    print("=" * 50)
    # Cap train eval at 5 000 sentences to keep it fast
    train_pp = perplexity(model, train[:5_000])
    val_pp   = perplexity(model, val)
    test_pp  = perplexity(model, test)
    print(f"  Train perplexity      : {train_pp:,.2f}")
    print(f"  Validation perplexity : {val_pp:,.2f}")
    print(f"  Test perplexity       : {test_pp:,.2f}")

    # ── 5. Generate ────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 5 — Text generation")
    print("=" * 50)

    print("\n[Sampling — random start]")
    for i in range(5):
        print(f"  {i + 1}. {generate(model, strategy='sample')}")

    print("\n[Greedy — random start]")
    for i in range(3):
        print(f"  {i + 1}. {generate(model, strategy='greedy')}")

    # ── 6. Save ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 6 — Saving model")
    print("=" * 50)
    model.save("ngram_model.pkl")
