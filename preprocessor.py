import re

# ── Arabic normalization patterns ──────────────────────────────────────────────
_TASHKEEL   = re.compile(r"[ً-ٰٟٴ]")   # diacritics
_ALEF       = re.compile(r"[أإآٱ]")                         # alef variants → ا
_WAW_HAMZA  = re.compile(r"ؤ")                              # waw hamza → و
_YEH_HAMZA  = re.compile(r"ئ")                              # yeh hamza → ي
_TEH_MARBUTA = re.compile(r"ة")                             # teh marbuta → ه

# ── Noise removal patterns ─────────────────────────────────────────────────────
_URL        = re.compile(r"http\S+|www\S+")
_MENTION    = re.compile(r"@\w+")
_HASHTAG    = re.compile(r"#\w+")
# Keep: Arabic script, Latin letters, Franco-Arabe digits (3,7,9…), spaces
_NOISE      = re.compile(r"[^؀-ۿݐ-ݿa-zA-Z0-9\s]")
_MULTI_SPACE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    text = _TASHKEEL.sub("", text)
    text = _ALEF.sub("ا", text)
    text = _WAW_HAMZA.sub("و", text)
    text = _YEH_HAMZA.sub("ي", text)
    text = _TEH_MARBUTA.sub("ه", text)
    return text


def clean(text: str) -> str:
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(" ", text)
    text = _NOISE.sub(" ", text)
    return _MULTI_SPACE.sub(" ", text).strip()


def _split_sentences(text: str) -> list[str]:
    # Each newline is a natural sentence boundary (critical for tweet structure).
    # Secondary split on sentence-ending punctuation equivalents.
    lines = re.split(r"\n+", text)
    sentences = []
    for line in lines:
        parts = re.split(r"[.!?؟،]+", line)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def preprocess(texts: list[str]) -> list[list[str]]:
    """
    Returns a list of tokenized sentences.
    Each sentence is a list of string tokens with <s> / </s> boundary markers.
    """
    all_sentences = []
    for text in texts:
        text = normalize_arabic(text)
        text = clean(text)
        for raw_sent in _split_sentences(text):
            tokens = raw_sent.split()
            if len(tokens) < 2:
                continue
            all_sentences.append(["<s>"] + tokens + ["</s>"])
    return all_sentences
