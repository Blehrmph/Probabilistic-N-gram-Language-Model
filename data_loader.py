import os

STORY_DIR = os.path.join("DATASET", "story-data")
TWITTER_DIR = os.path.join("DATASET", "twitter")


def _read_file(path, max_bytes=None):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read(max_bytes) if max_bytes else f.read()
    except FileNotFoundError:
        print(f"  Warning: {path} not found, skipping.")
        return ""


def load_story(max_files=2):
    files = sorted(
        f for f in os.listdir(STORY_DIR)
        if f.endswith(".txt") and not f.startswith("links")
    )
    texts = []
    for fname in files[:max_files]:
        path = os.path.join(STORY_DIR, fname)
        texts.append(_read_file(path))
        print(f"  Loaded: {fname}  ({os.path.getsize(path) / (1024 * 1024):.1f} MB)")
    return texts


def load_twitter(filename="tweets.txt", max_mb=50):
    path = os.path.join(TWITTER_DIR, filename)
    max_bytes = max_mb * 1024 * 1024
    content = _read_file(path, max_bytes=max_bytes)
    print(f"  Loaded: {filename}  (first {max_mb} MB)")
    return [content]


def load_corpus():
    print("Loading story-data (first 2 files)...")
    texts = load_story(max_files=2)
    print("Loading twitter data (first 50 MB)...")
    texts += load_twitter(max_mb=50)
    total_chars = sum(len(t) for t in texts)
    print(f"Total raw characters: {total_chars:,}")
    return texts
