# Probabilistic N-gram Language Model for Darija

A simple probabilistic trigram language model with linear interpolation, trained on a Darija (Moroccan Arabic) corpus curated from multiple online sources.

## Dataset

The corpus is sourced from the **Darija Open Dataset**, curated from multiple online sources covering a wide range of registers and domains of Moroccan Arabic (Darija).

**Download link:** [Darija Corpus on OneDrive](https://alakhawayn365-my.sharepoint.com/personal/a_mourhir_aui_ma/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fa%5Fmourhir%5Faui%5Fma%2FDocuments%2FDarija%2Fdata%2Erar&parent=%2Fpersonal%2Fa%5Fmourhir%5Faui%5Fma%2FDocuments%2FDarija&ga=1)

Once downloaded, extract the contents into a `DATASET/` directory at the project root.

### Corpus Sources

| Source | Size | Description |
|---|---|---|
| `darija-wiki/` | ~4.6 MB | Darija Wikipedia articles |
| `goud.ma/` | ~3.4 MB | Moroccan news articles and comments |
| `music-data/` | ~4.0 MB | Song lyrics in Darija |
| `story-data/` | ~378 MB | Narrative stories and dialogues |
| `twitter/` | ~389 MB | Tweets in Darija and mixed Arabic-French |
| `Youtube/` | ~4.0 MB | YouTube comments |

**Total corpus size:** ~782 MB across 42 text files.

### Data Used for Training

Out of the full corpus, this model trains on:
- **story-data**: first 2 files (~50 MB of narrative Darija)
- **twitter**: first 50 MB of `tweets.txt` (natural, informal Darija)

This gives ~100 MB of training data, balancing coverage and training speed.

> **Note:** The `DATASET/` directory and all `.txt` files are excluded from version control via `.gitignore`. Download the data separately using the link above.

## Project Structure

```
Probabilistic-N-gram-Language-Model/
├── data_loader.py       # Load and sample story-data and twitter files
├── preprocessor.py      # Arabic normalization, cleaning, and tokenization
├── ngram_model.py       # N-gram counting and interpolated probability model
├── evaluator.py         # Perplexity computation
├── generator.py         # Text generation (greedy and random sampling)
├── main.py              # End-to-end pipeline
├── ngram_model.pkl      # Saved model (generated after running main.py)
├── DATASET/             # Corpus data (not tracked in git)
└── README.md
```

## Implementation

### 1. Data Loading (`data_loader.py`)
Loads the first 2 story files and the first 50 MB of Twitter data. Large files are read partially using a byte-limit to keep memory usage manageable.

### 2. Preprocessing (`preprocessor.py`)
- Diacritic (tashkeel) removal: `ً ٌ ٍ َ ُ ِ ...`
- Arabic letter normalization: `أ إ آ ٱ` → `ا`, `ؤ` → `و`, `ئ` → `ي`, `ة` → `ه`
- Noise removal: URLs, mentions, hashtags, non-Arabic/Latin characters
- Franco-Arabe digits preserved (3, 7, 9 represent Arabic sounds in Latin-script Darija)
- Sentence segmentation on newlines and punctuation (`؟ ، . ! ?`)
- Word tokenization with `<s>` / `</s>` boundary markers per sentence

### 3. N-gram Model (`ngram_model.py`)
Builds unigram, bigram, and trigram frequency tables from the training sentences, then computes the interpolated probability:

```
P(w | w₁, w₂) = λ₁·P(w) + λ₂·P(w|w₂) + λ₃·P(w|w₁,w₂)
```

Lambda weights (λ₁ + λ₂ + λ₃ = 1) are estimated from the training data using **deleted interpolation** (Jelinek & Mercer, 1980): for each observed trigram event, one occurrence is temporarily removed and the adjusted MLE of each order is compared; weight accumulates toward whichever order wins. This lets the data decide how much each order contributes rather than using fixed hand-tuned values.

### 4. Evaluation (`evaluator.py`)
Perplexity is computed on held-out data using an 80/10/10 train/validation/test sentence-level split:

```
PP = 2^( -1/N · Σ log₂ P(wᵢ | wᵢ₋₂, wᵢ₋₁) )
```

Lower perplexity = better model fit.

### 5. Text Generation (`generator.py`)
Given an optional seed phrase, the model scores candidate next words drawn from the trigram context, bigram context, and top unigrams, then picks the next word by either:
- **Greedy**: always picks the highest-probability word
- **Sampling**: draws proportionally to the probability distribution

## Requirements

```
python >= 3.10
```

No external libraries required. Built entirely on the Python standard library (`collections`, `math`, `re`, `pickle`).

## Usage

```bash
# Run the full pipeline: train, evaluate, generate, save model
python main.py
```

To load a saved model and generate text interactively:

```python
from ngram_model import NgramModel
from generator import generate

model = NgramModel.load("ngram_model.pkl")
print(generate(model, seed="واش نتا", max_words=20, strategy="sample"))
```

## Language Notes

Darija (المغربية / Darija) is the colloquial Arabic spoken in Morocco. It differs significantly from Modern Standard Arabic and incorporates influences from Amazigh (Tamazight), French, and Spanish. The Twitter data in particular contains Franco-Arabe, Darija written in Latin script with digits substituting for sounds absent in the Latin alphabet (e.g., `3` for `ع`, `7` for `ح`, `9` for `ق`). Both scripts are preserved during preprocessing.

## Credits

Dataset curated by [Prof. A. Mourhir](mailto:a_mourhir@aui.ma) at Al Akhawayn University in Ifrane, Morocco. (Prof. Mourhir, if you see this pls make your exams easier :'))
