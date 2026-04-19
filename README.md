# IMDb Sentiment Classification — Model Comparison Study

A comparative study of three deep learning architectures for binary sentiment classification on the IMDb movie review dataset. Each notebook is self-contained and benchmarks a different combination of word embeddings and sequence model.

---

## Notebooks Overview

| Notebook | Embeddings | Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| `bert_transformer.ipynb` | BERT (`bert-base-uncased`) | BERT + Custom Transformer Head | **90.89%** | — | — |
| `glove_transformer.ipynb` | GloVe 6B 100d (frozen) | Transformer (from scratch) | 85.36% | 0.8569 | 0.9324 |
| `fasttext_transformer.ipynb` | FastText crawl-300d-2M (frozen) | Transformer (from scratch) | 87.80% | **0.8819** | **0.9520** |

**Dataset:** IMDb — 25,000 training reviews / 25,000 test reviews (binary: Positive / Negative)

---

## Project Structure

```
.
├── bert_transformer.ipynb        # BERT fine-tuning + Transformer head (GPU recommended)
├── glove_transformer.ipynb       # GloVe 100d + Transformer from scratch
├── fasttext_transformer.ipynb    # FastText 300d + Transformer from scratch
└── README.md
```

---

## What Each Notebook Covers

All three notebooks follow the same structured pipeline so results are directly comparable:

1. **Install & Import** — installs all dependencies in one cell
2. **Load & Clean IMDb Data** — decodes integer sequences back to text, then applies an NLTK pipeline (lowercase, strip HTML, remove punctuation, drop stopwords, keep words > 2 chars)
3. **Tokenize & Pad** — pads/truncates reviews to a fixed length; shows review length distribution
4. **Load Embeddings** — loads GloVe/FastText vectors and builds an embedding matrix (or uses BERT tokenizer)
5. **Build Model** — defines the Transformer encoder architecture in PyTorch
6. **Train** — trains with Adam optimizer; tracks loss and accuracy per epoch
7. **Evaluate** — classification report, confusion matrix, ROC curve & AUC
8. **Noise Robustness** — tests model degradation under three types of synthetic noise at 0–50% intensity:
   - *Character noise* — random character substitutions
   - *Word dropout* — randomly removes words
   - *OOV injection* — replaces words with out-of-vocabulary tokens
9. **Interpretability (LIME)** — explains individual predictions by identifying the words most influential to the model's decision (confident positives, confident negatives, wrong predictions, noisy inputs)
10. **Interpretability (t-SNE / Attention)** — GloVe & FastText notebooks visualize word vector space with t-SNE; the BERT notebook visualizes attention weights from the `[CLS]` token
11. **Final Summary** — F1-score heatmap across all test conditions

---

## Key Results

### Noise Robustness (F1-Score drop from clean baseline)

| Noise Type | BERT | GloVe Transformer | FastText Transformer |
|---|---|---|---|
| Char noise @ 50% | −0.0801 | −0.0859 | — |
| Word dropout @ 50% | −0.0656 | −0.0600 | — |
| OOV injection @ 50% | **−0.2986** | −0.0937 | — |

BERT is most sensitive to OOV injection because its subword tokenizer handles character noise gracefully but struggles when real words are replaced with unknown tokens at high rates. The GloVe and FastText Transformer models degrade more uniformly across all noise types.

---

## Requirements

### Common dependencies (all notebooks)
```
torch
tensorflow
numpy
matplotlib
seaborn
scikit-learn
nltk
lime
pandas
```

### BERT notebook only
```
transformers
```
> Requires a GPU (tested on NVIDIA T4). The notebook auto-installs all packages.

### GloVe notebook only
Download [GloVe 6B](https://nlp.stanford.edu/projects/glove/) and update the path:
```python
GLOVE_PATH = r'glove.6B\glove.6B.100d.txt'
```

### FastText notebook only
Download [FastText crawl-300d-2M](https://fasttext.cc/docs/en/english-vectors.html) and update the path:
```python
FASTTEXT_PATH = r'crawl-300d-2M.vec'
```

---

## Hyperparameter Reference

The Transformer architecture is shared across notebooks. Key parameters:

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `D_MODEL` | 128 / 768 (BERT) | 128–1024 | Hidden dimension; must be divisible by `N_HEADS` |
| `N_HEADS` | 4 / 8 (BERT) | 4–16 | Number of parallel attention patterns |
| `N_LAYERS` | 2 / 3 (BERT) | 1–6 | Depth of the Transformer encoder |
| `D_FF` | 256 / 1024 (BERT) | 256–2048 | Feed-forward layer dimension |
| `DROPOUT` | 0.3 / 0.2 (BERT) | 0.1–0.5 | Regularization |
| `MAX_LEN` | 300 / 256 (BERT) | up to 512 | Sequence length after padding |
| `BATCH_SIZE` | 64 / 32 (BERT) | 16–64 | Memory vs. throughput tradeoff |
| `EPOCHS` | 10 / 3 (BERT) | 2–10 | BERT converges faster |

---

## Takeaways

- **BERT achieves the highest accuracy (~91%)** through fine-tuning a pre-trained contextual model, but requires a GPU and significantly more compute.
- **FastText outperforms GloVe** on F1-score (0.882 vs. 0.857) and AUC-ROC (0.952 vs. 0.932) thanks to its higher-dimensional subword-aware vectors (300d vs. 100d).
- **All three models are robust to character noise and word dropout** but show steeper degradation under heavy OOV injection, especially BERT.
- **LIME explanations** confirm that all models correctly focus on sentiment-bearing words (e.g., *brilliant*, *terrible*) rather than neutral content words.

---

## Running the Notebooks

Each notebook is fully self-contained. Simply open in Jupyter or Google Colab and run all cells top to bottom. The first cell installs all required packages automatically.

```bash
jupyter notebook bert_transformer.ipynb
# or open directly in Google Colab
```
