# Fake vs Genuine Product Review Detector

> UW–Madison ECE — Final Project  
> **Team:** Brooks Harris (Undergraduate), Elice Priyadarshini (Graduate), Kaitlyn Yoo (Undergraduate)

## Overview

Deceptive product reviews harm buyers and honest sellers. This project builds a **fake vs genuine review detector** on the Amazon Fine Food Reviews corpus using three weak-labeling heuristics, a clean preprocessing pipeline, classical ML baselines, and a compact Transformer baseline (DistilBERT). The repository is organized around Jupyter notebooks that step from **data audit → preprocessing → models → explainability**.

---

## Team and Contribution

- **Brooks Harris** — TF‑IDF + Logistic Regression baseline, one‑layer MLP experiments, error analysis.  
  Email: `cbharris3@wisc.edu`
- **Elice Priyadarshini** — Data audit/EDA, project design, heuristic labeling, preprocessing pipeline, DistilBERT fine‑tuning, and explainability. Also led report writing.  
  Email: `epriyadarshi@wisc.edu`
- **Kaitlyn Yoo** — Data splits and repo organization, demo support, figures and editing.  
  Email: `gyoo7@wisc.edu`

---

## Notebooks

> Run notebooks in order (01 → 06). We are removing `07_streamlit_demo.ipynb`, so it is intentionally **not** documented here.

1. **`01_data_audit.ipynb` — Dataset audit and EDA**  
   - Loads the Amazon Fine Food Reviews dataset.  
   - Basic sanity checks (missing values, duplicates).  
   - Key distributions: star ratings, review length (words/chars), helpfulness votes.  
   - Notes on data skew and implications for evaluation metrics.

2. **`02_preprocessing.ipynb` — Cleaning and splits**  
   - Implements `clean_text()` (lowercase, strip HTML/URLs, remove non‑alpha, stopwords, lemmatize).  
   - Deduplication and filtering of near‑empty reviews.  
   - Defines **three labeling heuristics** and materializes labels:  
     - **HH** (Helpfulness Heuristic)  
     - **SP** (Sentiment Proxy)  
     - **BD** (Burst/Duplicate activity)  
   - Creates **stratified** train/val/test splits with a fixed seed; prevents text leakage across splits.  
   - Writes processed CSVs and summary figures to `data/processed/` and `figures/`.

3. **`03_baseline_tfidf_logreg.ipynb` — TF‑IDF + Logistic Regression**  
   - Vectorizer: `ngram_range=(1,2)`, `max_features=20_000`.  
   - Classifier: `LogisticRegression(class_weight="balanced", solver="saga")`.  
   - Hyperparameter sweep over `C ∈ {0.1, 1, 3}` via small 3‑fold CV.  
   - Evaluation: Accuracy, PR‑AUC, F1 for the **fake** class; confusion matrix and PR curve.  
   - Saves artifacts to `artifacts/tfidf_lr.joblib` and figures to `figures/`.

4. **`04_nn_tfidf.ipynb` — One‑layer MLP on TF‑IDF**  
   - Architecture: `20k → 512 ReLU → 2`, dropout, Adam.  
   - Early stopping on validation F1 (fake).  
   - Improves recall on borderline cases; compares against LR baseline.  
   - Saves model to `artifacts/mlp_tfidf.joblib` and figures to `figures/`.

5. **`05_distilbert_finetune.ipynb` — DistilBERT baselines**  
   - **Head‑only** quick baseline (2 epochs), then **small unfreeze** (last 2 blocks).  
   - Batch oversampling of the fake class (~1:1) and threshold tuning (e.g., τ=0.3) for higher recall.  
   - Uses Hugging Face `transformers` + `datasets`.  
   - Exports predictions, metrics, and plots; writes a confusion matrix to `figures/confusion_matrix_mix_from_splits.png`.

6. **`06_explainability.ipynb` — Model introspection**  
   - Token/feature contributions for LR and MLP (e.g., coefficient‑weighted n‑grams).  
   - Example‑level explanations and error slices.  
   - Guidance for iterative cleanup of heuristics and preprocessing.

---

## Data

- **Dataset:** Amazon Fine Food Reviews  
  - Kaggle page: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews  
  - Fields commonly used here: `Text`, `Score` (1–5★), helpfulness votes (`HelpfulnessNumerator`, `HelpfulnessDenominator`), `UserId`, `ProductId`, timestamps.
- Place raw data under `data/raw/` or update the paths inside `01_data_audit.ipynb`.

---

## Repo layout (recommended)

```
.
├── artifacts/                # saved models, vectorizers
├── configs/                  # optional YAMLs for runs
├── data/
│   ├── raw/                  # original CSV(s)
│   └── processed/            # cleaned/split data
├── figures/                  # plots used in the report
├── 01_data_audit.ipynb
├── 02_preprocessing.ipynb
├── 03_baseline_tfidf_logreg.ipynb
├── 04_nn_tfidf.ipynb
├── 05_distilbert_finetune.ipynb
├── 06_explainability.ipynb
└── README.md
```

---

## Environment

- Python 3.9+ (tested on modern 3.x)
- Suggested packages:
  - `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`
  - `nltk` (for stopwords/lemmatization; run the NLTK downloads in the notebook)
  - `transformers`, `datasets`, `torch`, `accelerate`, `evaluate`, `sentencepiece`
  - optional: `joblib`, `seaborn`, `lime`/`shap` for explainability

Create a fresh environment and install:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install numpy pandas scikit-learn matplotlib tqdm nltk             transformers datasets torch accelerate evaluate sentencepiece             joblib
```

GPU is optional; DistilBERT runs faster with a CUDA‑enabled GPU.

---

## Reproducing results

1. Run notebooks **01 → 06** in order.  
2. Keep the default random seed to match splits in the report.  
3. Trained artifacts and plots will be saved under `artifacts/` and `figures/`.  
4. The final confusion matrix from the mixture split is exported to `figures/confusion_matrix_mix_from_splits.png`.

---

## Notes on labeling

We use three weak‑label heuristics:  
- **HH:** helpfulness‑ratio and length cues with vote thresholds.  
- **SP:** 1–2★ as proxy fake, 4–5★ as proxy genuine, 3★ dropped.  
- **BD:** bursty user behavior across multiple products in short windows.

These are noisy by design; they provide scalable supervision and are combined with careful evaluation and error analysis.

---

## License & citation

- Dataset © the original providers; follow Kaggle’s licensing and terms.  
- Code in this repository is released for educational use.

If you use DistilBERT, please cite:
- Sanh et al., *DistilBERT: A Distilled Version of BERT*, 2019.

---

## Contact

Questions or issues: open an issue or email the team members listed above.
