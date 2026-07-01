# Loan Default Risk Prediction — Dependent vs. Independent Scoring

A data-mining and machine-learning project that predicts whether a loan applicant
will repay their loan (**Fully Paid → 0**) or default (**Charged Off → 1**), and
studies a specific research question: *can a model judge credit risk fairly using
only the applicant's raw characteristics and behaviour, without leaning on
pre-computed bank scores such as FICO, loan grade and interest rate?*

The project ships two parallel XGBoost models and an interactive **Streamlit**
dashboard that runs them side by side on the same client, so a loan officer can
compare their verdicts, adjust the decision threshold, and simulate "what-if"
scenarios in real time.

> **Course:** Data Mining (Podatkovno rudarjenje), 2025/26
> **Authors / co-authors:** **Bojan Jakšić, Tilen Butara, Martin Lazar**

> The original project reports (`osnutek.md`, `VMESNO.md`, `KONČNO.md`) and the
> presentation outline (`powerpoint_outline.md`) are written in Slovenian. This
> README is an English summary of the complete project and of how the app works.

---

## 1. The problem and the core idea

In lending, one of the most important decisions is estimating an applicant's
creditworthiness. Traditional systems rely heavily on internal bank metrics —
the **FICO** score, the assigned loan **grade / sub-grade**, and the **interest
rate**. These can behave almost like a self-fulfilling prophecy: a low-scored
client is automatically given a high interest rate, which in turn makes the debt
harder to repay.

To probe this, we built and compared **two models**:

1. **Dependent model** — a standard model that uses *all* available attributes,
   including the bank-generated scores (`fico_avg`, `grade`, `sub_grade`,
   `int_rate`).
2. **Independent model** — a "blind test" from which those artificial bank scores
   were deliberately removed. It must decide using only raw, fundamental client
   properties and behaviour: annual income, **DTI** (debt-to-income), credit
   history, loan purpose, revolving balance, and text-derived NLP signals.

The question: *can machine learning identify default risk on its own from basic
client properties, without relying on predetermined internal bank scores?*

---

## 2. Data

We used the open **Lending Club** dataset (issued loans, 2007–2018, ~2.2M rows,
~150 columns), sourced from
[Kaggle — Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club).

Because the raw file (`data/raw/accepted_2007_to_2018Q4.csv`, ~1.2 GB, git-ignored)
is too large to model directly, we **stratified-sampled exactly 20,000 rows**
(`data/lending_club_20k.csv`), preserving the original class ratio
(~20 % defaults). Processing produced, in stages:

| File | Meaning |
|------|---------|
| `*_raw.csv` | after target filtering & column selection |
| `*_processed.csv` | after imputation, encoding and scaling |
| `*_selected.csv` | after feature selection |
| `*_final.csv` | after NLP features + K-Means cluster were merged in |

Key preprocessing steps:

- **Target construction:** keep only `Fully Paid` (0) and `Charged Off` (1);
  a strongly imbalanced binary label (~80 % / 20 %).
- **Cleaning / leakage control:** drop missing-heavy rows and any attribute that
  leaks post-approval (future) information.
- **Feature engineering:** `fico_avg` (mean of FICO range), `credit_history_years`
  (parsed from `earliest_cr_line`), `loan_to_income`, `installment_to_income`.
- **Encoding:** ordinal for `grade`/`sub_grade`, one-hot for ownership & purpose.
- **Scaling:** `StandardScaler` on all numeric columns; the fitted scaler is
  saved to `models/scaler.pkl` so the app can convert between z-scores and real
  units ($, %, FICO points) in production.

The **final modelling table** (`data/test_final.csv`) has **68 columns**:
`loan_status` (target) + **50 TF-IDF** word features + **17** numeric / encoded
features. The dependent model trains on 67 features (+ a `risk_cluster` slot); the
independent model uses **63** (the four bank-score columns removed).

---

## 3. Pipeline (Jupyter notebooks)

The work is split across numbered notebooks in `notebooks/`:

| Notebook | Phase |
|----------|-------|
| `00_data_loading.ipynb` | Load the ~1.2 GB source, filter target, select columns, stratified 20k sample, train/test split |
| `01_eda_preprocessing.ipynb` | EDA, missing-value imputation, feature engineering, encoding, `StandardScaler` |
| `02_pattern_mining.ipynb` | Apriori association-rule mining (`mlxtend`), Random-Forest feature selection, K-Means `risk_cluster` |
| `03_nlp_risk_scoring.ipynb` | TF-IDF on `emp_title` + HuggingFace sentiment on `desc` → `desc_sentiment_score`, `tfidf_*` features |
| `04_modeling.ipynb` | Dependent model: Logistic Regression vs. XGBoost, class-imbalance handling, evaluation |
| `04b_independent_modeling.ipynb` | Independent ("blind") model: same pipeline without FICO/grade/int_rate, threshold tuning |
| `05_explainability.ipynb` | SHAP global (beeswarm) + local (waterfall) explanations |
| `06_literature_comparison.ipynb` | Benchmarking our metrics against published studies |

### Pattern mining & NLP highlights
- **Apriori:** the dominant *defaulter* profile combines a **high interest rate**,
  **renter** status (no owned property), and a **debt-consolidation** loan purpose.
- **Feature selection** (Random Forest importances) reduced the space to ~19 core
  indicators, later expanded to the final 68 columns by the NLP vectors and the
  K-Means cluster feature.
- **NLP:** `TfidfVectorizer` on job titles plus a sentiment score on the free-text
  loan description add a subtle but real separating signal.

---

## 4. Models and results

Both saved models are XGBoost classifiers with class-imbalance weighting
(`scale_pos_weight`). Metrics below were re-verified on `data/test_final.csv`
(4,000 rows, 799 positives = 20.0 %) with the currently pinned library versions:

| Model | Features | ROC-AUC | Recall @ 0.50 | Recall @ 0.40 |
|-------|----------|:-------:|:-------------:|:-------------:|
| **Dependent** (`xgb_model.pkl`) | 68 incl. FICO/grade/int_rate | **0.699** | 0.597 | 0.765 |
| **Independent** (`xgb_independent_model.pkl`) | 63, blind to bank scores | **0.637** | 0.577 | 0.832 |

**The threshold story.** At the classic **0.50** threshold, the independent model
let too many defaulters through (recall ≈ 0.58). Lowering the decision threshold
to a more conservative **0.40** (and regularising XGBoost with `max_depth=3`,
`subsample=0.8`) forces the model to be more cautious, and recall on real
defaulters climbs to **0.83** — at the expected cost of more false positives
(lower precision/accuracy). The independent confusion matrix at 0.40 is:

```
                 Predicted 0   Predicted 1
Actual 0 (paid)      1057          2144
Actual 1 (default)    134           665      → recall = 665 / 799 ≈ 0.83
```

For context, the wider study (`04_modeling.ipynb`, and the interim report)
compared Logistic Regression, Random Forest and XGBoost. On this cleaned 20k
sample the simple **Logistic Regression** was surprisingly competitive
(AUC ≈ 0.700, recall ≈ 64 %) — both `lr_model.pkl` and `lr_independent_model.pkl`
are kept for reference — but the **app uses the XGBoost pair** as the production
models.

**Takeaway.** Internal bank scores are, unsurprisingly, very strong predictors.
Even so, with NLP features, a lowered threshold and attention to DTI, the
*independent* model becomes a usable, more neutral mechanism for rejecting the
riskiest applicants without relying on artificial bank scores.

### Explainability (SHAP)
Because ML models act like black boxes — problematic in finance — we use
**SHAP waterfall** plots (`05_explainability.ipynb`, and live in the app) to
explain individual decisions: **red** bars pushed the prediction toward *default*,
**blue** bars toward *safe*. This lets a bank employee explain *why* a specific
loan was flagged.

---

## 5. The Streamlit app (`app.py`)

`app.py` is the "bank supervisor" dashboard that presents the whole project
interactively. It runs the **dependent and independent models side by side** on
the same selected client.

Features:

1. **Client selection** — pick any test client by ID (0–3999); its real
   repayment outcome is shown for comparison.
2. **Threshold control** — a bank-strictness slider for the approve/reject
   threshold (default **0.40**).
3. **"What-if" analysis with real units** — sliders in actual `$` / `%` / FICO
   points (not raw z-scores). Values are converted through `models/scaler.pkl`,
   so you can raise DTI, lower FICO or change income and watch each model's
   confidence move. A live **monthly installment** is recomputed from loan
   amount, term and interest rate.
4. **NLP profile** — the TF-IDF words describing the selected client are shown.
5. **Live SHAP waterfall** — per-decision explanation rendered for *both* models.
6. **What-if history table** — save successive scenarios into a comparison table.

> **Note:** the prediction is driven by the columns the models were actually
> trained on (DTI, income, FICO, credit history, revolving balance, installment,
> sentiment, TF-IDF words). A few sliders (e.g. raw loan amount / term) are shown
> for context and feed the installment calculation.

The app was the project's final showcase. During the school grading period it was
temporarily reduced to a minimal fallback version; this repository restores it to
its **complete original dashboard** state.

---

## 6. Running the project

Requires **Python 3** and the packages in `requirements.txt`
(`streamlit`, `xgboost`, `shap`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`,
`seaborn`, `mlxtend`, `transformers`, `torch`, `joblib`, `jupyter`).

```bash
# 1. create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch the interactive dashboard
streamlit run app.py
```

Then open the local URL Streamlit prints (default http://localhost:8501).
The pre-trained models, scaler, feature lists and test data are already committed
under `models/` and `data/`, so the app runs without re-training. To reproduce the
full pipeline, run the notebooks in `notebooks/` in numerical order (this requires
the ~1.2 GB raw Lending Club CSV placed in `data/raw/`).

---

## 7. Repository structure

```
.
├── app.py                     # Streamlit dashboard (dependent vs. independent)
├── requirements.txt
├── notebooks/                 # 00 … 06 pipeline notebooks
├── data/                      # sampled + processed CSVs (raw/ is git-ignored)
│   ├── lending_club_20k.csv
│   ├── train_final.csv / test_final.csv
│   └── …
├── models/                    # trained models + scaler + feature lists (.pkl)
│   ├── xgb_model.pkl              # dependent XGBoost
│   ├── xgb_independent_model.pkl  # independent XGBoost
│   ├── lr_model.pkl / lr_independent_model.pkl
│   ├── scaler.pkl
│   ├── dependent_features.pkl / independent_features.pkl
├── figures/                   # confusion matrix, SHAP waterfall, dashboard shot
├── osnutek.md                 # project proposal (SL)
├── VMESNO.md                  # interim report (SL)
├── KONČNO.md                  # final report (SL)
└── powerpoint_outline.md      # presentation outline (SL)
```

---

## 8. Authors

- **Bojan Jakšić**
- **Tilen Butara**
- **Martin Lazar**

Data Mining course project, 2025/26. Dataset: Lending Club (public, via Kaggle).
