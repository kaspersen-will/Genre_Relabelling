# Genre Labelling Beyond the Labels
## Auditing Music Taxonomies with Classification and Vector Similarity
### ML Foundations Group Project — Full Implementation Guide

**Dataset:** FMA (Free Music Archive) — develop on Small (8k), final run on Medium (25k), stretch goal Full (106k)  
**Task type:** Multi-class classification + unsupervised neighbourhood analysis  
**Core tools:** pandas, scikit-learn, Faiss / sklearn NearestNeighbors, SHAP  
**Deadline:** Pipeline & Report 28/05 | Presentation 21/05 | Poster 20/05

---

## Table of Contents

1. [Research Question and Project Framing](#1-research-question-and-project-framing)
2. [Dataset Selection and Loading](#2-dataset-selection-and-loading)
3. [Data Exploration and Preprocessing](#3-data-exploration-and-preprocessing)
4. [Feature Engineering](#4-feature-engineering)
5. [Data Partitioning and Leakage Prevention](#5-data-partitioning-and-leakage-prevention)
6. [Model Development](#6-model-development)
7. [Neighbourhood Analysis (Replacing Vector DB)](#7-neighbourhood-analysis-replacing-vector-db)
8. [Evaluation Methodology and Metrics](#8-evaluation-methodology-and-metrics)
9. [Hyperparameter Tuning](#9-hyperparameter-tuning)
10. [Model Interpretation and Feature Importance](#10-model-interpretation-and-feature-importance)
11. [Relabelling Candidate Analysis](#11-relabelling-candidate-analysis)
12. [Reflection and Failure Modes](#12-reflection-and-failure-modes)
13. [Report Writing Guide](#13-report-writing-guide)
14. [Week-by-Week Implementation Plan](#14-week-by-week-implementation-plan)
15. [Rubric Checklist](#15-rubric-checklist)

---

## 1. Research Question and Project Framing

### Primary question
> Do official subgenre labels in the FMA hierarchy reflect the underlying structure of musical feature space, or do embedding neighbourhoods reveal that the official taxonomy over-fragments some styles and mis-assigns others — suggesting candidate relabellings at the subgenre level?

### Why this framing matters for the rubric
The assignment explicitly rewards projects that embrace negative results. A model that struggles to classify subgenres is not a failure — it is evidence that the label system is internally inconsistent at that level of granularity. This means your worst-case scenario (low F1) becomes your most interesting finding, and you can discuss it rigorously rather than defensively. The subgenre framing strengthens this considerably: coarse top-level genres (Rock, Folk) are acoustically broad by design, so some confusion is expected and uninteresting. Subgenre confusion (Indie Rock clustering with Dream Pop, Ambient clustering with Drone) is a genuine finding about how the taxonomy was constructed.

### Task type declaration
This project is a **multi-class classification** problem as its primary ML task, with an unsupervised **nearest-neighbour retrieval** component used for post-hoc label analysis. You should state this explicitly in your notebook's first markdown cell and in the report introduction.

### The KNN design decision — state this upfront
One of the most important architectural choices in this project is that KNN is not treated as a throwaway baseline. It is chosen specifically because its classification mechanism — majority vote from the k most acoustically similar tracks — is identical to the neighbourhood consistency check that forms the analytical core of the project. This means KNN errors are not just model failures; they are label audit findings. Mention this in your notebook introduction so the reader understands why KNN appears in both the supervised modelling section and the neighbourhood analysis, and why that is deliberate rather than redundant. See Section 6.3 for the full argument and the recommended notebook markdown text.

### Scope statement (write this in your notebook)
You are not claiming to find the objectively "correct" genre labels. You are investigating whether the official **subgenre** label system is acoustically coherent — as measured by musical feature similarity in the FMA precomputed feature space — and using neighbourhood disagreements to generate candidate relabelling suggestions for ambiguous tracks. The claim is about label consistency, not ground truth.

---

## 2. Dataset Selection and Loading

### Which FMA split to use — three-tier compute strategy

Follow this progression deliberately. Do not jump ahead until the previous tier is fully working end-to-end.

**Tier 1 — Development: FMA Small**
- 8,000 tracks
- Using the hierarchical subgenre labels (not `genre_top`), you will get a subset of the 161-genre taxonomy filtered to subgenres with ≥50 tracks — expect roughly 10–20 usable subgenres on Small
- Balanced at the top-level, but unbalanced at the subgenre level — good for practising your imbalance handling
- All precomputed features included

Use Small for all development, debugging, and iteration. Every pipeline stage should be built and verified here first. The dataset is small enough that the full pipeline runs in minutes.

**Tier 2 — Final run: FMA Medium**
- 25,000 tracks
- With the subgenre strategy and ≥50 track filter, expect 30–50 usable subgenres — this is where the relabelling argument becomes genuinely rich
- Class imbalance is significant at this granularity, which strongly justifies macro F1 over accuracy and should be stated explicitly
- This is your primary submission result

**Tier 3 — Stretch goal: FMA Full (if time and compute allow)**
- 106,574 tracks, up to 161 subgenres after filtering
- Only attempt if Tier 2 is fully complete before the 28th
- On Colab free tier, hyperparameter tuning will likely time out; Colab Pro or local GPU needed
- Present as an additional experiment in the report, not the primary result

**Practical note:** The `SUBSET` and `MIN_GENRE_TRACKS` config variables in `pipeline.py` are the only things you change when moving between tiers. Everything else adapts automatically.

### Download and structure
```
fma_metadata/
├── tracks.csv       # track IDs, genre assignments (all levels), metadata
├── genres.csv       # genre hierarchy: title, parent, top_level for all 161 genres
├── features.csv     # 518 precomputed audio features for all tracks
└── echonest.csv     # optional: tempo, energy, danceability (not all tracks)
```

### Loading code structure — subgenre strategy
```python
import pandas as pd
import numpy as np
import ast

# Load genre hierarchy
genres_meta = pd.read_csv('fma_metadata/genres.csv', index_col=0)
# genres_meta columns: title, top_level, parent

# Load tracks metadata
tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0, 1])

# Filter to chosen subset
subset_mask = tracks['set', 'subset'] == 'small'  # or 'medium'
tracks_subset = tracks[subset_mask]

# Extract first (most specific) subgenre ID per track
def parse_first_genre(genre_str):
    try:
        ids = ast.literal_eval(str(genre_str))
        return int(ids[0]) if ids else None
    except Exception:
        return None

genre_ids = tracks_subset['track', 'genres'].apply(parse_first_genre)

# Map genre ID → subgenre name via genres.csv
y_raw = genre_ids.map(genres_meta['title']).dropna()

# Filter out rare subgenres (too few tracks for stable CV folds)
MIN_GENRE_TRACKS = 50
counts = y_raw.value_counts()
y_raw = y_raw[y_raw.isin(counts[counts >= MIN_GENRE_TRACKS].index)]

# Load and align features
features = pd.read_csv('fma_metadata/features.csv', index_col=0, header=[0, 1, 2])
features.columns = ['_'.join(col).strip() for col in features.columns.values]
common_idx = y_raw.index.intersection(features.index)
X_raw = features.loc[common_idx]
y_raw = y_raw.loc[common_idx]
```

### Justification to write in your notebook
State that FMA was chosen because: (1) it exceeds the 10,000-record threshold in the medium split; (2) it is publicly available and used in published MIR research; (3) precomputed features avoid raw audio processing, keeping the project within computational scope; (4) its 161-genre hierarchical taxonomy is directly relevant to the research question — we operate at the subgenre level rather than the coarse top-level to make the relabelling argument meaningful; (5) the `MIN_GENRE_TRACKS` filter is a principled decision documented explicitly, not arbitrary data cleaning.

---

## 3. Data Exploration and Preprocessing

### 3.1 Statistical summary
```python
# Shape and dtypes
print(X.shape)
print(y.value_counts())

# Feature summary stats
X.describe()
```

Write in your notebook: how many features there are (FMA Small has ~518 after loading), what the feature groups are (MFCC, chroma, spectral centroid, spectral bandwidth, spectral contrast, spectral rolloff, tonnetz, ZCR), and whether any are constant or near-zero variance.

### 3.2 Class distribution and imbalance
FMA Small is roughly balanced by design, but check anyway:
```python
import matplotlib.pyplot as plt
y.value_counts().plot(kind='bar', title='Genre Distribution')
plt.tight_layout()
plt.show()
```

If using the medium split, imbalance will be more pronounced. Report the imbalance ratio and justify your metric choices accordingly (F1-macro over accuracy).

### 3.3 Missing values
FMA features should have no NaNs, but verify:
```python
print(X.isnull().sum().sum())
```

If there are missing values (can happen with echonest features), document their location and justify imputation strategy. For MFCC-based features, median imputation by genre is most principled.

### 3.4 Outlier analysis
```python
from scipy import stats
z_scores = np.abs(stats.zscore(X))
print(f"Percentage of values beyond 3σ: {(z_scores > 3).mean().mean():.2%}")
```

Audio features can have genuine extreme values (e.g., a track with an unusually high spectral centroid may simply be a very high-pitched piece). Do not clip blindly — justify any decisions based on domain reasoning. Recommended approach: retain outliers but apply robust scaling (see below).

### 3.5 Feature correlations
With 518 features, full pairwise correlation is unwieldy. Instead:
```python
# Check within-group correlation for one feature group
mfcc_features = X['mfcc']
corr_matrix = mfcc_features.corr()
```

Visualise with a heatmap and note that MFCC coefficients are often correlated. This motivates PCA in the feature engineering step.

### 3.6 Visualising class separation
Before modeling, check whether genres are separable in 2D:
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_encoded, cmap='tab10', alpha=0.5, s=10)
plt.title('PCA projection coloured by genre')
plt.show()
```

This is important to include in the report as it gives visual evidence of how cleanly genres cluster before any modelling.

---

## 4. Feature Engineering

### 4.1 Feature flattening
FMA features have multi-level column names (feature group / statistic / number). Flatten them:
```python
X.columns = ['_'.join(col).strip() for col in X.columns.values]
```

### 4.2 Scaling — use RobustScaler
Given potential outliers in audio features:
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
# FIT ONLY ON TRAINING DATA — apply to all splits
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

Justify RobustScaler over StandardScaler: it uses median and IQR instead of mean and variance, making it less sensitive to the extreme values common in audio feature distributions.

### 4.3 Dimensionality reduction (optional but recommended)
518 features is high. PCA to 50–100 components retains most variance and reduces compute:
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=100, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```

Justify: this is a stated strategy from the course for managing computational constraints. Report the variance explained and state what you retained.

### 4.4 Label encoding
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# Store for reverting predictions
genre_names = le.classes_
```

### 4.5 Leakage prevention checklist
Write a markdown cell explicitly stating:
- RobustScaler: fit on `X_train` only, transform applied to val/test
- PCA: fit on `X_train` only, transform applied to val/test  
- LabelEncoder: fit on `y_train` only (class names should be fixed)
- No feature statistics computed on the full dataset before splitting

---

## 5. Data Partitioning and Leakage Prevention

### Split strategy
```python
from sklearn.model_selection import train_test_split

# First split: train+val vs test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Second split: train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.176, random_state=42, stratify=y_trainval
    # 0.176 of 0.85 ≈ 0.15 of total → 70/15/15 split
)
```

### Justification to write in your notebook
A 70/15/15 split is appropriate here because: (1) 8,000 tracks gives sufficient training data at 70%; (2) stratification ensures each genre appears proportionally in all splits; (3) a separate held-out test set (15%) is kept entirely unseen until final evaluation, preventing any optimism bias from repeated validation comparisons.

For small FMA, you may supplement with **5-fold cross-validation** on the training set for hyperparameter tuning, reporting both mean and standard deviation of performance.

---

## 6. Model Development

All models should be wrapped in sklearn `Pipeline` objects to ensure preprocessing and modelling are encapsulated together. This prevents leakage during cross-validation and makes the pipeline serialisable.

### 6.1 Trivial baseline — majority class predictor
```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train_scaled, y_train_enc)
y_pred_dummy = dummy.predict(X_val_scaled)

print(f"Dummy F1 (macro): {f1_score(y_val_enc, y_pred_dummy, average='macro'):.3f}")
```

This is your floor. Any meaningful model must beat this.

### 6.2 Classical baseline — Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

lr_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('pca', PCA(n_components=100, random_state=42)),
    ('clf', LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'))
])

lr_pipeline.fit(X_train, y_train_enc)
```

Justification: Logistic Regression with multinomial output is the standard linear baseline for multi-class classification. Its coefficients are interpretable, and it is fast enough to run with full features.

### 6.3 Classical baseline — K-Nearest Neighbours ⚠️ Central design decision

```python
from sklearn.neighbors import KNeighborsClassifier

knn_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('pca', PCA(n_components=100, random_state=42)),
    ('clf', KNeighborsClassifier(n_neighbors=15, metric='cosine'))
])

knn_pipeline.fit(X_train, y_train_enc)
```

**This model is the architectural spine of the entire project. Read this carefully.**

Every other project in this course will treat KNN as a throwaway baseline — something to implement, report a score for, and move past. In this project, KNN is doing something categorically different: it is the bridge between the supervised classification component and the neighbourhood analysis component, and this connection must be made explicit in your notebook and report.

Here is the logic that ties the whole project together:

- KNN classifies a track by asking: *what genre do the k most acoustically similar tracks belong to?*
- Your neighbourhood analysis asks: *does the official label agree with the genre the k nearest neighbours belong to?*

These are the same question. KNN is not a separate model sitting alongside the neighbourhood analysis — it **is** the neighbourhood analysis, applied as a classifier. When KNN predicts genre X for a track officially labelled genre Y, that is not just a misclassification: it is a concrete, model-supported argument that the track's acoustic identity belongs to genre X. The neighbourhood of that track in feature space disagrees with its official label.

This means every KNN error is a potential relabelling candidate, and the KNN confusion matrix is not just an evaluation artefact — it is a genre-audit result. When KNN systematically confuses Folk with Experimental at a high rate, that is a finding about those two genre labels, not merely a limitation of the model.

**What to write in your notebook markdown for this section:**

> KNN occupies a unique dual role in this pipeline. As a classifier, it serves as a required classical ML baseline. But its underlying mechanism — majority vote from the k nearest neighbours in scaled feature space — is identical to the neighbourhood consistency check we perform in Section 7. This is not a coincidence: we chose KNN specifically because its classification logic is interpretable as a label audit. A KNN misclassification is a case where the official label disagrees with the acoustic neighbourhood, which is the central quantity of interest in this project. The KNN confusion matrix therefore serves both as an evaluation table and as the first systematic inventory of genre label inconsistencies in the FMA Small dataset.

This paragraph, or something close to it, should appear word-for-word in your notebook. It is the argument that justifies the coherence of your project design and will be what sets your report apart.

### 6.4 Advanced model — Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

rf_pipeline.fit(X_train, y_train_enc)
```

Note: Random Forest does not require PCA since it is tree-based and handles high-dimensional data well. You can include PCA as an ablation study — compare with and without dimensionality reduction.

### 6.5 Advanced model — Gradient Boosting (XGBoost or sklearn's HistGradientBoosting)
```python
from sklearn.ensemble import HistGradientBoostingClassifier

gb_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', HistGradientBoostingClassifier(random_state=42, max_iter=200))
])

gb_pipeline.fit(X_train, y_train_enc)
```

Use `HistGradientBoostingClassifier` over standard `GradientBoostingClassifier` for speed. If your team is comfortable with XGBoost, it also works.

### 6.6 Optional advanced model — Multi-Layer Perceptron
```python
from sklearn.neural_network import MLPClassifier

mlp_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('pca', PCA(n_components=100, random_state=42)),
    ('clf', MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42))
])
```

Only include this if your team has time. It is not required but earns points under "advanced methods."

---

## 7. Neighbourhood Analysis (Replacing Vector DB)

This is the part of the project that makes it distinctive. You are replacing the vector database concept with two sklearn tools that are course-appropriate and fully adequate.

### 7.1 Building the neighbourhood index
```python
from sklearn.neighbors import NearestNeighbors

# Use the scaled+PCA-reduced features
nn_index = NearestNeighbors(n_neighbors=21, metric='cosine', algorithm='brute')
nn_index.fit(X_train_pca)
```

### 7.2 Neighbourhood purity per genre
Neighbourhood purity measures: for a given track, what fraction of its k nearest neighbours share the same official genre label?

```python
def compute_neighbourhood_purity(X_query, y_query, nn_index, y_index, k=20):
    distances, indices = nn_index.kneighbors(X_query)
    purities = []
    for i, (idx_row, label) in enumerate(zip(indices, y_query)):
        neighbor_labels = y_index[idx_row]
        purity = (neighbor_labels == label).mean()
        purities.append(purity)
    return np.array(purities)

# Compute on validation set
purity_scores = compute_neighbourhood_purity(
    X_val_pca, y_val_enc, nn_index, y_train_enc
)

# Per-genre purity
for g_idx, g_name in enumerate(genre_names):
    mask = y_val_enc == g_idx
    print(f"{g_name}: mean purity = {purity_scores[mask].mean():.3f}")
```

This gives you a table like:
| Genre | Mean Neighbourhood Purity |
|---|---|
| Hip-Hop | 0.82 |
| Folk | 0.71 |
| Experimental | 0.34 |

Genres with low purity are your most interesting findings — they indicate that the official label does not reflect musical coherence.

### 7.3 Top-k label agreement metric
For each track, look at the majority vote from its k nearest neighbours and compare to the official label:
```python
from scipy import stats as scipy_stats

def neighbour_vote_label(X_query, nn_index, y_index, k=20):
    distances, indices = nn_index.kneighbors(X_query)
    voted_labels = []
    for idx_row in indices:
        neighbor_labels = y_index[idx_row]
        voted_label = scipy_stats.mode(neighbor_labels, keepdims=True).mode[0]
        voted_labels.append(voted_label)
    return np.array(voted_labels)

voted = neighbour_vote_label(X_val_pca, nn_index, y_train_enc)
label_agreement = (voted == y_val_enc).mean()
print(f"Neighbour vote agreement with official label: {label_agreement:.2%}")
```

### 7.4 Relabelling candidates
Tracks where the neighbour majority vote disagrees with the official label:
```python
disagreement_mask = voted != y_val_enc

# Show examples
disagreement_df = pd.DataFrame({
    'track_id': X_val.index[disagreement_mask],
    'official_label': le.inverse_transform(y_val_enc[disagreement_mask]),
    'neighbour_vote': le.inverse_transform(voted[disagreement_mask])
})

print(disagreement_df.head(20))
```

These are your case studies for the report. Pick three or four vivid examples: a track labelled "Rock" whose neighbours are mostly "Folk," for instance, is a candidate for relabelling or a flag as a hybrid.

---

## 8. Evaluation Methodology and Metrics

### 8.1 Classification metrics
Since this is multi-class and potentially imbalanced (especially on the medium split), use:

```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

y_pred = rf_pipeline.predict(X_val)
y_prob = rf_pipeline.predict_proba(X_val)

# Full classification report
print(classification_report(y_val_enc, y_pred, target_names=genre_names))

# Macro F1 — primary metric
macro_f1 = f1_score(y_val_enc, y_pred, average='macro')

# ROC-AUC (one-vs-rest)
roc_auc = roc_auc_score(y_val_enc, y_prob, multi_class='ovr', average='macro')

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_val_enc, y_pred, display_labels=genre_names)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Why macro F1 is the primary metric:** It gives equal weight to each genre regardless of class size. Accuracy would be misleading if one genre dominates. This must be argued in your notebook and report.

### 8.2 Neighbourhood metrics
Report the following in a results table:

| Metric | Description |
|---|---|
| Mean neighbourhood purity | Average fraction of k nearest neighbours sharing the official label |
| Per-genre purity | The above broken down by genre |
| Neighbour vote agreement | Fraction of tracks where neighbour majority vote matches official label |
| Label disagreement rate | Fraction of tracks flagged as relabelling candidates |

### 8.3 Cross-validation
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    rf_pipeline, X_trainval, y_trainval_enc,
    cv=cv, scoring='f1_macro', n_jobs=-1
)

print(f"CV F1 macro: {scores.mean():.3f} ± {scores.std():.3f}")
```

Report both the mean and standard deviation. High variance indicates instability; low variance with consistent performance indicates a robust model.

### 8.4 Model comparison summary table
Produce this table for the report:

| Model | Val F1 (macro) | Val ROC-AUC | CV F1 mean ± std | Test F1 (macro) |
|---|---|---|---|---|
| Majority class dummy | | | | |
| Logistic Regression | | | | |
| KNN (k=15) | | | | |
| Random Forest | | | | |
| Gradient Boosting | | | | |

Fill this in from your actual results. Report the test set only once, for the selected final model.

### 8.5 Learning curves
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    rf_pipeline, X_trainval, y_trainval_enc,
    cv=5, scoring='f1_macro', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train F1')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation F1')
plt.fill_between(train_sizes,
                 val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
plt.xlabel('Training set size')
plt.ylabel('F1 macro')
plt.legend()
plt.title('Learning Curve — Random Forest')
plt.tight_layout()
plt.show()
```

---

## 9. Hyperparameter Tuning

### 9.1 Random Forest tuning
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'clf__n_estimators': [100, 200, 300, 500],
    'clf__max_depth': [None, 10, 20, 30],
    'clf__min_samples_split': [2, 5, 10],
    'clf__max_features': ['sqrt', 'log2', 0.3]
}

rf_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=param_dist,
    n_iter=30,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_search.fit(X_trainval, y_trainval_enc)
print(f"Best params: {rf_search.best_params_}")
print(f"Best CV F1: {rf_search.best_score_:.3f}")
```

### 9.2 KNN tuning
```python
knn_param_grid = {
    'clf__n_neighbors': [5, 10, 15, 20, 30],
    'clf__metric': ['cosine', 'euclidean', 'manhattan'],
    'clf__weights': ['uniform', 'distance']
}

from sklearn.model_selection import GridSearchCV

knn_search = GridSearchCV(
    knn_pipeline,
    param_grid=knn_param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

knn_search.fit(X_trainval, y_trainval_enc)
```

### 9.3 Justifying your tuning choices
Write in your notebook:
- Why RandomizedSearchCV over GridSearchCV for Random Forest: the parameter space is large (combinatorial) and exhaustive search would be computationally prohibitive; random search samples efficiently across the space.
- Why GridSearchCV for KNN: the parameter space is small and discrete, making exhaustive search feasible.
- State your `n_iter=30` budget and why it is sufficient: 30 samples explores the major regions of the space without being excessive.
- Always tune on the training+validation set using cross-validation, never on the held-out test set.

---

## 10. Model Interpretation and Feature Importance

### 10.1 Random Forest feature importance
```python
# Get feature names after pipeline steps
importances = rf_search.best_estimator_.named_steps['clf'].feature_importances_
feature_names = X_train.columns.tolist()

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Top 20
importance_df.head(20).plot(
    kind='barh', x='feature', y='importance',
    title='Top 20 Feature Importances — Random Forest'
)
plt.tight_layout()
plt.show()
```

### 10.2 SHAP values (advanced interpretation)
SHAP satisfies the rubric's advanced methods requirement:
```python
import shap

# TreeExplainer for Random Forest
explainer = shap.TreeExplainer(rf_search.best_estimator_.named_steps['clf'])
X_val_transformed = rf_search.best_estimator_[:-1].transform(X_val)
shap_values = explainer.shap_values(X_val_transformed)

# Summary plot — shows global feature impact per class
shap.summary_plot(shap_values, X_val_transformed,
                  feature_names=X_train.columns.tolist(),
                  class_names=genre_names)
```

If PCA is applied before the classifier, SHAP values are on the PCA components. In this case, use permutation importance instead, which operates in the original feature space:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf_search.best_estimator_, X_val, y_val_enc,
    n_repeats=10, random_state=42, scoring='f1_macro'
)

perm_df = pd.DataFrame({
    'feature': X_val.columns,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)
```

### 10.3 What to interpret
In your report and notebook, specifically discuss:
- Which audio feature groups (MFCC, chroma, spectral) are most predictive
- Whether temporal features (ZCR, spectral flux) carry genre signal
- Which genres rely on the same features (suggesting potential confusion zones)

---

## 11. Relabelling Candidate Analysis

This is the signature analytical contribution of your project.

### 11.1 Three-way label comparison
For each track in your validation set, compare:
1. Official genre label
2. Classifier prediction (Random Forest best model)
3. Neighbourhood majority vote (KNN / NearestNeighbors)

```python
comparison_df = pd.DataFrame({
    'official': le.inverse_transform(y_val_enc),
    'classifier': le.inverse_transform(y_pred_rf),
    'neighbour_vote': le.inverse_transform(voted_val)
})

# Tracks where all three agree — high confidence official labels
full_agreement = comparison_df.query('official == classifier == neighbour_vote')

# Tracks where classifier and neighbour agree but differ from official — relabelling candidates
candidate_relabel = comparison_df.query(
    'classifier == neighbour_vote and official != classifier'
)

# Tracks where all three disagree — highly ambiguous / hybrid
full_disagreement = comparison_df.query(
    'official != classifier and classifier != neighbour_vote'
)

print(f"Full agreement: {len(full_agreement)} tracks ({len(full_agreement)/len(comparison_df):.1%})")
print(f"Relabelling candidates: {len(candidate_relabel)} tracks")
print(f"Highly ambiguous: {len(full_disagreement)} tracks")
```

### 11.2 Case studies
Select at least three strong examples for your report. Good cases are:
- A track the model confidently predicts as genre X, whose neighbours are overwhelmingly genre X, but which is officially labelled genre Y
- A track whose neighbours are split roughly 50/50 between two genres — a genuine hybrid
- A genre (e.g., Experimental) whose tracks scatter across multiple regions of the feature space, suggesting it is not a coherent musical category at all

### 11.3 Genre-level confusion analysis
```python
# Which genre pairs are most commonly confused?
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_val_enc, y_pred_rf)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_norm, xticklabels=genre_names, yticklabels=genre_names,
            annot=True, fmt='.2f', cmap='Blues')
plt.title('Normalised Confusion Matrix — Random Forest')
plt.ylabel('Official Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()
```

The off-diagonal elements with highest values are the genre pairs that most frequently blur. These become your main argument for label inconsistency.

---

## 12. Reflection and Failure Modes

### What to expect and how to frame it
Subgenre classification from acoustic features alone is a genuinely hard problem — harder than top-level genre classification. F1 scores in the range of 0.35–0.60 for 30–50 subgenres on FMA Medium are realistic and expected. Do not be alarmed if your scores sit there. Instead, use this in your reflection:

> "The difficulty of subgenre prediction from acoustic features corroborates our central argument: subgenre labels in the FMA taxonomy are not purely acoustic constructs — they reflect cultural, historical, and community-tagging conventions that overlap considerably in the acoustic feature space. Our neighbourhood analysis reveals which subgenres are acoustically coherent and which are not, providing a more nuanced picture of taxonomy quality than classification accuracy alone."

### Likely failure modes to analyse
1. **Experimental and Noise subgenres** — catch-all categories with the lowest acoustic coherence by design. Expect the lowest neighbourhood purity and highest prediction error. Frame this as a label design problem: these genre names function as community taxonomy placeholders rather than acoustic descriptors.
2. **Subgenre pairs within the same top-level genre** — e.g., Indie Rock vs. Lo-Fi, Ambient vs. Drone, Bluegrass vs. Folk. The confusion matrix will show these blurring into each other. This is your strongest relabelling evidence: tracks at the boundary between two closely related subgenres are prime candidates for a suggested merge or reassignment.
3. **Cross-top-level confusion** — e.g., Acoustic Pop clustering with Folk subgenres. These are your most vivid case studies for the report since they cross the official top-level boundary entirely.

### Ethical and data limitations to mention
- FMA is a non-commercial dataset that skews toward independent/alternative music; major label pop and hip-hop are under-represented. Genre labels may reflect community tagging conventions rather than formal musicological categories.
- Genre is a social construct. A model trained on these labels inherits whatever biases the tagging community has.
- The project does not claim its relabelling suggestions are objectively "correct" — only that they are more consistent with acoustic similarity structure.

### Group process reflection (required in report)
Write a short section covering:
- How tasks were divided
- What tools you used (Colab, GitHub, shared notebooks)
- What you would do differently with more time (e.g., use raw audio + a CNN, experiment with multi-label classification, include lyric-based features)

---

## 13. Report Writing Guide

### Structure (within 2000 word limit)

**Title, team, dataset, GitHub repo link** — approx 50 words

**Problem statement** — approx 200 words  
State the research question at the subgenre level: you are investigating whether the FMA's 161-genre hierarchical taxonomy is acoustically coherent, or whether embedding neighbourhoods reveal systematic over-fragmentation and mis-assignment at the subgenre level. Explain why this matters — streaming recommendation systems, music information retrieval, playlist generation all depend on genre taxonomies being internally consistent. Note the dataset, the subgenre label strategy (hierarchical `genres` column via `genres.csv`, not the coarse `genre_top`), and the `MIN_GENRE_TRACKS` filter as a principled preprocessing decision.

**Pipeline overview** — approx 300 words  
Describe your pipeline stages: loading, subgenre label extraction, EDA, preprocessing, model stack, neighbourhood analysis. A pipeline diagram here is excellent and does not count toward the word limit. Include:
- The subgenre label extraction decision and why `genre_top` was deliberately avoided
- Preprocessing steps and leakage prevention
- Models used and rationale for each
- How the neighbourhood analysis connects to the supervised component via the KNN dual-role design

**Key results and model comparison table** — approx 400 words  
Present the comparison table. Discuss: which model performed best and why, how KNN performance connects to the neighbourhood purity results, what the cross-validation variance tells you about model stability at the subgenre level. Present the neighbourhood purity table broken down by subgenre — the range between highest and lowest purity subgenres is your central quantitative finding. Include the relabelling patterns table showing the most common official→suggested transitions.

**Failure and limitation analysis** — approx 300 words  
Analyse the Experimental genre specifically. Discuss the genre confusion pairs. Present two or three relabelling case studies with track IDs and your suggested alternatives.

**Reflection** — approx 250 words  
Ethical considerations, data limitations, what surprised you, what you would change.

**References** — not counted in word limit  
Cite: FMA paper (Defferrard et al., 2017), sklearn documentation, SHAP paper if used, any musicology papers you reference.

---

## 14. Week-by-Week Implementation Plan

| Week | Dates | Tasks | Owner suggestion |
|---|---|---|---|
| 1 | By ~5 May | Dataset download, loading script, EDA notebook, statistical summaries | EDA lead |
| 2 | By ~12 May | Preprocessing pipeline, all baselines running, KNN model | Modelling lead |
| 3 | By ~15 May | Advanced models, hyperparameter tuning, evaluation tables | Modelling lead |
| 4 | By ~18 May | Neighbourhood analysis, relabelling candidates, SHAP | Analysis lead |
| — | **20 May** | **Poster deadline** | All |
| — | **21 May** | **Presentation deadline** | All |
| 5 | By ~25 May | Report writing, final notebook clean-up, GitHub README | Report lead |
| — | **28 May** | **Pipeline + Report deadline** | All |

---

## 15. Rubric Checklist

Use this as a final check before submission.

### Data Exploration and Preprocessing (20 pts)
- [ ] Statistical summary of features and target distribution
- [ ] Visualisation of class distribution
- [ ] Missing value check with justification
- [ ] Outlier analysis with decision documented
- [ ] Correlation / feature group analysis
- [ ] PCA 2D visualisation of genre separation
- [ ] At least one transformation technique justified (RobustScaler + PCA)
- [ ] Clear documentation that scaler and PCA fit on training set only
- [ ] Stratified train/val/test split with proportions justified

### Model Complexity and Justification (20 pts)
- [ ] Trivial baseline (DummyClassifier) implemented
- [ ] Logistic Regression as classical linear baseline
- [ ] KNN implemented and conceptually linked to neighbourhood analysis
- [ ] Random Forest as advanced ensemble model
- [ ] Gradient Boosting as second advanced model
- [ ] Each model choice explicitly justified in markdown

### Evaluation Methodology and Rigor (20 pts)
- [ ] Macro F1 as primary metric with justification
- [ ] ROC-AUC (one-vs-rest) reported
- [ ] Per-class precision, recall, F1 via classification_report
- [ ] Confusion matrix visualised and interpreted
- [ ] Learning curves showing overfitting/underfitting diagnosis
- [ ] 5-fold CV with mean ± std reported
- [ ] Hyperparameter tuning with RandomizedSearchCV + GridSearchCV
- [ ] Test set evaluated only once at the end

### Code Quality and Reproducibility (10 pts)
- [ ] All cells run top-to-bottom without errors
- [ ] All models wrapped in Pipeline objects
- [ ] Random seeds set throughout (random_state=42)
- [ ] Markdown cells at every major section
- [ ] GitHub README includes clear run instructions
- [ ] No data leakage (verify scaler/PCA fit scope)

### Advanced Methods (10 pts)
- [ ] SHAP values or permutation importance implemented
- [ ] Neighbourhood analysis with NearestNeighbors
- [ ] Neighbourhood purity metric computed per genre
- [ ] Three-way label comparison (official / classifier / neighbour vote)
- [ ] Relabelling candidate identification

### Report Clarity (10 pts)
- [ ] Under 2000 words (excluding figures/tables/references)
- [ ] Submitted as PDF
- [ ] All required sections present
- [ ] Model comparison table included
- [ ] GitHub link in report

### Presentation (10 pts)
- [ ] Exactly 10 slides
- [ ] All team members speak
- [ ] Includes: problem, data, methods, results, failures, reflection
- [ ] Visual design is clear (not walls of text)
- [ ] Within 20 minutes

---

*This guide is intended to be used alongside the assignment brief, not as a replacement for it. All design decisions should be justified in your own words in the notebook.*