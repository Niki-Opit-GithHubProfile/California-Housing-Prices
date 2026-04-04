# California Housing Prices: Data Preparation Report

## Abstract
This report documents an end-to-end exploratory data analysis and preprocessing workflow for the California Housing dataset. The work emphasizes reproducibility, leakage-safe transformations, and export of versioned artifacts for downstream modeling. The pipeline includes missing-value treatment, feature engineering, stratified splitting, outlier handling, categorical encoding, scaling, skewness correction, and diagnostic analysis (correlation and feature-importance preview).

## 1. Introduction
The goal of this project is to prepare a high-quality modeling dataset from raw housing records while preserving methodological rigor. The implemented workflow is located in [notebooks/data_exploration.ipynb](notebooks/data_exploration.ipynb) and uses [data/raw/housing.csv](data/raw/housing.csv) as the immutable source.

### 1.1 Objectives
- Perform median imputation of `total_bedrooms`.
- Create meaningful ratio features.
- Encode the categorical variable `ocean_proximity`.
- Apply feature scaling.
- Handle outliers using 1st/99th percentile thresholds.
- Create a stratified train-test split using income.
- Build a reusable preprocessing pipeline.
- Apply log transformations to skewed features.
- Run correlation analysis and feature-importance preview.
- Save versioned processed datasets and metadata.

## 2. Data and Experimental Setup

### 2.1 Dataset
- Source file: [data/raw/housing.csv](data/raw/housing.csv)
- Total rows: 20640
- Total columns: 10
- Target variable: `median_house_value`

### 2.2 Environment and Reproducibility Controls
- Dependencies are pinned in [requirements.txt](requirements.txt).
- Random seed: `42`
- All learned preprocessing statistics are derived from training data only.

## 3. Methodology

### 3.1 Initial Data Quality Audit
The workflow begins with checks for:
- Data types and summary statistics
- Missing values
- Duplicate rows
- Category frequencies for `ocean_proximity`

Observed issue:
- Missing values in `total_bedrooms`: 207 records

### 3.2 Stratified Train-Test Split
To reduce sampling bias, stratification is based on `median_income` using quantile bins (`pd.qcut`) with labels:
- `very_low`, `low`, `mid`, `high`, `very_high`

Split configuration:
- Test size: 0.20
- Method: `StratifiedShuffleSplit`
- Seed: `42`

### 3.3 Missing Value Imputation
`total_bedrooms` is imputed using the training median and then applied to both train and test.

Imputation value:
- `433.0`

### 3.4 Feature Engineering
Created required ratio features:
- `rooms_per_household`
- `bedrooms_per_room`
- `population_per_household`

Additional ratio candidates evaluated:
- `rooms_per_person`
- `bedrooms_per_household`

### 3.5 Skewness Correction
Numeric features with high positive skew on training data are transformed using `log1p`, excluding geographic/age fields used for direct interpretation.

Log-transformed features:
- `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`
- `rooms_per_household`, `bedrooms_per_room`, `population_per_household`
- `rooms_per_person`, `bedrooms_per_household`

### 3.6 Outlier Handling
Outliers are handled by row removal using 1st/99th percentile thresholds computed on training data only, then applied to train and test.

### 3.7 Encoding and Scaling Pipeline
The preprocessing stage is implemented with `ColumnTransformer`:
- Numeric branch: `StandardScaler`
- Categorical branch: `OneHotEncoder(handle_unknown="ignore")`

The transformer is fit on training data and used to transform both splits.

## 4. Results

### 4.1 Data Retention
- Train before outlier filtering: `16512`
- Train after outlier filtering: `14425`
- Test before outlier filtering: `4128`
- Test after outlier filtering: `3575`

Retention rates:
- Train: `87.36%`
- Test: `86.60%`

### 4.2 Final Feature Space
- Final processed feature count: `18`
- One-hot encoded `ocean_proximity` columns: 5

### 4.3 Correlation Findings (Exploratory)
Raw numeric correlation with target indicates strongest linear association for `median_income` (about `0.688`).

Engineered variables showing notable target relationships include:
- `rooms_per_household` (positive)
- `bedrooms_per_room` (negative)
- `population_per_household` (negative)

### 4.4 Feature Importance Preview (Exploratory)
Using `RandomForestRegressor` as an EDA diagnostic (not final inference):
- `median_income`: ~`0.4429`
- `ocean_proximity_INLAND`: ~`0.1452`
- `population_per_household`: ~`0.1194`
- `longitude`: ~`0.0570`
- `latitude`: ~`0.0504`

## 5. Output Artifacts
Generated files under [data/processed](data/processed):
- [data/processed/train_engineered.csv](data/processed/train_engineered.csv)
- [data/processed/test_engineered.csv](data/processed/test_engineered.csv)
- [data/processed/X_train_processed.csv](data/processed/X_train_processed.csv)
- [data/processed/X_test_processed.csv](data/processed/X_test_processed.csv)
- [data/processed/y_train.csv](data/processed/y_train.csv)
- [data/processed/y_test.csv](data/processed/y_test.csv)
- [data/processed/processed_matrices.npz](data/processed/processed_matrices.npz)
- [data/processed/metadata.json](data/processed/metadata.json)

The metadata file records run timestamp, split settings, imputation details, engineered/log features, outlier thresholds, encoding/scaling details, and artifact paths.

## 6. Discussion
The implemented workflow satisfies the requested preparation tasks and establishes a reliable preprocessing baseline for supervised learning experiments. The strongest observed driver remains `median_income`, while engineered density/ratio metrics add additional signal. The outlier policy improves robustness but removes approximately 13% of observations in each split, which may affect generalization in rare-market segments.

## 7. Limitations
- The workflow is notebook-centric and not yet packaged as a production pipeline module.
- Outlier removal can discard legitimate extreme observations.
- Feature importance is model-dependent and exploratory.
- No full benchmark comparison (cross-validation/hyperparameter tuning) is included.

## 8. Conclusion and Next Steps
This milestone delivers a complete, leakage-safe preprocessing report and reproducible artifact set. Recommended next actions are:

1. Add baseline regression metrics (RMSE, MAE, R2).
2. Compare outlier strategies (row removal vs clipping) quantitatively.
3. Add k-fold cross-validation for robustness checks.
4. Persist and version the preprocessing object for reuse in training/inference.

## 9. Reproduction Instructions
From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Run all cells in [notebooks/data_exploration.ipynb](notebooks/data_exploration.ipynb) from top to bottom.
