# Waze User Churn Analysis

An end-to-end data analysis and machine learning project investigating user churn
behaviour in the Waze navigation app. The project follows the full analytical
pipeline — from raw data inspection to a production-ready churn prediction model —
using Python across five structured stages.

---

## Project Overview

**Business problem:** Waze leadership needed to understand why users were
uninstalling or stopping use of the app, and to build a model that could
predict churn before it happened.

**Approach:** Five-stage analytical pipeline covering data inspection, EDA,
statistical testing, logistic regression, and tree-based machine learning.

**Outcome:** XGBoost champion model achieving 18.1% recall on the minority
churn class, with key behavioural insight that high-intensity, low-frequency
drivers are most at risk of churning — a finding with direct implications for
targeted retention strategies.

---

## Repository Structure
waze-churn-analysis/

├── data/

│   └── waze_dataset.csv          # 14,999 user records, 13 features

├── notebooks/

│   ├── 01_data_inspection.ipynb  # Data loading, structure, missing values

│   ├── 02_eda_visualization.ipynb # EDA, distributions, feature engineering

│   ├── 03_hypothesis_testing.ipynb # Two-sample t-test: iPhone vs Android

│   ├── 04_logistic_regression.ipynb # Binomial logistic regression model

│   └── 05_tree_models.ipynb      # Random forest + XGBoost with GridSearchCV

└── README.md

---

## Key Findings

### Behavioural profile of churned users
| Metric | Churned | Retained |
|---|---|---|
| Median drives per month | 50 | 47 |
| Median driving days | 6 | 14 |
| Median km per driving day | 698 km | 290 km |
| Median drives per driving day | 10 | 4 |

Churned users drove significantly more distance per driving day but used
the app on far fewer days — consistent with a professional or long-haul
driver profile whose needs differ from the app's core commuter audience.

### Missing data
700 rows (4.7%) had missing values in the `label` column (the churn target).
Comparison of behavioural statistics between null and non-null rows confirmed
the data was missing at random — safely excluded from modelling without
introducing bias.

### Hypothesis test (Course 3)
A two-sample Welch's t-test found no statistically significant difference
in mean number of drives between iPhone and Android users
(t = 1.46, p = 0.143). Device type is not a meaningful churn predictor.

### Model performance summary
| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.52 | 0.09 | 0.16 | 0.82 |
| Random Forest (CV) | 0.46 | 0.13 | 0.20 | 0.82 |
| XGBoost (CV) | 0.43 | 0.17 | 0.24 | 0.81 |
| **XGBoost (test)** | **0.42** | **0.18** | **0.25** | **0.81** |

Recall was selected as the primary evaluation metric given the low cost of
false positives (sending a retention notification to a user who wasn't
going to churn) relative to false negatives (missing a user who was).

### Top predictive features (XGBoost)
Engineered features accounted for 6 of the top 10 most important features,
confirming that domain-informed feature engineering is the highest-leverage
lever for model improvement with this dataset.

---

## Methodology

### Tools and libraries
- **Python:** pandas, numpy, matplotlib, seaborn
- **Statistics:** scipy.stats
- **Machine learning:** scikit-learn, XGBoost
- **Framework:** PACE (Plan → Analyze → Construct → Execute)

### Feature engineering
New features created to improve model signal:
- `km_per_driving_day` — distance intensity per active day
- `km_per_drive` — average trip length
- `drives_per_driving_day` — drive frequency per active day
- `percent_sessions_in_last_month` — recency of engagement
- `total_sessions_per_day` — long-run engagement rate
- `km_per_hour` — speed proxy (flagged as unrealistic; raised as data quality concern)
- `percent_of_drives_to_favorite` — navigation routine vs exploration
- `professional_driver` — binary flag for high-intensity users (≥60 drives AND ≥15 driving days)

### Outlier handling
- Logistic regression: values above the 95th percentile capped for 7 columns
- Tree models: no imputation required (tree-based models are outlier-resilient)

### Class imbalance
Target variable: 82.3% retained, 17.7% churned. Handled via stratified
train/test/validation splits. No resampling applied.

---

## Recommendations to Waze

1. **Investigate the professional driver segment.** Users driving 700+ km
   on days they use the app churn at a disproportionately high rate. Waze
   should determine whether this group has unmet needs (e.g. truck-specific
   routing, HGV restrictions, rest stop information) that a targeted feature
   set could address.

2. **Collect churn reason data.** Exit surveys or in-app feedback at the
   point of uninstall would dramatically improve future model signal.

3. **Clarify data pipeline for `label`.** The 700 missing values in the
   target variable need a source explanation to ensure future datasets are
   complete.

4. **Lower the decision threshold for retention interventions.** At the
   default 0.5 threshold, recall is 18%. Lowering to 0.194 yields ~50%
   recall with acceptable precision (~29%) — appropriate if retention
   outreach is low-cost and non-intrusive.

5. **Gather drive-level data.** Trip timestamps, start/end locations, and
   route types would enable significantly richer feature engineering and
   stronger churn prediction.

---

## About This Project

This project was completed as part of the
[Google Advanced Data Analytics Certificate](https://grow.google/certificates/data-analytics/)
(Courses 1–5), using the Waze fictional dataset provided by the programme.
All analysis, interpretation, feature engineering, and documentation is
original work.

**Skills demonstrated:** Python · pandas · seaborn · scikit-learn · XGBoost ·
hypothesis testing · logistic regression · random forest · GridSearchCV ·
feature engineering · EDA · executive communication · PACE framework

---
