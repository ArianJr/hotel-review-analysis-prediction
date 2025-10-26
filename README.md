# Hotel Review Analysis & Score Prediction

[![Notebook](https://img.shields.io/badge/notebook-Jupyter-orange.svg)](./hotel_review_analysis_prediction.ipynb)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()

## Overview

**Hotel Review Analysis & Score Prediction** is a reproducible data science project that explores hotel, review, and user metadata to predict an overall review score (`score_overall`). The project combines exploratory data analysis (EDA), feature engineering, and supervised regression models to compare performance and produce a deployable prediction pipeline.

The notebook demonstrates:
- Data ingestion and merging of hotel, review, and user datasets
- EDA (distributions, relationships, correlation heatmaps)
- Preprocessing and feature scaling
- Model training and evaluation (Linear Regression, Random Forest, XGBoost)
- Visualization of model performance and prediction diagnostics

> Note: The current notebook version does not yet use review text for prediction. A high-impact next step is to include NLP features (TF-IDF or transformer embeddings) for improved predictive power.

---

## Key results

| Model                | R² Score  | RMSE       |
|---------------------:|:---------:|:----------:|
| Linear Regression    | 0.96736   | 0.03319    |
| Random Forest        | 0.94641   | 0.04253    |
| XGBoost (XGBRF)      | 0.76461   | 0.08914    |

(Results above were computed on a single train/test split — see *Reproducibility* and *Improvements* sections for recommended validation procedures.)

---

## Project structure


> Keep raw data out of the repository; include a `data/README.md` describing how to obtain the data or include a small sample dataset for demonstration.

---

## Getting started

### Requirements

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```
