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

### Run the notebook

1. Place the CSV files in the `data/` directory:

- `data/hotels.csv`
- `data/reviews.csv`
- `data/users.csv`

2. Launch Jupyter:
```bash
jupyter notebook
# or
jupyter lab
```

3. Open `hotel_review_analysis_prediction.ipynb` and run cells from top to bottom.

---

## Reproducibility & recommended workflow

To improve reproducibility and production-readiness:

- Break the notebook into modular scripts under `src/`:
  - `src/data_preprocessing.py`
  - `src/features.py`
  - `src/train.py`
  - `src/evaluate.py`
  - `src/infer.py`

- Use `sklearn.pipeline.Pipeline` to encapsulate preprocessing + model.
- Replace single train/test split with **k-fold cross-validation** (e.g., `KFold` or `StratifiedKFold` depending on data).
- Use `RandomizedSearchCV` or `GridSearchCV` for hyperparameter tuning.
- Persist the final pipeline with `joblib.dump()` (add sample code to `src/inference.py`).
- Add unit tests for critical functions in `tests/` and configure CI (GitHub Actions).

---

## Suggestions for improvement

1. **Include NLP features**: use TF-IDF or transformer embeddings on `review_text`. This is likely to increase performance significantly.
2. **Feature importance & interpretability**: compute permutation importance or SHAP values to explain model predictions.
3. **Model serving / demo**: add a simple Flask/FastAPI demo that accepts sample review metadata + text and returns a predicted score.
4. **Data versioning**: consider DVC or another solution for dataset version control if the dataset evolves.
5. **Documentation & examples**: include a short `examples/` folder with sample inference calls.

---

## Contributing

Contributions are welcome! If you want to contribute:

1. Fork the repository
2. Create a branch for your feature: `git checkout -b feat/your-feature`
3. Make changes and add tests where appropriate
4. Submit a pull request describing your changes

Please follow the existing code style and add clear commit messages.

---

## License

This project is released under the **MIT License**. See [LICENSE] for details.

---

## Contact

Author: *Arian Jr* — [My GitHub](https://github.com/ArianJr)





