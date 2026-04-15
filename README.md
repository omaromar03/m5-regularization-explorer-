# Regularization Explorer

This project visualizes how Logistic Regression coefficients change as regularization strength varies across 20 logarithmically spaced values of C from 0.001 to 100.

## Objective
The goal is to compare how **L1** and **L2** regularization affect feature coefficients in a telecom churn prediction task.

## What the visualization shows
- **L1 regularization** pushes some coefficients exactly to zero, making feature elimination visible.
- **L2 regularization** shrinks coefficients gradually toward zero, but usually keeps all features in the model.
- Features that remain relatively stable across the path are likely to be more robust predictors.

## Interpretation
The regularization path shows that L1 produces sparse solutions by eliminating weaker predictors as regularization becomes stronger. In contrast, L2 keeps all features while reducing their magnitudes more smoothly. Based on these results, L1 is a strong choice when feature selection and simpler models are preferred, while L2 is better when preserving all available predictive information is more important.

## Files
- `regularization_explorer.py` — main script
- `regularization_path.png` — output visualization
- `data/telecom_churn.csv` — dataset
- `requirements.txt` — dependencies
