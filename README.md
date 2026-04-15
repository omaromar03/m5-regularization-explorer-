# m5-regularization-explorer-
Regularization path visualization using Logistic Regression
# Regularization Explorer

This project visualizes how Logistic Regression coefficients change as regularization strength varies across 20 logarithmically spaced values of C from 0.001 to 100.

## What this shows
- **L1 regularization** drives some coefficients exactly to zero, which makes feature elimination visible.
- **L2 regularization** shrinks coefficients smoothly toward zero, but usually does not eliminate them completely.
- Features that remain relatively stable across the regularization path are likely to be more robust predictors.

## Interpretation
The plot shows that L1 regularization creates sparse solutions by forcing weaker features to zero as regularization becomes stronger. L2 regularization, in contrast, keeps all features in the model while reducing their magnitudes more gradually. Based on this behavior, L1 is useful when model simplicity and feature selection are priorities, while L2 is often preferable when we want to preserve all predictive signals and handle multicollinearity more smoothly.

## Files
- `regularization_explorer.py`
- `regularization_path.png`
- `data/telecom_churn.csv`