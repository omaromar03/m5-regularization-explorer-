# Regularization Explorer

This project visualizes how Logistic Regression coefficients change as regularization strength varies across 20 logarithmically spaced values of C from 0.001 to 100.

## Objective
The goal is to compare how **L1** and **L2** regularization affect feature coefficients in a telecom churn prediction task.

## What the visualization shows
- **L1 regularization** pushes some coefficients exactly to zero, making feature elimination visible.
- **L2 regularization** shrinks coefficients gradually toward zero, but usually keeps all features in the model.
- Features that remain relatively stable across the path are likely to be more robust predictors.

## Interpretation
The plot illustrates how the model behaves differently under L1 and L2 regularization as the value of C changes, where C represents the inverse strength of regularization. When using L1 regularization, the coefficients of some features decrease progressively and eventually reach exactly zero when C is small (i.e., when regularization is strong). This indicates that the model considers these features unimportant and effectively removes them, which is known as feature selection. 

In contrast, L2 regularization causes the coefficients to shrink gradually toward zero but rarely eliminates them completely. This means that all features remain in the model, although their influence is reduced. Additionally, some features remain relatively stable across different values of C, suggesting that they are strong and reliable predictors, while others shrink rapidly, indicating lower importance and sensitivity to regularization. 

Based on these observations, L1 regularization is more suitable when the goal is to build a simpler model with fewer features, whereas L2 is preferred when it is important to retain all available information and handle multicollinearity more effectively.
## Files
- `regularization_explorer.py` — main script
- `regularization_path.png` — output visualization
- `data/telecom_churn.csv` — dataset
- `requirements.txt` — dependencies
