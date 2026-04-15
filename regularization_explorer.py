import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_data(filepath="data/telecom_churn.csv"):
    return pd.read_csv(filepath)


def prepare_features(df):
    # تنظيف أسماء الأعمدة
    df.columns = df.columns.str.strip().str.lower()

    # التأكد من وجود الهدف
    if "churned" not in df.columns:
        raise ValueError(f"'churned' column not found. Columns: {df.columns}")

    # حذف ID
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # تحويل total_charges إلى رقم إذا فيه مشاكل
    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    # حذف القيم الفارغة
    df = df.dropna()

    # تقسيم البيانات
    X = df.drop(columns=["churned"])
    y = df["churned"]

    # تحويل الهدف إلى رقم
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower().map({
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0
        })

    # encoding
    X = pd.get_dummies(X, drop_first=True)

    feature_names = X.columns.tolist()

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.astype(int), feature_names


def fit_path(X, y, C_values, penalty):
    coefs = []

    for C in C_values:
        solver = "liblinear" if penalty == "l1" else "lbfgs"

        model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=2000
        )

        model.fit(X, y)
        coefs.append(model.coef_[0])

    return np.array(coefs)


def plot_paths(C_values, coef_l1, coef_l2, features):
    plt.figure(figsize=(14, 6))

    # L1
    plt.subplot(1, 2, 1)
    for i in range(len(features)):
        plt.plot(C_values, coef_l1[:, i])
    plt.xscale("log")
    plt.title("L1 Regularization")
    plt.xlabel("C")
    plt.ylabel("Coefficient")

    # L2
    plt.subplot(1, 2, 2)
    for i in range(len(features)):
        plt.plot(C_values, coef_l2[:, i])
    plt.xscale("log")
    plt.title("L2 Regularization")
    plt.xlabel("C")

    plt.tight_layout()
    plt.savefig("regularization_path.png")
    plt.show()


def main():
    df = load_data()

    X, y, features = prepare_features(df)

    C_values = np.logspace(-3, 2, 20)

    coef_l1 = fit_path(X, y, C_values, "l1")
    coef_l2 = fit_path(X, y, C_values, "l2")

    print("\nFeatures that go to zero (L1):")
    for i, f in enumerate(features):
        if np.any(coef_l1[:, i] == 0):
            print(f)

    plot_paths(C_values, coef_l1, coef_l2, features)


if __name__ == "__main__":
    main()