"""
overfitting_demo.py
Minh hoa hien tuong OVERFITTING va cac ky thuat khac phuc,
su dung bo du lieu data/house_prices.csv.

Cach chay:
    python overfitting_demo.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = "data/house_prices.csv"
TARGET_COL = "gia_nha"

# ---------------------------------------------------------
# 1. Doc du lieu
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Chu y: lay tap TRAIN rat nho (10%) de de gay overfitting hon
# (it du lieu + model phuc tap -> hoc thuoc long thay vi hoc quy luat)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.1, random_state=42
)
print(f"So mau train: {len(X_train)} | So mau test: {len(X_test)}")


def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    r2_train = r2_score(y_tr, model.predict(X_tr))
    r2_test = r2_score(y_te, model.predict(X_te))
    mae_test = mean_absolute_error(y_te, model.predict(X_te))
    print(f"\n--- {name} ---")
    print(f"  R2 tren TRAIN : {r2_train:.4f}")
    print(f"  R2 tren TEST  : {r2_test:.4f}")
    print(f"  MAE tren TEST : {mae_test:,.1f} trieu VND")
    print(f"  Chenh lech R2 (train - test) : {r2_train - r2_test:.4f}")
    return r2_train, r2_test


results = {}

# ---------------------------------------------------------
# 2. TAO OVERFITTING: dung Polynomial Features bac cao
#    (rat nhieu he so, model qua phuc tap so voi luong du lieu train)
# ---------------------------------------------------------
degree = 6
overfit_model = make_pipeline(
    PolynomialFeatures(degree=degree, include_bias=False),
    StandardScaler(),
    LinearRegression(),
)
overfit_model.fit(X_train, y_train)
results["Overfit (Poly bac 6)"] = evaluate(
    f"MO HINH BI OVERFITTING (Polynomial bac {degree})",
    overfit_model, X_train, y_train, X_test, y_test,
)

# ---------------------------------------------------------
# 3. KHAC PHUC - Cach 1: Giam do phuc tap cua model
#    (giam bac polynomial xuong con 2, gan voi ban chat du lieu tuyen tinh)
# ---------------------------------------------------------
simple_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    LinearRegression(),
)
simple_model.fit(X_train, y_train)
results["Giam do phuc tap (Poly bac 2)"] = evaluate(
    "KHAC PHUC 1: Giam bac Polynomial xuong 2",
    simple_model, X_train, y_train, X_test, y_test,
)

# ---------------------------------------------------------
# 4. KHAC PHUC - Cach 2: Regularization (Ridge / Lasso)
#    Giu nguyen Polynomial bac 6 nhung them "phat" he so lon
# ---------------------------------------------------------
ridge_model = make_pipeline(
    PolynomialFeatures(degree=degree, include_bias=False),
    StandardScaler(),
    Ridge(alpha=50.0),
)
ridge_model.fit(X_train, y_train)
results["Ridge (L2 regularization)"] = evaluate(
    "KHAC PHUC 2: Ridge Regression (L2), alpha=50",
    ridge_model, X_train, y_train, X_test, y_test,
)

lasso_model = make_pipeline(
    PolynomialFeatures(degree=degree, include_bias=False),
    StandardScaler(),
    Lasso(alpha=30.0, max_iter=50000),
)
lasso_model.fit(X_train, y_train)
results["Lasso (L1 regularization)"] = evaluate(
    "KHAC PHUC 3: Lasso Regression (L1), alpha=30.0",
    lasso_model, X_train, y_train, X_test, y_test,
)

# ---------------------------------------------------------
# 5. KHAC PHUC - Cach 3: Cross-validation de kiem tra do on dinh
# ---------------------------------------------------------
cv_scores = cross_val_score(
    make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        Ridge(alpha=10.0),
    ),
    X, y, cv=5, scoring="r2"
)
print("\n--- KHAC PHUC 4: Cross-validation (5-fold) tren toan bo data ---")
print(f"  R2 tung fold : {np.round(cv_scores, 4)}")
print(f"  R2 trung binh: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ---------------------------------------------------------
# 6. Ve bieu do so sanh R2 train vs test giua cac model
# ---------------------------------------------------------
labels = list(results.keys())
train_scores = [results[k][0] for k in labels]
test_scores = [results[k][1] for k in labels]

x_pos = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x_pos - width / 2, train_scores, width, label="R2 - Train")
plt.bar(x_pos + width / 2, test_scores, width, label="R2 - Test")
plt.axhline(0, color="gray", linewidth=0.8)
plt.xticks(x_pos, labels, rotation=20, ha="right")
plt.ylabel("R2 score")
plt.title("So sanh R2 Train vs Test: Overfitting va cac cach khac phuc")
plt.legend()
plt.tight_layout()
plt.savefig("overfitting_comparison.png", dpi=150)
print("\nDa luu bieu do so sanh vao 'overfitting_comparison.png'")

print("""
=== TOM TAT ===
- Mo hinh Polynomial bac 6 tren tap train rat nho -> R2 train rat cao
  nhung R2 test thap (thap chi can am) => OVERFITTING ro ret.
- Giam do phuc tap (Poly bac 2) -> khoang cach train/test giam han.
- Them Ridge/Lasso regularization giup "phat" he so lon, giam overfitting
  ma van giu duoc do phuc tap cua Polynomial bac 6.
- Cross-validation giup danh gia mo hinh on dinh hon, khong phu thuoc
  vao 1 lan chia train/test duy nhat.
""")
