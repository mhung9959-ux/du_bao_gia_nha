"""
linear_regression.py
Xây dựng mô hình Hồi quy tuyến tính (Linear Regression) để dự báo
giá nhà dựa trên bộ dữ liệu data/house_prices.csv.

Cách chạy:
    python linear_regression.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/house_prices.csv"
TARGET_COL = "gia_nha"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Đã đọc {len(df)} dòng, {df.shape[1]} cột từ '{path}'")
    return df


def train_and_evaluate(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n=== KẾT QUẢ MÔ HÌNH ===")
    print(f"MAE  (sai số tuyệt đối trung bình) : {mae:,.2f} triệu VNĐ")
    print(f"RMSE (căn bậc 2 sai số bình phương): {rmse:,.2f} triệu VNĐ")
    print(f"R^2  (độ phù hợp mô hình)          : {r2:.4f}")

    print("\n=== HỆ SỐ HỒI QUY (COEFFICIENTS) ===")
    for name, coef in zip(X.columns, model.coef_):
        print(f"  {name:<16}: {coef:,.2f}")
    print(f"  {'intercept':<16}: {model.intercept_:,.2f}")

    return model, X_test, y_test, y_pred


def plot_results(y_test, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Dự đoán hoàn hảo")
    plt.xlabel("Giá thực tế (triệu VNĐ)")
    plt.ylabel("Giá dự đoán (triệu VNĐ)")
    plt.title("So sánh giá thực tế và giá dự đoán")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ket_qua_du_bao.png", dpi=150)
    print("\nĐã lưu biểu đồ kết quả vào 'ket_qua_du_bao.png'")


def predict_sample(model, X_columns):
    """Ví dụ dự đoán giá cho một căn nhà cụ thể."""
    sample = pd.DataFrame([{
        "dien_tich": 100,
        "so_phong_ngu": 3,
        "so_phong_tam": 2,
        "tuoi_nha": 5,
        "khoang_cach_tt": 6.5,
    }])[X_columns]
    pred = model.predict(sample)[0]
    print("\n=== VÍ DỤ DỰ ĐOÁN ===")
    print(sample.to_string(index=False))
    print(f"=> Giá dự đoán: {pred:,.1f} triệu VNĐ")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    model, X_test, y_test, y_pred = train_and_evaluate(df)
    plot_results(y_test, y_pred)
    predict_sample(model, X_test.columns)
