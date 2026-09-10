"""
generate_data.py
Tạo bộ dữ liệu giả lập (synthetic) về giá nhà để phục vụ cho việc
huấn luyện mô hình hồi quy tuyến tính.

Các đặc trưng (feature):
- dien_tich       : Diện tích nhà (m2)
- so_phong_ngu    : Số phòng ngủ
- so_phong_tam    : Số phòng tắm
- tuoi_nha        : Tuổi của căn nhà (năm)
- khoang_cach_tt  : Khoảng cách đến trung tâm thành phố (km)

Biến mục tiêu (target):
- gia_nha (đơn vị: triệu VNĐ)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 500

dien_tich = np.random.normal(90, 30, N_SAMPLES).clip(25, 300)
so_phong_ngu = np.random.randint(1, 6, N_SAMPLES)
so_phong_tam = np.random.randint(1, 4, N_SAMPLES)
tuoi_nha = np.random.randint(0, 40, N_SAMPLES)
khoang_cach_tt = np.random.exponential(8, N_SAMPLES).clip(0.5, 40)

# Công thức giả định để tạo giá nhà có quan hệ tuyến tính (+ nhiễu ngẫu nhiên)
gia_nha = (
    dien_tich * 25
    + so_phong_ngu * 150
    + so_phong_tam * 100
    - tuoi_nha * 20
    - khoang_cach_tt * 30
    + np.random.normal(0, 300, N_SAMPLES)
    + 500
)
gia_nha = gia_nha.clip(300, None)  # giá không âm

df = pd.DataFrame({
    "dien_tich": dien_tich.round(1),
    "so_phong_ngu": so_phong_ngu,
    "so_phong_tam": so_phong_tam,
    "tuoi_nha": tuoi_nha,
    "khoang_cach_tt": khoang_cach_tt.round(2),
    "gia_nha": gia_nha.round(1),
})

df.to_csv("data/house_prices.csv", index=False, encoding="utf-8-sig")
print(f"Đã tạo {len(df)} dòng dữ liệu tại data/house_prices.csv")
print(df.head())
