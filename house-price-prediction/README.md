# Dự báo giá nhà bằng Hồi quy tuyến tính

Dự án minh họa việc xây dựng mô hình **Hồi quy tuyến tính (Linear Regression)**
để dự báo giá nhà dựa trên các đặc trưng cơ bản của căn nhà.

**Nhánh 1: version 2**

 Cấu trúc thư mục

```
house-price-prediction/
├── data/
│   └── house_prices.csv    
├── generate_data.py         
├── linear_regression.py     
├── requirements.txt       
└── README.md
```

 Bộ dữ liệu (`data/house_prices.csv`)

Gồm 500 mẫu với các cột:

| Cột              | Ý nghĩa                                   |
|------------------|--------------------------------------------|
| `dien_tich`      | Diện tích nhà (m²)                          |
| `so_phong_ngu`   | Số phòng ngủ                                |
| `so_phong_tam`   | Số phòng tắm                                |
| `tuoi_nha`       | Tuổi căn nhà (năm)                          |
| `khoang_cach_tt` | Khoảng cách đến trung tâm thành phố (km)    |
| `gia_nha`        | Giá nhà (triệu VNĐ) — **biến mục tiêu**     |

 Cài đặt

```bash
pip install -r requirements.txt
```

 Chạy chương trình

```bash
# (tuỳ chọn) tạo lại dữ liệu
python generate_data.py

# huấn luyện và đánh giá mô hình
python linear_regression.py
```

 Kết quả

Script `linear_regression.py` sẽ:

1. Đọc dữ liệu từ `data/house_prices.csv`.
2. Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%).
3. Huấn luyện mô hình `LinearRegression` của scikit-learn.
4. In ra các chỉ số đánh giá: **MAE**, **RMSE**, **R²**.
5. In ra hệ số hồi quy của từng đặc trưng.
6. Lưu biểu đồ so sánh giá thực tế và giá dự đoán vào `ket_qua_du_bao.png`.
7. Dự đoán thử giá cho một căn nhà mẫu.

Kết quả tham khảo trên bộ dữ liệu mẫu:

- **R² ≈ 0.86** (mô hình giải thích được ~86% biến thiên của giá nhà)
- **MAE ≈ 229 triệu VNĐ**
- **RMSE ≈ 300 triệu VNĐ**

 Ghi chú

- Bộ dữ liệu trong repo này là dữ liệu **giả lập (synthetic)**, được sinh ra
  bằng công thức tuyến tính có thêm nhiễu ngẫu nhiên, chỉ nhằm mục đích minh
  hoạ thuật toán hồi quy tuyến tính. Nếu muốn dùng dữ liệu giá nhà thực tế,
  thay file `data/house_prices.csv` bằng bộ dữ liệu thật (giữ nguyên tên cột
  hoặc chỉnh sửa lại code cho phù hợp).
