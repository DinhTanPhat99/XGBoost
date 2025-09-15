# Credit Scoring với XGBoost (Give Me Some Credit Dataset)

## 📌 Giới thiệu
Dự án này xây dựng mô hình **Credit Scoring** (dự đoán rủi ro vỡ nợ) sử dụng **XGBoost** trên bộ dữ liệu [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit).  
Ngoài ra, dự án có phần **Flask web app** để nhập dữ liệu khách hàng và dự đoán trực tiếp.
---
## 📂 Cấu trúc thư mục
---
XGBoost/
-app.py # Flask web app (chạy giao diện web dự đoán)
-cs-training.csv # Dataset (Give Me Some Credit)
-XGBoost.py # Script train XGBoost + RandomizedSearchCV
-output_model/ # Lưu model, preprocessor, tham số và kết quả
+best_params.joblib # Tham số tối ưu sau RandomizedSearchCV
+confusion_matrix.png # Ma trận nhầm lẫn (trên test set)
+feature_names.csv # Danh sách tên feature
+preprocessor.joblib # Pipeline tiền xử lý (imputer + scaler)
+xgb_final_model.joblib # Model XGBoost final
-templates/ # Thư mục giao diện web Flask
+index.html
-index.html # Form input + kết quả dự đoán (Bootstrap 5)
-README.md # Hướng dẫn sử dụng
<img width="1912" height="613" alt="image" src="https://github.com/user-attachments/assets/1e387bad-e4f4-4718-ba09-ca9199f76a49" />

