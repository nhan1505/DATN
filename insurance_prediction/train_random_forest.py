import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# Đọc dữ liệu từ file CSV
try:
    data = pd.read_csv('data/insurance.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'data/insurance.csv'. Kiểm tra đường dẫn file!")
    exit()

# Kiểm tra các cột cần thiết
required_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
if not all(col in data.columns for col in required_columns):
    print("Lỗi: Dữ liệu thiếu một số cột cần thiết!")
    exit()

# Chuyển đổi các cột chữ thành số
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
# Sử dụng one-hot encoding cho region
data = pd.get_dummies(data, columns=['region'], prefix='region')

# Xử lý giá trị thiếu
if data.isnull().any().any():
    print("Xử lý giá trị thiếu bằng cách điền giá trị trung bình...")
    data = data.fillna(data.mean(numeric_only=True))

# Chọn các cột đặc trưng và nhãn
feature_columns = ['age', 'sex', 'bmi', 'children', 'smoker'] + [col for col in data.columns if col.startswith('region_')]
X = data[feature_columns]
y = data['charges']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Kích thước tập huấn luyện: {X_train.shape[0]} mẫu")
print(f"Kích thước tập kiểm tra: {X_test.shape[0]} mẫu")

# Thiết lập hyperparameter
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', None]
}

# Khởi tạo và huấn luyện mô hình
rf = RandomForestRegressor(random_state=42)
print("Bắt đầu tìm kiếm hyperparameter tối ưu...")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Lấy mô hình tốt nhất
best_model = grid_search.best_estimator_
print(f"Hyperparameter tốt nhất: {grid_search.best_params_}")

# Đánh giá mô hình
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
train_mae = mean_absolute_error(y_train, best_model.predict(X_train))
test_mae = mean_absolute_error(y_test, best_model.predict(X_test))
print(f"Random Forest - R² trên tập huấn luyện: {train_score:.4f}")
print(f"Random Forest - R² trên tập kiểm tra: {test_score:.4f}")
print(f"Random Forest - MAE trên tập huấn luyện: {train_mae:.2f}")
print(f"Random Forest - MAE trên tập kiểm tra: {test_mae:.2f}")

# Lưu mô hình
try:
    os.makedirs('model', exist_ok=True)
    joblib.dump(best_model, 'model/random_forest_model.pkl')
    print("Mô hình Random Forest đã được lưu'.")
except Exception as e:
    print(f"Lỗi khi lưu mô hình: {e}")
