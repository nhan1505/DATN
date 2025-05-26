import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

# Đọc dữ liệu
try:
    data = pd.read_csv('data/insurance.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'data/insurance.csv'.")
    exit()

# Kiểm tra cột
required_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
if not all(col in data.columns for col in required_columns):
    print("Lỗi: Thiếu cột cần thiết!")
    exit()

# Chuyển đổi categorical variables
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data = pd.get_dummies(data, columns=['region'], prefix='region')

# Xử lý giá trị thiếu
if data.isnull().any().any():
    print("Xử lý giá trị thiếu...")
    data = data.fillna(data.mean(numeric_only=True))

# Chọn đặc trưng và nhãn
feature_columns = ['age', 'sex', 'bmi', 'children', 'smoker'] + [col for col in data.columns if col.startswith('region_')]
X = data[feature_columns]
y = data['charges']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Kích thước tập huấn luyện: {X_train.shape[0]} mẫu")
print(f"Kích thước tập kiểm tra: {X_test.shape[0]} mẫu")

# Tối ưu hóa hyperparameter
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
dt = DecisionTreeRegressor(random_state=42)
print("Bắt đầu tìm kiếm hyperparameter tối ưu...")
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Lấy mô hình tốt nhất
best_model = grid_search.best_estimator_
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
train_mae = mean_absolute_error(y_train, best_model.predict(X_train))
test_mae = mean_absolute_error(y_test, best_model.predict(X_test))
print(f"Decision Tree - Best parameters: {grid_search.best_params_}")
print(f"Decision Tree - R² trên tập huấn luyện: {train_score:.4f}")
print(f"Decision Tree - R² trên tập kiểm tra: {test_score:.4f}")
print(f"Decision Tree - MAE trên tập huấn luyện: {train_mae:.2f}")
print(f"Decision Tree - MAE trên tập kiểm tra: {test_mae:.2f}")

# Lưu mô hình
try:
    os.makedirs('model', exist_ok=True)
    joblib.dump(best_model, 'model/decision_tree_model.pkl')
    print("Mô hình Decision Tree đã được lưu.")
except Exception as e:
    print(f"Lỗi khi lưu mô hình: {e}")
