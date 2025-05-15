import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib

# Đọc dữ liệu
data = pd.read_csv('data/insurance.csv')

# Tiền xử lý dữ liệu
# Mã hóa biến phân loại
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data['region'] = data['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})

# Chọn các đặc trưng và nhãn
X = data[['age', 'sex', 'bmi', 'smoker', 'region']]
y = data['charges']

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tối ưu hóa hyperparameter
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Lấy mô hình tốt nhất
best_model = grid_search.best_estimator_

# Đánh giá mô hình
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Độ chính xác trên tập huấn luyện: {train_score:.4f}")
print(f"Độ chính xác trên tập kiểm tra: {test_score:.4f}")

# Lưu mô hình
joblib.dump(best_model, 'model/insurance_model_optimized.pkl')

print("Mô hình tối ưu đã được lưu.")