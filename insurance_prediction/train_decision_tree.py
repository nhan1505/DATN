import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import joblib

# Đọc dữ liệu
data = pd.read_csv('data/insurance.csv')
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data['region'] = data['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})

# Chọn đặc trưng và nhãn
X = data[['age', 'sex', 'bmi', 'smoker', 'region']]
y = data['charges']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tối ưu hóa hyperparameter
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['auto', 'sqrt', 'log2', None]
}
dt = DecisionTreeRegressor(random_state=42)
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Lấy mô hình tốt nhất
best_model = grid_search.best_estimator_
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
print(f"Decision Tree - Best parameters: {grid_search.best_params_}")
print(f"Decision Tree - Độ chính xác trên tập huấn luyện: {train_score:.4f}")
print(f"Decision Tree - Độ chính xác trên tập kiểm tra: {test_score:.4f}")

# Lưu mô hình
joblib.dump(best_model, 'model/decision_tree_model.pkl')
print("Mô hình Decision Tree đã được lưu.")