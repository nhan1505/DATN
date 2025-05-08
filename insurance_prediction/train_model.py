import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Đọc dữ liệu (giả sử có file insurance.csv)
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

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện mô hình
model = RandomForestRegressor(n_estimators=170, random_state=42)
model.fit(X_train_scaled, y_train)

# Đánh giá mô hình
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Độ chính xác trên tập huấn luyện: {train_score:.4f}")
print(f"Độ chính xác trên tập kiểm tra: {test_score:.4f}")

# Lưu mô hình và scaler
joblib.dump(model, 'model/insurance_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("Mô hình và scaler đã được lưu.")