from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Tải mô hình và scaler
model = joblib.load('model/insurance_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Tỷ giá USD sang VND
USD_TO_VND = 25000

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    age = float(request.form['age'])
    sex = int(request.form['sex'])  # 0: Nam, 1: Nữ
    height = float(request.form['height']) / 100  # Chuyển cm sang m
    weight = float(request.form['weight'])  # kg
    smoker = int(request.form['smoker'])  # 0: Không, 1: Có
    region = int(request.form['region'])  # 0: southwest, 1: southeast, 2: northwest, 3: northeast
    
    # Tính BMI
    bmi = weight / (height * height)
    
    # Tạo mảng đầu vào
    input_data = np.array([[age, sex, bmi, smoker, region]])
    
    # Chuẩn hóa dữ liệu đầu vào
    input_data_scaled = scaler.transform(input_data)
    
    # Dự đoán (USD)
    prediction_usd = model.predict(input_data_scaled)[0]
    
    # Chuyển sang VND
    prediction_vnd = prediction_usd * USD_TO_VND
    
    # Định dạng số tiền VND với dấu phân cách hàng nghìn
    prediction_vnd_formatted = "{:,.0f}".format(prediction_vnd)
    
    return render_template('index.html', prediction_text=f'Số tiền bảo hiểm dự đoán: {prediction_vnd_formatted} VND')

if __name__ == "__main__":
    app.run(debug=True)