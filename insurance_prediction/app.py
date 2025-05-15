from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model/insurance_model_optimized.pkl')
USD_TO_VND = 25000

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    sex = int(request.form['sex'])
    bmi = float(request.form['bmi'])
    smoker = int(request.form['smoker'])
    region = int(request.form['region'])

    input_data = np.array([[age, sex, bmi, smoker, region]])
    prediction_usd = model.predict(input_data)[0]
    prediction_vnd = prediction_usd * USD_TO_VND
    prediction_vnd_formatted = "{:,.0f}".format(prediction_vnd)

    return render_template('index.html', prediction_text=f'Số tiền bảo hiểm dự đoán: {prediction_vnd_formatted} VND')

if __name__ == "__main__":
    app.run(debug=True)
