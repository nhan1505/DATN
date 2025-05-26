from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Tải cả hai mô hình
models = {
    'random_forest': joblib.load('model/random_forest_model.pkl'),
    'decision_tree': joblib.load('model/decision_tree_model.pkl')
}
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
    model_choice = request.form['model']

    input_data = np.array([[age, sex, bmi, smoker, region]])
    model = models[model_choice]
    prediction_usd = model.predict(input_data)[0]
    prediction_vnd = prediction_usd * USD_TO_VND
    prediction_vnd_formatted = "{:,.0f}".format(prediction_vnd)

    model_name = "Random Forest" if model_choice == 'random_forest' else "Decision Tree"
    return render_template('index.html', prediction_text=f'Số tiền bảo hiểm dự đoán (mô hình {model_name}): {prediction_vnd_formatted} VND')

if __name__ == "__main__":
    app.run(debug=True)