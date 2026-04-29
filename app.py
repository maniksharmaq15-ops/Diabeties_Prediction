from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age']),
        ]
        input_array = np.asarray(features).reshape(1, -1)
        std_data = scaler.transform(input_array)
        prediction = model.predict(std_data)

        if prediction[0] == 0:
            result = "Not Diabetic"
            result_class = "negative"
        else:
            result = "Diabetic"
            result_class = "positive"

        return render_template('index.html', prediction=result, result_class=result_class)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}", result_class="error")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
