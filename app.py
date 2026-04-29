from flask import Flask, request, render_template
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'diabetes.csv')

def train_and_save():
    print("Training model...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns='Outcome', axis=1)
    Y = df['Outcome']
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)
    clf = SVC(kernel='linear')
    clf.fit(X_train, Y_train)
    pickle.dump(clf, open(MODEL_PATH, 'wb'))
    pickle.dump(scaler, open(SCALER_PATH, 'wb'))
    print("model.pkl and scaler.pkl saved.")
    return clf, scaler

# Train on startup if pkl files are missing
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    model, scaler = train_and_save()
else:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))

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
            result, result_class = "Not Diabetic", "negative"
        else:
            result, result_class = "Diabetic", "positive"
        return render_template('index.html', prediction=result, result_class=result_class)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}", result_class="error")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
