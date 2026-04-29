import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train
classifier = SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Save model and scaler
pickle.dump(classifier, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("model.pkl and scaler.pkl saved successfully!")
