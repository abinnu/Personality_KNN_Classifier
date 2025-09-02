from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__, template_folder='template') 
model = joblib.load("model/Knnmodel.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = []
    feature_names = ['Age', 'Openness', 'Conscientiousness',
    'Extraversion','Agreeableness', 'Neuroticism', 'Stage_fear']
    
    for feature in feature_names:
        value = request.form.get(feature)
        features.append(float(value))
    
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    pred = model.predict(features_scaled)
    personality = "Extrovert" if pred[0] == 1 else "Introvert"
    
    return render_template('index.html', prediction=personality)

if __name__ == '__main__':
    app.run(debug=True)


