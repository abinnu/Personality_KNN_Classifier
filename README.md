# Personality_KNN_Classifier

🌿 Personality Type Prediction using KNN

📌 Overview

The Personality Type Prediction project uses Machine Learning to classify individuals as Introvert or Extrovert based on behavioral and social activity features. Using the K-Nearest Neighbors (KNN) algorithm, the model learns from patterns in user responses to predict personality type accurately.


Key highlights:

🧹 Cleaned & preprocessed psychometric dataset
🔍 Built a KNN classifier for personality type prediction
📊 Evaluated model performance using multiple metrics
🐳 Dockerized deployment for real-time usage


📂 Dataset

Source: [Kaggle - Personality Prediction Dataset](https://www.kaggle.com/datasets/xyz/personality-prediction)
Shape: 2901 rows × 8 features
Features:
  *Time_spent_Alone → Hours spent alone per week
  * Stage_fear → Confidence in public speaking
  * Social_event_attendance → Frequency of attending events
  * Going_outside → Outdoor activity level
  * Drained_after_socializing → Energy level after socializing
  * Friends_circle_size → Number of close friends
  * Post_frequency → Frequency of social media posting
Target:
  * Personality → Introvert 🧘‍♂️ or Extrovert 🗣️


🚀 Features

✅ Data Cleaning & Preprocessing
✅ Feature Scaling & Normalization
✅ KNN Model Training & Optimization
✅ Model Evaluation with Accuracy, Precision, Recall, F1-Score
✅ Dockerized Deployment for Real-time Usage


🧠 Tech Stack

* Language → Python 
* ML Algorithm → KNN (Scikit-learn)
* Data Processing → Pandas, NumPy
* Visualization → Matplotlib, Seaborn
* Deployment → Docker


Overall Metrics

Accuracy = 92% ✅
Macro Avg Precision = 92%
Macro Avg Recall = 92%
Macro Avg F1-score = 92%
Weighted Avg Precision = 92%
Weighted Avg Recall = 92%
Weighted Avg F1-score = 92%


🌐 Deployment (Dockerized)

You can deploy the app using Docker for real-time personality predictions.

1️⃣ Build Docker Image
docker build -t ml-personality-app .

2️⃣ Run Docker Container
docker run -d -p 5000:5000 ml-personality-app

3️⃣ Access the Application
Once the container is running, open your browser and visit:
http://localhost:5000


🤝 Contribution

Contributions are welcome! 🎉
If you'd like to improve this project:

Fork the repository 🍴
Create a feature branch
Submit a pull request 🚀


👨‍💻 Author

Abinnu John Peter.P
📧 Email: abinnu75@gmail.com
🔗 LinkedIn : www.linkedin.com/in/abinnu
