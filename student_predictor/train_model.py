import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
file_path = 'student_predictor/data/student_score.csv'
data = pd.read_csv(file_path)

# Features and target
X = data[['Hours_Studied', 'Sleep_Hours', 'Attendance', 'Participation']]
y = data['Final_Score']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'student_predictor/models/predictor_model.pkl')

print('Model trained and saved as student_predictor/models/predictor_model.pkl')
