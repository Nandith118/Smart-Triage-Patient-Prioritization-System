# train_triage_ai.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

print("⏳ Simulating historical emergency room patient admission records...")
np.random.seed(42)
num_records = 15000

# Feature Engineering: Simulating critical real-world vitals
age = np.random.randint(1, 90, num_records)
heart_rate = np.random.uniform(50, 160, num_records)       # normal is ~60-100 bpm
oxygen_sat = np.random.uniform(75, 100, num_records)       # critical danger is <90%
pain_score = np.random.randint(1, 11, num_records)         # pain scale 1-10
systolic_bp = np.random.uniform(90, 200, num_records)      # blood pressure spikes


# low oxygen drops, extreme heart rates, high pain, and older age drive the danger up.
base_urgency = (100 - oxygen_sat) * 2.5 
hr_anomaly = np.abs(heart_rate - 75) * 0.3
bp_anomaly = np.where(systolic_bp > 140, (systolic_bp - 140) * 0.4, 0)
pain_impact = pain_score * 2.0
age_multiplier = np.where(age > 65, 1.2, 1.0)

urgency_score = (base_urgency + hr_anomaly + bp_anomaly + pain_impact) * age_multiplier
urgency_score += np.random.normal(0, 3, num_records)       
urgency_score = np.clip(urgency_score, 0.0, 100.0)         

df = pd.DataFrame({
    'age': age,
    'heart_rate': heart_rate,
    'oxygen_sat': oxygen_sat,
    'pain_score': pain_score,
    'systolic_bp': systolic_bp,
    'urgency_score': urgency_score
})

X = df[['age', 'heart_rate', 'oxygen_sat', 'pain_score', 'systolic_bp']]
y = df['urgency_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(" Training AI Triage Officer Model..")
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

print(f"Training Complete! Model Accuracy Score: {model.score(X_test, y_test):.4f}")

with open('triage_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved model artifact as 'triage_model.pkl'!")