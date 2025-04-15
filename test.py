from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Loading the model files for testing
model_cls = joblib.load("xgb_task_priority_model.pkl")
model_reg = joblib.load("xgb_energy_model.pkl")
preprocessor = joblib.load("task_preprocessor.pkl")

#Using the imported Label Encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['high', 'low', 'medium'])

# Flask app initialization
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df['cpu_mem_ratio'] = df['cpu_usage'] / (df['memory_usage'] + 1e-6)
    df['power_per_instruction'] = df['power_consumption'] / (df['num_executed_instructions'] + 1e-6)
    df['instruction_density'] = df['num_executed_instructions'] / (df['execution_time'] + 1e-6)
    df['power_per_sec'] = df['power_consumption'] / (df['execution_time'] + 1e-6)
    df['cpu_per_sec'] = df['cpu_usage'] / (df['execution_time'] + 1e-6)

    X = preprocessor.transform(df)

    # Task Priority prediction 
    pred_cls = model_cls.predict(X)[0]
    task_priority = label_encoder.inverse_transform([int(pred_cls)])[0]

    # Energy efficiency prediction
    pred_energy = model_reg.predict(X)[0]

    return jsonify({
        "task_priority": task_priority,
        "energy_efficiency": float(pred_energy)
    })

if __name__ == '__main__':
    app.run(debug=True)
