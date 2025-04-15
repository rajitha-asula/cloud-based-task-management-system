import requests

url = "http://127.0.0.1:5000/predict"

payload = {
  "cpu_usage": 155.0,
  "memory_usage": 260.0,
  "network_traffic": 4250.0,
  "power_consumption": 4150.0,
  "num_executed_instructions": 7000,
  "execution_time": 50.0,
  "hour": 134,
  "day": 12,
  "weekday": 5,
  "task_type": "compute",
  "task_status": "completed"
}

response = requests.post(url, json=payload)

print("🔍 Status Code:", response.status_code)
try:
    print("Prediction Result:", response.json())
except Exception as e:
    print("Failed to decode JSON:", e)
    print("Raw response text:", response.text)