import requests

response = requests.post(
    "http://localhost:8000/predict",
    files={"file": open(r"..\..\data\raw_landmarks\a\a_0001_20251127_181813.jpg", "rb")}
)

print(response.json())