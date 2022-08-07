import json

import requests

with open("inference/example_application.json", "r", encoding="utf-8") as f:
    application = json.load(f)


url = "http://localhost:9696/app"
response = requests.post(url, json=application)
print(response.json())
