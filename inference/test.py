import requests

application = {
    "transactionID": 100000,
    "price": 50,
    "income": 10000,
    "age": 35,
    "sex": 1,
    "defaulted_earlier": 0,
    "late_earlier": 0,
}

url = "http://localhost:9696/app"
response = requests.post(url, json=application)
print(response.json())
