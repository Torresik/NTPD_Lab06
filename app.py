from fastapi import FastAPI
from model import train_and_predict, get_accuracy

app = FastAPI(title="CI/CD ML API")

@app.get("/")
def read_root():
    return {"message": "API Modelu Iris dziala"}

@app.get("/metrics")
def metrics():
    # Zwraca dokladnosc modelu
    return {"accuracy": get_accuracy()}