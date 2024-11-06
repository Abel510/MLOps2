import json
import mlflow
from web_service import app
from sklearn.linear_model import LinearRegression

def test_predict_endpoint():
    with app.test_client() as client:
        response = client.post("/predict", json={"MedInc": 3.0, "HouseAge": 19.0, "AveRooms": 5.0, "AveBedrms": 2.0, "Population": 2100, "AveOccup": 2.0, "Latitude": 37.0, "Longitude": -120.0})
        assert response.status_code == 200
        assert "prediction" in json.loads(response.data)

def test_update_model_endpoint():
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run():
        mlflow.sklearn.log_model(LinearRegression(), "model")
        model_uri = f"models:/california-housing/1"

    with app.test_client() as client:
        response = client.post("/update-model", json={"model_uri": model_uri})
        assert response.status_code == 200
        assert json.loads(response.data)["status"] == "Model updated successfully"