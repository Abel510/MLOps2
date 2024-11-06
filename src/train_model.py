import mlflow
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("california-housing")

with mlflow.start_run() as run:
    lr = LinearRegression()
    lr.fit(X, y)

    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_epochs", 100)
    mlflow.log_metric("mse", mean_squared_error(y, lr.predict(X)))

    mlflow.sklearn.log_model(lr, "model")

    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "california-housing")