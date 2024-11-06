import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()
model_name = "california-housing"
model_version = 1

client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="staging"
)

print(f"Model {model_name} version {model_version} transitioned to staging.")