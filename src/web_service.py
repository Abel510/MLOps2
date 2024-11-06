import mlflow
from flask import Flask, request, jsonify

app = Flask(__name__)

mlflow.set_tracking_uri("http://localhost:5000")

current_model = mlflow.sklearn.load_model("models:/california-housing/1")

next_model = mlflow.sklearn.load_model("models:/california-housing/1")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = [[data[key] for key in sorted(data.keys())]]

    p = 0.9
    if mlflow.active_run():
        p = mlflow.active_run().data.params.get("canary_probability", p)
    result = p * current_model.predict(input_data) + (1 - p) * next_model.predict(input_data)
    return jsonify({"prediction": float(result[0])})


@app.route("/update-model", methods=["POST"])
def update_model():
    new_model = mlflow.sklearn.load_model(request.get_json()["model_uri"])
    global next_model
    next_model = new_model
    return jsonify({"status": "Model updated successfully"})

@app.route("/accept-next-model", methods=["POST"])
def accept_next_model():
    global current_model, next_model
    current_model = next_model
    next_model = mlflow.sklearn.load_model("models:/boston-housing/staging")
    return jsonify({"status": "Next model accepted as current"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

    