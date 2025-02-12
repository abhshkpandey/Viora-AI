from flask import Flask, request, jsonify
import torch
from models.model import VSLMModel

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VSLMModel().to(device)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_tensor = torch.tensor(data["input"]).float().to(device)
    output = model(input_tensor)
    return jsonify({"output": output.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
