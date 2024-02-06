import numpy as np
import sys
import os
import torch
from flask import Flask, request, jsonify
import json

from src_classification.model_cls import LunaModel

app = Flask(__name__)

model = LunaModel()
model.load_state_dict(torch.load(sys.argv[1], map_location="cpu")["model_state"])

model.eval()

def run_inference(in_tensor):
    with torch.no_grad():
        # model takes a batch and outputs a tuple (score, probabilities)
        out_tensor = model(in_tensor.unsqueeze(0))[1].squeeze(0)
    probabilities = out_tensor.tolist()
    out = {'probability_malignant' : probabilities[1]}
    
    return out

# expect a form submission (HTTP POST) at the "/predict" endpoint
@app.route("/predict", methods=["POST"])
def predict():
    meta = json.load(request.files["meta"]) # request will have one file called meta
    blob = request.files["blob"].read()
    in_tensor = torch.from_numpy(
        np.frombuffer(blob, dtype=np.float32)   # convert data from binary blob to torch
    )
    in_tensor = in_tensor.view(*meta["shape"])
    out = run_inference(in_tensor)

    return jsonify(out)     # encode response content as JSON

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
    print(sys.argv[1])

