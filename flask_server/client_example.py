import json
import requests
import io
import torch

image, cl, series_id, positive = torch.load("flask_server/cls_val_example.pt")

meta = io.StringIO(json.dumps({"shape": list(image.shape)}))
data = io.BytesIO(bytearray(image.numpy()))

r = requests.post(
    "http://localhost:8000/predict",
    files={"meta": meta, "blob": data}
)

response = json.loads(r.content)

print("Model predicted probability of being malignant: ", response['probability_malignant'])