"""Improved conversion of json to csv."""
import json

# time it
import time

import pandas as pd

start = time.time()

file = "zip_proc.json"
with open(file) as f:
    data = json.load(f)

df = pd.DataFrame(columns=["index", "frame", "detection_class", "detection_score"])

for y, number in enumerate(data):
    detections = data[number]["detection_class"]
    prob = data[number]["prob"]
    for x, p in zip(detections, prob):
        df.loc[y] = [y, number, x, p]

df.to_csv(f"{file.split('.')[0]}.csv", index=False)

print(f"Time: {time.time() - start}")
