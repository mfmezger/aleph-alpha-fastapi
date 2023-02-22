"""In this file is used to converse the json file to a csv file."""
import json
import time

import pandas as pd

start = time.time()

file = "LT1_conv_proc.json"
# load the json file
with open(file) as f:
    data = json.load(f)

# print the json file
df = pd.DataFrame(columns=["index", "frame", "detection_class", "detection_score"])
y = 0
for d in data:
    # save number
    number = d
    detections = data[number]["detection_class"]
    prob = data[d]["prob"]
    for x in detections:
        for p in prob:

            df.loc[y] = [y, number, x, p]
            y += 1


# save df
df.to_csv(f"{file.split('.')[0]}.csv", index=False)
print(f"Time: {time.time() - start}")
