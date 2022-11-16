import json

import pandas as pd

# load the json file
with open("tiktok.json") as f:
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
df.to_csv("processed_tiktok.csv", index=False)
