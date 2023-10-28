import json

with open("results.json", "r") as f:
    results = json.load(f)

acc = sum(results) / len(results)

print("acc:", acc)