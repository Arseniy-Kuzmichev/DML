import json
import sys

import yaml


def load_metrics():
    with open("metrics/metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_model():
    metrics = load_metrics()
    params = load_params()

    accuracy = metrics["accuracy"]
    accuracy_min = params["accuracy_min"]

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Minimum required accuracy: {accuracy_min:.4f}")

    if accuracy < accuracy_min:
        print("Model validation failed: accuracy is below the required threshold.")
        sys.exit(1)

    print("Model validation passed!")


if __name__ == "__main__":
    validate_model()
