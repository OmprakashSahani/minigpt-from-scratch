import json
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    def __init__(self, experiment_name):
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.run_name = (
            f"{experiment_name}_{timestamp}"
        )

        self.log_dir = Path(
            "experiments/logs"
        )

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metrics_path = (
            self.log_dir /
            f"{self.run_name}.json"
        )

        self.metrics = []

    def log(self, metrics_dict):
        self.metrics.append(metrics_dict)

    def save(self):
        with open(
            self.metrics_path,
            "w",
        ) as f:
            json.dump(
                self.metrics,
                f,
                indent=2,
            )

        print(
            f"Saved experiment log: "
            f"{self.metrics_path}"
        )