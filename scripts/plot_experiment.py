import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_latest_experiment():
    log_dir = Path("experiments/logs")

    log_files = sorted(
        log_dir.glob("*.json")
    )

    if not log_files:
        raise FileNotFoundError(
            "No experiment logs found."
        )

    latest_log = log_files[-1]

    with open(latest_log) as f:
        data = json.load(f)

    return latest_log, data


def main():
    log_path, data = load_latest_experiment()

    steps = [item["step"] for item in data]

    train_losses = [
        item["train_loss"]
        for item in data
    ]

    val_losses = [
        item["val_loss"]
        for item in data
    ]

    learning_rates = [
        item["learning_rate"]
        for item in data
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        steps,
        train_losses,
        label="Train Loss",
    )

    plt.plot(
        steps,
        val_losses,
        label="Validation Loss",
    )

    plt.xlabel("Training Step")
    plt.ylabel("Loss")

    plt.title(
        "MiniGPT Experiment Loss Curves"
    )

    plt.legend()

    plt.grid(True)

    plt.savefig(
        "experiment_loss_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        steps,
        learning_rates,
    )

    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")

    plt.title(
        "Learning Rate Schedule"
    )

    plt.grid(True)

    plt.savefig(
        "learning_rate_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"Loaded experiment log: "
        f"{log_path}"
    )

    print(
        "Saved experiment_loss_curve.png"
    )

    print(
        "Saved learning_rate_curve.png"
    )


if __name__ == "__main__":
    main()