import matplotlib.pyplot as plt


def read_losses(path):
    with open(path) as f:
        return [float(line.strip()) for line in f if line.strip()]


def main():
    train_losses = read_losses("train_loss_history.txt")
    val_losses = read_losses("val_loss_history.txt")

    steps = [i * 50 for i in range(len(train_losses))]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_losses, label="Train Loss")
    plt.plot(steps, val_losses, label="Validation Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("MiniGPT Train vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig("loss_curve.png", dpi=200, bbox_inches="tight")
    print("Saved loss curve to loss_curve.png")


if __name__ == "__main__":
    main()