import matplotlib.pyplot as plt


def main():
    with open("loss_history.txt") as f:
        losses = [float(line.strip()) for line in f if line.strip()]

    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("MiniGPT Training Loss")
    plt.grid(True)

    plt.savefig("loss_curve.png", dpi=200, bbox_inches="tight")
    print("Saved loss curve to loss_curve_previous.png")


if __name__ == "__main__":
    main()
