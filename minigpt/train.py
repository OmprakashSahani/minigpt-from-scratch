import torch

from minigpt import config
from minigpt.experiment_logger import ExperimentLogger
from minigpt.metrics import compute_perplexity
from minigpt.model import MiniGPT
from minigpt.tokenizer_factory import build_tokenizer
from minigpt.utils import (
    count_parameters,
    count_trainable_parameters,
    get_device,
)


def get_batch(split_data, block_size, batch_size):
    ix = torch.randint(
        0,
        len(split_data) - block_size - 1,
        (batch_size,),
        device=split_data.device,
    )

    x = torch.stack([split_data[i:i + block_size] for i in ix])

    y = torch.stack(
        [split_data[i + 1:i + block_size + 1] for i in ix]
    )

    return x, y


@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    block_size,
    batch_size,
    eval_iters=20,
):
    model.eval()

    out = {}

    for split, split_data in [
        ("train", train_data),
        ("val", val_data),
    ]:
        losses = []

        for _ in range(eval_iters):
            x, y = get_batch(
                split_data,
                block_size,
                batch_size,
            )

            _, loss, _ = model(x, y)

            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()

    return out


def build_checkpoint_dict(
    model,
    optimizer,
    scheduler,
    scaler,
    tokenizer,
):
    return {
        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "tokenizer_type": config.TOKENIZER_TYPE,

        "vocab": getattr(
            tokenizer,
            "vocab",
            None,
        ),

        "inverse_vocab": getattr(
            tokenizer,
            "inverse_vocab",
            None,
        ),

        "stoi": getattr(
            tokenizer,
            "stoi",
            None,
        ),

        "itos": getattr(
            tokenizer,
            "itos",
            None,
        ),

        "embed_dim": config.EMBED_DIM,
        "block_size": config.BLOCK_SIZE,
        "num_heads": config.NUM_HEADS,
        "num_layers": config.NUM_LAYERS,
    }


def main():
    text = open("data/input.txt").read()

    tokenizer = build_tokenizer(text)

    device = get_device(config.DEVICE)

    use_amp = (
        config.USE_AMP and
        device.type == "cuda"
    )

    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")

    logger = ExperimentLogger(
        config.EXPERIMENT_NAME
    )

    data = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
        device=device,
    )

    split_idx = int(0.9 * len(data))

    train_data = data[:split_idx]
    val_data = data[split_idx:]

    model = MiniGPT(
        vocab_size=len(tokenizer.vocab)
        if hasattr(tokenizer, "vocab")
        else tokenizer.vocab_size,

        embed_dim=config.EMBED_DIM,
        block_size=config.BLOCK_SIZE,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
    ).to(device)

    if config.RESUME_CHECKPOINT:
        print(
            f"Loading checkpoint: "
            f"{config.CHECKPOINT_PATH}"
        )

        checkpoint = torch.load(
            config.CHECKPOINT_PATH,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print("Loaded model weights")

    print(
        f"Total parameters: "
        f"{count_parameters(model):,}"
    )

    print(
        f"Trainable parameters: "
        f"{count_trainable_parameters(model):,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )

    if config.RESUME_CHECKPOINT:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        print("Loaded optimizer state")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.STEPS,
    )

    if config.RESUME_CHECKPOINT:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        print("Loaded scheduler state")

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp,
    )

    if config.RESUME_CHECKPOINT:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        print("Loaded AMP scaler state")

    train_loss_history = []
    val_loss_history = []

    for step in range(config.STEPS):
        x, y = get_batch(
            train_data,
            config.BLOCK_SIZE,
            config.BATCH_SIZE,
        )

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(
            enabled=use_amp,
        ):
            _, loss, _ = model(x, y)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.GRAD_CLIP,
        )

        scaler.step(optimizer)

        scaler.update()

        if config.USE_SCHEDULER:
            scheduler.step()

        if step % 50 == 0:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                block_size=config.BLOCK_SIZE,
                batch_size=config.BATCH_SIZE,
            )

            train_perplexity = compute_perplexity(
                losses["train"]
            )

            val_perplexity = compute_perplexity(
                losses["val"]
            )

            train_loss_history.append(
                losses["train"]
            )

            val_loss_history.append(
                losses["val"]
            )

            print(
                f"step {step}, "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, "
                f"train ppl {train_perplexity:.2f}, "
                f"val ppl {val_perplexity:.2f}, "
                f"lr {optimizer.param_groups[0]['lr']:.6f}"
            )

            logger.log(
                {
                    "step": step,

                    "train_loss": losses["train"],

                    "val_loss": losses["val"],

                    "train_perplexity": train_perplexity,

                    "val_perplexity": val_perplexity,

                    "learning_rate": (
                        optimizer.param_groups[0]["lr"]
                    ),
                }
            )

        if (
            step > 0 and
            step % config.SAVE_CHECKPOINT_EVERY == 0
        ):
            checkpoint_path = (
                f"checkpoint_step_{step}.pt"
            )

            torch.save(
                build_checkpoint_dict(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    tokenizer,
                ),

                checkpoint_path,
            )

            print(
                f"Saved intermediate checkpoint: "
                f"{checkpoint_path}"
            )

    torch.save(
        build_checkpoint_dict(
            model,
            optimizer,
            scheduler,
            scaler,
            tokenizer,
        ),

        "model.pt",
    )

    logger.save()

    print("Saved model to model.pt")

    with open(
        "train_loss_history.txt",
        "w",
    ) as f:
        for value in train_loss_history:
            f.write(f"{value}\n")

    with open(
        "val_loss_history.txt",
        "w",
    ) as f:
        for value in val_loss_history:
            f.write(f"{value}\n")

    print(
        "Saved train loss history "
        "to train_loss_history.txt"
    )

    print(
        "Saved validation loss history "
        "to val_loss_history.txt"
    )


if __name__ == "__main__":
    main()