import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(preferred_device="cuda"):
    if preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")