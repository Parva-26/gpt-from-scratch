
import torch


def configure_optimizers(model, lr=3e-4, weight_decay=0.1):

    decay = []
    no_decay = []

    for name, param in model.named_parameters():

        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    optim_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    return optimizer
