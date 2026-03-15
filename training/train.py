
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml
from model.gpt import GPT, GPTConfig
from training.dataset import ShardedDataset
from training.optimizer import configure_optimizers
import torch.cuda.amp as amp
import time

def load_config(path):

    with open(path) as f:
        config = yaml.safe_load(f)

    return config


CHECKPOINT_DIR = "/content/drive/MyDrive/gpt_checkpoints"


def save_checkpoint(model, optimizer, step):

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    path = os.path.join(CHECKPOINT_DIR, f"ckpt_{step}.pt")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step
        },
        path
    )

    print(f"Checkpoint saved: {path}")


def load_latest_checkpoint(model, optimizer):

    if not os.path.exists(CHECKPOINT_DIR):
        return 0

    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]

    if len(checkpoints) == 0:
        return 0

    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))

    latest = checkpoints[-1]

    path = os.path.join(CHECKPOINT_DIR, latest)

    ckpt = torch.load(path)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    print(f"Resuming from {latest}")

    return ckpt["step"]

def estimate_loss(model, val_loader, device, batch_size):

    model.eval()

    losses = []

    with torch.no_grad():

        for _ in range(50):

            x, y = val_loader.next_batch(batch_size)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            _, loss = model(x, y)

            losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


def train():

    config = load_config("configs/small_gpt.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.set_float32_matmul_precision("high")

    from training.dataset import ShardedDataset

    train_loader = ShardedDataset(
        "/content/shards",
        config["model"]["block_size"]
    )

    val_loader = train_loader

    gpt_config = GPTConfig(**config["model"])

    model = GPT(gpt_config).to(device)

    optimizer = configure_optimizers(
        model,
        config["optimizer"]["lr"],
        config["optimizer"]["weight_decay"]
    )

    scaler = amp.GradScaler(enabled=(device == "cuda"))

    grad_accum = config["training"]["grad_accum_steps"]

    max_steps = config["training"]["max_steps"]

    start_step = load_latest_checkpoint(model, optimizer)

    for step in range(start_step, max_steps):

      t0 = time.time()

      optimizer.zero_grad()

      total_loss = 0

      for micro_step in range(grad_accum):
        x, y = train_loader.next_batch(
            config["training"]["batch_size"]
        )

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with amp.autocast(enabled=(device == "cuda")):
            _, loss = model(x, y)
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        total_loss += loss.item()

    
      scaler.unscale_(optimizer)

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      scaler.step(optimizer)

      scaler.update()

      t1 = time.time()

      tokens_processed = (
          config["training"]["batch_size"]
          * config["model"]["block_size"]
          * grad_accum
      )

      tokens_per_sec = tokens_processed / (t1 - t0)

      if step % 100 == 0:
        print(
            f"step {step} | loss {total_loss:.4f} | tok/sec {tokens_per_sec:.2f}"
        )

      if step % config["training"]["eval_interval"] == 0:
        val_loss = estimate_loss(
            model,
            val_loader,
            device,
            config["training"]["batch_size"]
        )
        print(f"validation loss: {val_loss:.4f}")

      if step % config["training"]["save_interval"] == 0:
        save_checkpoint(model, optimizer, step)

if __name__ == "__main__":
    train()
