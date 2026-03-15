
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml
from model.gpt import GPT, GPTConfig
from training.dataset import ShardedDataset
import math


def load_model(checkpoint, config_path, device):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    gpt_config = GPTConfig(**config["model"])

    model = GPT(gpt_config)

    ckpt = torch.load(checkpoint, map_location=device)

    model.load_state_dict(ckpt["model"])

    model.to(device)

    model.eval()

    return model, config


def evaluate(model, dataset, device, batch_size):

    losses = []

    with torch.no_grad():

        for i in range(200):
          if i % 20 == 0:
              print(f"Evaluating batch {i}/200")

          x, y = dataset.next_batch(batch_size)

          x = x.to(device)
          y = y.to(device)

          _, loss = model(x, y)

          losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)

    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


def main():

    checkpoint = "/content/drive/MyDrive/gpt_checkpoints/ckpt_53000.pt"
    config_path = "configs/small_gpt.yaml"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, config = load_model(checkpoint, config_path, device)

    dataset = ShardedDataset("/content/shards", config["model"]["block_size"])

    loss, ppl = evaluate(
        model,
        dataset,
        device,
        config["training"]["batch_size"]
    )

    print("Validation Loss:", loss)
    print("Perplexity:", ppl)


if __name__ == "__main__":
    main()
