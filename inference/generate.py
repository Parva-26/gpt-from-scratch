
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import tiktoken
import argparse
from model.gpt import GPT, GPTConfig
import yaml


def load_model(checkpoint_path, config_path, device):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    gpt_config = GPTConfig(**config["model"])

    model = GPT(gpt_config)

    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model"])

    model.to(device)

    model.eval()

    return model


def generate_text(model, prompt, max_new_tokens, temperature, top_k, device):

    enc = tiktoken.get_encoding("gpt2")

    tokens = enc.encode(prompt)

    x = torch.tensor(tokens, dtype=torch.long)[None, :].to(device)

    for _ in range(max_new_tokens):
      x_cond = x[:, -model.config.block_size:]

      logits, _ = model(x_cond)

      logits = logits[:, -1, :] / temperature

      if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("Inf")

      probs = torch.softmax(logits, dim=-1)

      next_token = torch.multinomial(probs, num_samples=1)

      x = torch.cat((x, next_token), dim=1)

    enc = tiktoken.get_encoding("gpt2")

    output = enc.decode(x[0].tolist())

    return output


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/small_gpt.yaml")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.checkpoint, args.config, device)

    output = generate_text(
        model,
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_k,
        device
    )

    print("\nGenerated Text:\n")
    print(output)


if __name__ == "__main__":
    main()
