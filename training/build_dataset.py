
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


DATA_DIR = "data/shards"
SHARD_SIZE = 10_000_000   # tokens per shard
TARGET_TOKENS = 800_000_000


def write_shard(tokens, shard_id):

    filename = os.path.join(DATA_DIR, f"shard_{shard_id:05d}.bin")

    arr = np.array(tokens, dtype=np.uint16)

    arr.tofile(filename)


def tokenize_stream(dataset, enc, token_buffer, shard_id):

    for sample in dataset:

        text = sample["text"]

        if not text or len(text) < 20:
            continue

        tokens = enc.encode(text)

        token_buffer.extend(tokens)

        while len(token_buffer) >= SHARD_SIZE:

            shard_tokens = token_buffer[:SHARD_SIZE]

            write_shard(shard_tokens, shard_id)

            token_buffer[:] = token_buffer[SHARD_SIZE:]

            shard_id += 1

        if shard_id * SHARD_SIZE >= TARGET_TOKENS:
            return shard_id

    return shard_id


def build_dataset():

    os.makedirs(DATA_DIR, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")

    token_buffer = []

    shard_id = 0

    print("Streaming Wikipedia : ")

    wiki = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True
    )

    shard_id = tokenize_stream(wiki, enc, token_buffer, shard_id)

    if shard_id * SHARD_SIZE >= TARGET_TOKENS:
        return

    print("Streaming TinyStories : ")

    tiny = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True
    )

    shard_id = tokenize_stream(tiny, enc, token_buffer, shard_id)

    if shard_id * SHARD_SIZE >= TARGET_TOKENS:
        return

    print("Streaming OpenWebText : ")

    openweb = load_dataset(
        "openwebtext",
        split="train",
        streaming=True
    )

    shard_id = tokenize_stream(openweb, enc, token_buffer, shard_id)

    print("Dataset creation complete.")
    print(f"Created {shard_id} shards.")
    print(f"Total tokens ≈ {shard_id * SHARD_SIZE}")
    

if __name__ == "__main__":
    build_dataset()
