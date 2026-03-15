
import os
import numpy as np
import torch


class ShardedDataset:

    def __init__(self, data_dir, block_size):

        self.data_dir = data_dir
        self.block_size = block_size

        self.shards = sorted(
            [s for s in os.listdir(data_dir) if s.endswith(".bin")]
        )

        assert len(self.shards) > 0, "No shard files found."

        print("Found shards:", len(self.shards))

        self.current_shard = 0
        self.tokens = self.load_shard(self.current_shard)

        self.pos = 0


    def load_shard(self, shard_index):

        path = os.path.join(self.data_dir, self.shards[shard_index])

        print(f"Loading shard {path}")

        tokens = np.fromfile(path, dtype=np.uint16)

        return tokens


    def next_batch(self, batch_size):

        B = batch_size
        T = self.block_size

        if self.pos + B * (T + 1) >= len(self.tokens):

            self.current_shard = (self.current_shard + 1) % len(self.shards)

            self.tokens = self.load_shard(self.current_shard)

            self.pos = 0

        chunk = self.tokens[self.pos:self.pos + B * (T + 1)]

        chunk = torch.from_numpy(chunk.astype(np.int64))

        chunk = chunk.view(B, T + 1)

        x = chunk[:, :-1]
        y = chunk[:, 1:]

        self.pos += B * T

        return x, y
