
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import TransformerBlock


class GPTConfig:

    def __init__(
        self,
        vocab_size=50257,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1
    ):

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.n_embd,
                config.n_head,
                config.dropout
            )
            for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None):

        B, T = idx.shape

        positions = torch.arange(
            0, T,
            device=idx.device
        ).unsqueeze(0)

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)

        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss
