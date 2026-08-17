"""Tiny decoder-only transformer with behavior-level auxiliary heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .tokenizer import PAD_ID


@dataclass
class ModelOutput:
    logits: Tensor
    task_logits: Tensor
    error_logits: Tensor
    confidence_logits: Tensor
    loss: Tensor | None = None
    lm_loss: Tensor | None = None
    task_loss: Tensor | None = None
    error_loss: Tensor | None = None
    confidence_loss: Tensor | None = None


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.block_size, config.block_size, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores.masked_fill(self.causal_mask[:length, :length], float("-inf"))
        if attention_mask is not None:
            key_padding = attention_mask[:, None, None, :].eq(0)
            scores = scores.masked_fill(key_padding, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(attended))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.ln_1(x), attention_mask)
        return x + self.mlp(self.ln_2(x))


class CognitionSLM(nn.Module):
    """Small language model plus three observable-behavior classifiers."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.task_head = nn.Linear(config.n_embd, len(config.task_types))
        self.error_head = nn.Linear(config.n_embd, len(config.error_categories))
        self.confidence_head = nn.Linear(config.n_embd, 3)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _pool_last(self, hidden: Tensor, attention_mask: Tensor | None) -> Tensor:
        if attention_mask is None:
            return hidden[:, -1]
        positions = attention_mask.sum(dim=1).long().clamp_min(1) - 1
        batch_positions = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_positions, positions]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        task_labels: Tensor | None = None,
        error_labels: Tensor | None = None,
        confidence_labels: Tensor | None = None,
    ) -> ModelOutput:
        batch, length = input_ids.shape
        if length > self.config.block_size:
            raise ValueError(
                f"sequence length {length} exceeds block_size {self.config.block_size}"
            )
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        hidden = self.dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        pooled = self._pool_last(hidden, attention_mask)
        task_logits = self.task_head(pooled)
        error_logits = self.error_head(pooled)
        confidence_logits = self.confidence_head(pooled)

        losses: dict[str, Tensor] = {}
        if length > 1:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            losses["lm_loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=PAD_ID,
            )
        if task_labels is not None:
            losses["task_loss"] = F.cross_entropy(task_logits, task_labels)
        if error_labels is not None:
            losses["error_loss"] = F.cross_entropy(error_logits, error_labels)
        if confidence_labels is not None:
            losses["confidence_loss"] = F.cross_entropy(confidence_logits, confidence_labels)
        total_loss = None
        if losses:
            total_loss = losses.get("lm_loss", torch.zeros((), device=input_ids.device))
            total_loss = total_loss + 0.25 * losses.get(
                "task_loss", torch.zeros((), device=input_ids.device)
            )
            total_loss = total_loss + 0.25 * losses.get(
                "error_loss", torch.zeros((), device=input_ids.device)
            )
            total_loss = total_loss + 0.25 * losses.get(
                "confidence_loss", torch.zeros((), device=input_ids.device)
            )
        return ModelOutput(
            logits=logits,
            task_logits=task_logits,
            error_logits=error_logits,
            confidence_logits=confidence_logits,
            loss=total_loss,
            lm_loss=losses.get("lm_loss"),
            task_loss=losses.get("task_loss"),
            error_loss=losses.get("error_loss"),
            confidence_loss=losses.get("confidence_loss"),
        )
