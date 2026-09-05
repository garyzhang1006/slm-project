"""Tiny decoder-only transformer with behavior-level auxiliary heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

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


class RMSNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        hidden = 4 * n_embd
        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


def _rotate_half(x: Tensor) -> Tensor:
    first = x[..., ::2]
    second = x[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.use_rotary = config.architecture == "modern"
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.block_size, config.block_size, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        if self.use_rotary:
            inverse_frequency = 1.0 / (
                config.rope_theta
                ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
            )
            positions = torch.arange(config.block_size).float()
            frequencies = torch.outer(positions, inverse_frequency)
            embeddings = torch.repeat_interleave(frequencies, 2, dim=-1)
            self.register_buffer("rope_cos", embeddings.cos()[None, None, :, :], persistent=False)
            self.register_buffer("rope_sin", embeddings.sin()[None, None, :, :], persistent=False)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.use_rotary:
            cos = self.rope_cos[:, :, :length, :].to(dtype=q.dtype)
            sin = self.rope_sin[:, :, :length, :].to(dtype=q.dtype)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        allowed = None
        if attention_mask is not None:
            # SDPA boolean masks use True for allowed attention positions.
            allowed = (~self.causal_mask[:length, :length])[None, None, :, :]
            allowed = allowed & attention_mask[:, None, None, :].bool()
        attended = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=allowed,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=allowed is None,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(attended))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        norm = RMSNorm if config.architecture == "modern" else nn.LayerNorm
        self.ln_1 = norm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = norm(config.n_embd)
        if config.architecture == "modern":
            self.mlp = SwiGLU(config.n_embd, config.dropout)
        else:
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
        self.gradient_checkpointing = False
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = (
            nn.Embedding(config.block_size, config.n_embd)
            if config.architecture == "legacy"
            else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.final_norm = (
            RMSNorm(config.n_embd)
            if config.architecture == "modern"
            else nn.LayerNorm(config.n_embd)
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.task_head = nn.Linear(config.n_embd, len(config.task_types))
        self.error_head = nn.Linear(config.n_embd, len(config.error_categories))
        self.confidence_head = nn.Linear(config.n_embd, 3)
        self.apply(self._init_weights)
        if config.architecture == "modern":
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _pool_last(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None,
        pool_positions: Tensor | None,
    ) -> Tensor:
        if pool_positions is None:
            if attention_mask is None:
                positions = torch.full(
                    (hidden.size(0),), hidden.size(1) - 1, device=hidden.device, dtype=torch.long
                )
            else:
                positions = attention_mask.sum(dim=1).long().clamp_min(1) - 1
        else:
            if pool_positions.shape != (hidden.size(0),):
                raise ValueError("pool_positions must have shape (batch,)")
            positions = pool_positions.to(hidden.device, dtype=torch.long)
            if torch.any(positions < 0) or torch.any(positions >= hidden.size(1)):
                raise ValueError("pool_positions must point inside the sequence")
            if attention_mask is not None:
                batch_positions = torch.arange(hidden.size(0), device=hidden.device)
                if torch.any(attention_mask[batch_positions, positions] == 0):
                    raise ValueError("pool_positions cannot point to padding")
        batch_positions = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_positions, positions]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        pool_positions: Tensor | None = None,
        lm_loss_mask: Tensor | None = None,
        task_labels: Tensor | None = None,
        error_labels: Tensor | None = None,
        confidence_labels: Tensor | None = None,
    ) -> ModelOutput:
        if input_ids.ndim != 2 or input_ids.size(0) == 0 or input_ids.size(1) == 0:
            raise ValueError("input_ids must have non-empty shape (batch, sequence_length)")
        batch, length = input_ids.shape
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            attention_mask = attention_mask.to(input_ids.device)
            if torch.any((attention_mask != 0) & (attention_mask != 1)):
                raise ValueError("attention_mask must contain only zero and one")
            if torch.any(attention_mask[:, 0] == 0) or torch.any(
                attention_mask[:, 1:] > attention_mask[:, :-1]
            ):
                raise ValueError("attention_mask must describe non-empty, right-padded sequences")
        if length > self.config.block_size:
            raise ValueError(
                f"sequence length {length} exceeds block_size {self.config.block_size}"
            )
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding(positions)[None, :, :]
        hidden = self.dropout(hidden)
        # Unpadded batches can use SDPA's causal fast path without a dense mask.
        block_mask = attention_mask
        if attention_mask is not None and bool(attention_mask.bool().all()):
            block_mask = None
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, block_mask, use_reentrant=False)
            else:
                hidden = block(hidden, block_mask)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        pooled = self._pool_last(hidden, attention_mask, pool_positions)
        task_logits = self.task_head(pooled)
        error_logits = self.error_head(pooled)
        confidence_logits = self.confidence_head(pooled)

        losses: dict[str, Tensor] = {}
        if length > 1:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            token_losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=PAD_ID,
                reduction="none",
            ).view_as(shift_labels)
            valid_targets = shift_labels.ne(PAD_ID)
            if attention_mask is not None:
                valid_targets = valid_targets & attention_mask[:, 1:].bool()
            if lm_loss_mask is not None:
                if lm_loss_mask.shape != shift_labels.shape:
                    raise ValueError("lm_loss_mask must have shape (batch, sequence_length - 1)")
                valid_targets = valid_targets & lm_loss_mask.to(input_ids.device).bool()
            selected = token_losses.masked_select(valid_targets)
            # Empty supervision remains differentiable for accumulation/backward.
            losses["lm_loss"] = selected.sum() / valid_targets.sum().clamp_min(1)
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
