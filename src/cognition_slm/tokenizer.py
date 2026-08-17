"""A deterministic UTF-8 byte tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
BYTE_OFFSET = 3
VOCAB_SIZE = BYTE_OFFSET + 256


@dataclass(frozen=True)
class ByteTokenizer:
    """Tokenize text without a learned vocabulary or external files."""

    vocab_size: int = VOCAB_SIZE
    pad_id: int = PAD_ID
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID

    def __post_init__(self) -> None:
        if self.vocab_size < VOCAB_SIZE:
            raise ValueError("vocab_size must be at least 259")

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        byte_ids = [BYTE_OFFSET + value for value in text.encode("utf-8")]
        special_count = int(add_bos) + int(add_eos)
        if max_length is not None:
            if max_length <= special_count:
                raise ValueError("max_length must leave room for requested special tokens")
            byte_ids = byte_ids[: max_length - special_count]
        result: list[int] = []
        if add_bos:
            result.append(self.bos_id)
        result.extend(byte_ids)
        if add_eos:
            result.append(self.eos_id)
        return result

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        values: list[int] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in (self.pad_id, self.bos_id, self.eos_id):
                if skip_special_tokens:
                    continue
                raise ValueError("special tokens cannot be decoded as bytes")
            if not BYTE_OFFSET <= token_id < BYTE_OFFSET + 256:
                raise ValueError(f"token id outside byte vocabulary: {token_id}")
            values.append(token_id - BYTE_OFFSET)
        return bytes(values).decode("utf-8", errors="replace")
