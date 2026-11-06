"""Engram n-gram hashing: the table layout and the per-position hash ids.

A 1:1 port of ``inference/engram.py`` (read-only oracle): the compressed token map, the per-(layer,
n-gram size, head) prime buckets, the per-layer hash multipliers, and the rolling-xor hash itself.

What has to match exactly, because a single wrong prime or multiplier rehashes the whole 189 GiB table:

* **compressed token map** -- NFKC/NFD/strip-accents/lowercase/whitespace-collapse over each token's
  decoded text, with a private-use sentinel so a token that is exactly one space survives.  Its size
  must equal ``engram_compressed_vocab_size`` (99,092 for this tokenizer) because every multiplier is
  bounded by it.
* **prime buckets** -- drawn from ``engram_vocab_size - 1`` upward per ``(layer, n-gram, head)``,
  never reused; the layer's row count is the *sum* of its primes (the S0 probe verified both tables).
* **multipliers** -- ``numpy.random.default_rng(10007 * layer_id).integers(0, bound, (max_ngram,)) *
  2 + 1`` where ``bound = (int64max // vocab) // 2``.
* **hash** -- ``rolling = tokens[0] * mult[0]``; for ``k >= 1``: ``rolling ^= tokens[k] * mult[k]``,
  then ``rolling % prime[k-1] + offset[k-1]``; tokens look back over the *compressed* ids with
  ``pad_id`` substituted once the sequence starts or a dead token (image span) is crossed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import Dsv41Config

__all__ = [
    "DEAD",
    "find_next_prime",
    "build_compressed_token_map",
    "compute_hash_multipliers",
    "compute_primes",
    "primes_rows",
    "EngramHash",
]

DEAD = -1


def find_next_prime(start: int, seen: set) -> int:
    from sympy import isprime

    candidate = start + 1
    while not isprime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> Tuple[List[int], int]:
    """``inference/engram.py:build_compressed_token_map`` -- tokens that normalize alike collapse."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: Sequence[int], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """One odd multiplier per (layer, lookback), from a per-layer RNG so layers hash differently."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


def compute_primes(
    layer_ids: Sequence[int], max_ngram_size: int, n_heads: int, vocab_base: int
) -> Tuple[Tuple[Tuple[Tuple[int, ...], ...], ...], List[int]]:
    """``EngramLayout.from_args`` prime draw, plus each layer's row count (the sum of its primes)."""
    primes: List[Tuple[Tuple[int, ...], ...]] = []
    seen: set = set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes = []
            current = int(vocab_base) - 1
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    rows = [int(sum(p for per in layer for p in per)) for layer in primes]
    return tuple(primes), rows


def primes_rows(cfg: Dsv41Config) -> List[int]:
    """Row count per engram layer for this config (what the table must have)."""
    _, rows = compute_primes(
        cfg.engram_layer_ids, cfg.engram_max_ngram_size, cfg.engram_n_heads, cfg.engram_vocab_size
    )
    return rows


class EngramHash:
    """``inference/engram.py:NgramHashState`` -- maps each position to its n-gram hash ids.

    ``tokenizer`` is only needed to build the compressed token map; an already-built
    ``(lookup, vocab_size)`` pair can be passed instead (the test does that to avoid re-normalizing
    129,280 tokens per case).
    """

    def __init__(
        self,
        cfg: Dsv41Config,
        tokenizer=None,
        lookup: Optional[Sequence[int]] = None,
        vocab_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        bsz: int = 1,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = device
        if lookup is None or vocab_size is None:
            if tokenizer is None:
                raise ValueError("EngramHash needs a tokenizer or a prebuilt (lookup, vocab_size)")
            lookup, vocab_size = build_compressed_token_map(tokenizer)
        if int(vocab_size) != int(cfg.engram_compressed_vocab_size):
            raise ValueError(
                f"compressed vocab {vocab_size} != engram_compressed_vocab_size "
                f"{cfg.engram_compressed_vocab_size}; every hash multiplier derives from it, so a "
                "mismatch would silently rehash the whole table"
            )
        self.vocab_size = int(vocab_size)
        self.pad_id = int(lookup[cfg.engram_pad_token_id])

        primes, _rows = compute_primes(
            cfg.engram_layer_ids,
            cfg.engram_max_ngram_size,
            cfg.engram_n_heads,
            cfg.engram_vocab_size,
        )
        self.primes = torch.tensor(primes, dtype=torch.int64, device=device)
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.offsets = torch.tensor(np.array(offsets), dtype=torch.int64, device=device)
        self.multipliers = compute_hash_multipliers(
            cfg.engram_layer_ids, cfg.engram_max_ngram_size, self.vocab_size
        ).to(device)
        self.token_map = torch.tensor(list(lookup), dtype=torch.int64, device=device)
        seq = int(max_seq_len or cfg.max_seq_len)
        self.cache = torch.empty(bsz, seq, dtype=torch.int64, device=device)

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, start_pos: int, token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``NgramHashState.forward`` -- hash ids ``[b, s, n_engram_layers, n_hash_cols]``."""
        # the caller may hand us ids on another device (vLLM passes NPU tensors while the hash state
        # lives wherever it was built); the lookup table and the cache are the authority here
        if input_ids.device != self.cache.device:
            input_ids = input_ids.to(self.cache.device)
        if token_mask is not None and token_mask.device != self.cache.device:
            token_mask = token_mask.to(self.cache.device)
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, torch.full_like(compressed, DEAD))
        self.cache[:batch, start_pos : start_pos + seqlen] = compressed

        positions = torch.arange(
            start_pos, start_pos + seqlen, device=input_ids.device
        ).expand(batch, seqlen)
        tokens: List[torch.Tensor] = []
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(int(self.cfg.engram_max_ngram_size)):
            source = self.cache[:batch].gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == DEAD)
            tokens.append(torch.where(blocked, torch.full_like(source, self.pad_id), source))
        tokens = torch.stack(tokens, dim=-1)  # [b, s, max_ngram_size]

        products = tokens.unsqueeze(2) * self.multipliers  # [b,s,n_layers,max_ngram]
        rolling, hashes = products[..., 0], []
        for i in range(1, int(self.cfg.engram_max_ngram_size)):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets
