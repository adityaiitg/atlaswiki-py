"""Scalar Quantization (SQ8: float32 -> int8) for compact vector search."""

from __future__ import annotations
import math
import struct
from typing import Sequence


class Sq8Vector:
    """An 8-bit scalar quantized vector representation."""

    __slots__ = ("quantized", "min_val", "max_val", "norm")

    def __init__(
        self, quantized: bytes, min_val: float, max_val: float, norm: float
    ) -> None:
        self.quantized = quantized
        self.min_val = min_val
        self.max_val = max_val
        self.norm = norm

    @classmethod
    def from_float(cls, vec: Sequence[float]) -> Sq8Vector:
        if not vec:
            return cls(b"", 0.0, 0.0, 0.0)

        min_val = min(vec)
        max_val = max(vec)
        range_val = max_val - min_val

        # Compute Euclidean norm
        sum_sq = sum(x * x for x in vec)
        norm = math.sqrt(sum_sq)

        if range_val < 1e-7:
            # All values virtually identical
            q_bytes = bytes([128] * len(vec))
            return cls(q_bytes, min_val, max_val, norm)

        scale = 255.0 / range_val
        q_list = []
        for x in vec:
            clamped = max(min_val, min(max_val, x))
            q = int(round((clamped - min_val) * scale))
            q = max(0, min(255, q))
            q_list.append(q)

        return cls(bytes(q_list), min_val, max_val, norm)

    def to_float(self) -> list[float]:
        if not self.quantized:
            return []
        range_val = self.max_val - self.min_val
        scale = range_val / 255.0
        return [self.min_val + (b * scale) for b in self.quantized]

    def dot_product(self, other: Sq8Vector) -> float:
        """Approximate dot product between two quantized vectors."""
        if len(self.quantized) != len(other.quantized):
            raise ValueError("Vector dimensionality mismatch")

        scale_a = (self.max_val - self.min_val) / 255.0
        scale_b = (other.max_val - other.min_val) / 255.0

        raw_dot = sum(a * b for a, b in zip(self.quantized, other.quantized))
        sum_a = sum(self.quantized)
        sum_b = sum(other.quantized)
        n = len(self.quantized)

        dot = (
            (scale_a * scale_b * raw_dot)
            + (scale_a * other.min_val * sum_a)
            + (scale_b * self.min_val * sum_b)
            + (n * self.min_val * other.min_val)
        )
        return dot

    def cosine_similarity(self, other: Sq8Vector) -> float:
        """Compute cosine similarity between two quantized vectors."""
        if self.norm < 1e-7 or other.norm < 1e-7:
            return 0.0
        dp = self.dot_product(other)
        sim = dp / (self.norm * other.norm)
        return max(-1.0, min(1.0, sim))

    def asymmetric_cosine_similarity(self, query: Sequence[float]) -> float:
        """Compute asymmetric cosine similarity between unquantized query and quantized vector."""
        if not query or len(query) != len(self.quantized) or self.norm < 1e-7:
            return 0.0

        q_norm = math.sqrt(sum(x * x for x in query))
        if q_norm < 1e-7:
            return 0.0

        scale = (self.max_val - self.min_val) / 255.0
        dot = 0.0
        for q_val, b in zip(query, self.quantized):
            approx_val = self.min_val + (b * scale)
            dot += q_val * approx_val

        sim = dot / (q_norm * self.norm)
        return max(-1.0, min(1.0, sim))

    def to_bytes(self) -> bytes:
        """Pack quantized vector metadata and bytes into binary buffer."""
        # 4 bytes min, 4 bytes max, 4 bytes norm, followed by quantized bytes
        header = struct.pack(
            "<fffI", self.min_val, self.max_val, self.norm, len(self.quantized)
        )
        return header + self.quantized

    @classmethod
    def from_bytes(cls, buf: bytes) -> Sq8Vector:
        """Unpack binary buffer to Sq8Vector."""
        min_val, max_val, norm, length = struct.unpack_from("<fffI", buf, 0)
        q_bytes = buf[16 : 16 + length]
        return cls(q_bytes, min_val, max_val, norm)
