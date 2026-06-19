"""Scenario instruction processors (raw and encoded) for CrafText."""

from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

import numpy as np

ProcessedT = TypeVar("ProcessedT")


class ScenarioProcessor(ABC, Generic[ProcessedT]):
    @abstractmethod
    def process(self, instructions: List[str]) -> ProcessedT:
        """Process a list of instructions into runtime payload."""
        ...


class EncodedProcessor(ScenarioProcessor[np.ndarray]):
    def __init__(self, encode_model):
        self.encode_model = encode_model

    def process(self, instructions: List[str]) -> np.ndarray:
        encoded_instructions = self.encode_model.encode(instructions)
        # Assuming encode_model might return multiple variants per instruction
        if len(instructions) == 0:
            return np.array([])
        num_variants = len(encoded_instructions) // len(instructions)
        assert len(encoded_instructions) == len(instructions) * num_variants, \
            f"Unexpected size of encoded instructions ({len(encoded_instructions)} vs {len(instructions)}). Ensure encode_model is consistent."
        
        return encoded_instructions


class RawProcessor(ScenarioProcessor[List[None]]):
    def process(self, instructions: List[str]) -> List[None]:
        # For raw scenarios, embeddings are None and there's only one variant.
        return [None] * len(instructions)
