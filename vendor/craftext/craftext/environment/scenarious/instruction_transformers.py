"""Instruction transformation hooks used before scenario materialization."""

from abc import ABC, abstractmethod
import json
import logging
from typing import List
from craftext.environment.craftext_constants import plans_path

logger = logging.getLogger(__name__)

class AbstractInstructionTransformer(ABC):
    @abstractmethod
    def transform(self, instructions_list: List[str]) -> List[str]:
        pass

class DefaultInstructionTransformer(AbstractInstructionTransformer):
    def transform(self, instructions_list: List[str]) -> List[str]:
        return instructions_list

class PlansInstructionTransformer(AbstractInstructionTransformer):
    def __init__(self, plan_file_path: str = plans_path):
        self.plan_file_path = plan_file_path
        with open(self.plan_file_path, 'r', encoding='utf-8') as f:
            self.action_plans = json.load(f)
        logger.info(f"Loaded action plans from {self.plan_file_path}")

    def transform(self, instructions_list: List[str]) -> List[str]:
        updated_instructions = [self.action_plans.get(instr, "none") for instr in instructions_list]
        logger.info("Using preloaded plans in craftext_scenarios.py")
        return updated_instructions
