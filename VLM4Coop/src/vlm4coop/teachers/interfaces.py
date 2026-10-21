"""Common teacher result and structural inference interface."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass
class TeacherResult:
    probs: np.ndarray
    raw: str
    repaired: bool
    latency_seconds: float = 0.0
    generated_tokens: int = 0
    cache_hit: bool = False
    source: str = 'teacher'
    format_attempts: int = 1
    decision_id: str = ''



class Teacher(Protocol):
    def predict(self, image, n_agents, use_cache=True) -> TeacherResult: ...
