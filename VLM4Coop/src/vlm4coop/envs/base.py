"""Small contracts for environment-owned metadata; no learning/backend imports."""
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class TeacherSpec:
    actions: tuple
    system_prompt: Callable[[int], str]
    user_prompt: Callable[[int], str]
    cache_identity: Callable[[], dict]

    @property
    def n_actions(self):
        return len(self.actions)


class EnvironmentAdapter(Protocol):
    name: str
    def epymarl_args(self, n_agents, seed, config) -> dict: ...
    def prepare(self, n_agents, config): ...
    def validate_info(self, args, info): ...
    def episode_fields(self, info) -> dict: ...
    def teacher_spec(self) -> TeacherSpec: ...
