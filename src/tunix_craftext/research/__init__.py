"""Research-only PPO mechanics kept outside the production Tunix Agentic path.

The modules in this package are useful for unit-level math checks, notebooks and
small Flax/Optax smoke updates. They are not the production trainer boundary:
production GRPO/PPO must go through Tunix ``RLCluster`` and Agentic learners.
"""

from .algorithm_registry import AlgorithmSpec, LossOutput, PpoLossBatch, get_algorithm
from .algorithms import (
    generalized_advantage_estimation,
    masked_token_ppo_loss,
    masked_token_returns,
    ppo_loss,
)
from .learner import (
    ActorCritic,
    PromptConditionedTokenActorCritic,
    create_state,
    create_token_state,
    full_token_ppo_update,
    ppo_update,
    token_actor_critic_outputs,
    token_ppo_update,
)
from .llm_ppo import (
    LlmPpoEvaluation,
    evaluate_llm_actor_critic_ppo,
    evaluate_separate_llm_actor_critic_ppo,
)

__all__ = [
    "ActorCritic",
    "AlgorithmSpec",
    "LossOutput",
    "LlmPpoEvaluation",
    "PpoLossBatch",
    "PromptConditionedTokenActorCritic",
    "create_state",
    "create_token_state",
    "evaluate_llm_actor_critic_ppo",
    "evaluate_separate_llm_actor_critic_ppo",
    "full_token_ppo_update",
    "generalized_advantage_estimation",
    "get_algorithm",
    "masked_token_ppo_loss",
    "masked_token_returns",
    "ppo_loss",
    "ppo_update",
    "token_actor_critic_outputs",
    "token_ppo_update",
]
