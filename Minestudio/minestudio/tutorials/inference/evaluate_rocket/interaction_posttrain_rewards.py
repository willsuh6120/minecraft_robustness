from dataclasses import dataclass
from typing import Dict, Mapping

from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_spec import InteractionBenchmarkTaskSpec
from minestudio.tutorials.inference.evaluate_rocket.interaction_success_checker import (
    InteractionBenchmarkSuccessTracker,
    InteractionSuccessResult,
)
from minestudio.tutorials.inference.evaluate_rocket.success_checker import event_delta


@dataclass
class RewardEvent:
    reward: float
    done: bool
    success: bool
    failure: bool
    reason: str
    metrics: Dict[str, float]


class InteractionPostTrainRewardTracker:
    def __init__(
        self,
        task_spec: InteractionBenchmarkTaskSpec,
        initial_info: Mapping,
        success_reward: float = 1.0,
        failure_penalty: float = -1.0,
        step_penalty: float = 0.0,
    ):
        self.task_spec = task_spec
        self.initial_info = dict(initial_info)
        self.success_tracker = InteractionBenchmarkSuccessTracker(task_spec, initial_info)
        self.success_reward = float(success_reward)
        self.failure_penalty = float(failure_penalty)
        self.step_penalty = float(step_penalty)
        self._already_succeeded = False
        self._already_failed = False

    def evaluate(self, info: Mapping, env_reward: float = 0.0) -> RewardEvent:
        result = self.success_tracker.update(info)
        reward = self.step_penalty + float(env_reward)
        done = False
        success = False
        failure = False
        reason = result.reason
        metrics = dict(result.progress)
        metrics["env_reward"] = float(env_reward)

        if result.supported and bool(result.success) and not self._already_succeeded:
            reward += self.success_reward
            done = True
            success = True
            self._already_succeeded = True
            return RewardEvent(reward=reward, done=done, success=success, failure=failure, reason=reason, metrics=metrics)

        task_name = self.task_spec.task_key or self.task_spec.task_config_name
        if task_name == "hunt_cow_do_not_touch_sheep":
            sheep_kills = event_delta(info, self.initial_info, "kill_entity", ["sheep"])
            metrics["sheep_kills"] = sheep_kills
            if sheep_kills > 0 and not self._already_failed:
                reward += self.failure_penalty
                done = True
                failure = True
                self._already_failed = True
                reason = "Constraint violation: sheep was killed."
                return RewardEvent(reward=reward, done=done, success=success, failure=failure, reason=reason, metrics=metrics)

        return RewardEvent(reward=reward, done=done, success=success, failure=failure, reason=reason, metrics=metrics)

    def success_result(self, info: Mapping) -> InteractionSuccessResult:
        return self.success_tracker.update(info)
