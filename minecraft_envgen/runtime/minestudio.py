from __future__ import annotations

from functools import partial
from typing import Any, Iterable, List, Mapping

from minecraft_envgen.core.contracts import CompiledEnvironment, EngineTarget


def _coerce_callback_spec(raw: Any) -> Any:
    if hasattr(raw, "before_step") or hasattr(raw, "after_reset"):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError(f"Unsupported callback spec type: {type(raw)!r}")
    if "name" not in raw or "config" not in raw:
        raise ValueError("Callback specs must contain 'name' and 'config'.")
    return _build_callback(str(raw["name"]), dict(raw.get("config", {})))


def _build_callback(name: str, config: Mapping[str, Any]) -> Any:
    from minestudio.simulator.callbacks import (
        CommandsCallback,
        FastResetCallback,
        HardResetCallback,
        MaskActionsCallback,
        RecordCallback,
        SpeedTestCallback,
        SummonMobsCallback,
        VoxelsCallback,
    )
    from minestudio.simulator.callbacks.rewards import RewardsCallback

    if name == "CommandsCallback":
        return CommandsCallback(commands=list(config.get("commands", [])))
    if name == "SummonMobsCallback":
        return SummonMobsCallback(list(config.get("mobs", [])))
    if name == "FastResetCallback":
        return FastResetCallback(
            biomes=list(config["biomes"]),
            random_tp_range=int(config["random_tp_range"]),
            start_time=int(config.get("start_time", 0)),
            start_weather=str(config.get("start_weather", "clear")),
        )
    if name == "HardResetCallback":
        return HardResetCallback(list(config["spawn_positions"]))
    if name == "RewardsCallback":
        return RewardsCallback(list(config.get("rewards", [])))
    if name == "MaskActionsCallback":
        return MaskActionsCallback(**dict(config))
    if name == "VoxelsCallback":
        return VoxelsCallback(list(config.get("voxels_ins", [-7, 7, -7, 7, -7, 7])))
    if name == "RecordCallback":
        return RecordCallback(**dict(config))
    if name == "SpeedTestCallback":
        return SpeedTestCallback(int(config.get("interval", 50)))
    raise ValueError(f"Unsupported MineStudio callback '{name}'.")


def build_minestudio_callbacks(
    callback_specs: Iterable[Any],
    extra_callbacks: Iterable[Any] | None = None,
) -> List[Any]:
    callbacks = [_coerce_callback_spec(spec) for spec in callback_specs]
    if extra_callbacks:
        callbacks.extend(_coerce_callback_spec(spec) for spec in extra_callbacks)
    return callbacks


def build_minecraft_sim(
    compiled: CompiledEnvironment,
    extra_callbacks: Iterable[Any] | None = None,
    **sim_overrides: Any,
) -> Any:
    if compiled.engine != EngineTarget.MINESTUDIO:
        raise ValueError(f"Expected a MineStudio compiled environment, got '{compiled.engine.value}'.")

    from minestudio.simulator import MinecraftSim

    runtime = dict(compiled.runtime["sim_kwargs"])
    runtime.update(sim_overrides)
    callbacks = build_minestudio_callbacks(compiled.runtime.get("callbacks", []), extra_callbacks)
    return MinecraftSim(callbacks=callbacks, **runtime)


def make_env_generator(
    compiled: CompiledEnvironment,
    extra_callbacks: Iterable[Any] | None = None,
    **sim_overrides: Any,
) -> Any:
    if compiled.engine != EngineTarget.MINESTUDIO:
        raise ValueError(f"Expected a MineStudio compiled environment, got '{compiled.engine.value}'.")

    from minestudio.simulator import MinecraftSim

    runtime = dict(compiled.runtime["sim_kwargs"])
    runtime.update(sim_overrides)
    callbacks = build_minestudio_callbacks(compiled.runtime.get("callbacks", []), extra_callbacks)
    return partial(MinecraftSim, callbacks=callbacks, **runtime)


def make_default_env_generator(
    *,
    action_type: str = "agent",
    obs_size: Iterable[int] = (128, 128),
    extra_callbacks: Iterable[Any] | None = None,
    **sim_overrides: Any,
) -> Any:
    from minestudio.simulator import MinecraftSim

    runtime = dict(sim_overrides)
    callbacks = build_minestudio_callbacks([], extra_callbacks)
    return partial(MinecraftSim, callbacks=callbacks, action_type=action_type, obs_size=list(obs_size), **runtime)


def make_vpt_agent_generator(model_uri: str) -> Any:
    from minestudio.models import VPTPolicy

    return lambda: VPTPolicy.from_pretrained(model_uri)
