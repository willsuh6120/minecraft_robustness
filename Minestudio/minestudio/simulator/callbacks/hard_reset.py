'''
Date: 2024-11-11 16:15:32
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-16 23:45:32
FilePath: /MineStudio/minestudio/simulator/callbacks/hard_reset.py
'''
import random
import numpy as np
from typing import Any, Dict, List, Optional
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.utils.register import Registers


@Registers.simulator_callback.register
class HardResetCallback(MinecraftCallback):
    """Performs a hard reset of the Minecraft environment.

    This callback forces a full environment reset by setting a specific seed
    and teleporting the player to a predefined spawn position. It is used
    when a complete and predictable reset is required.

    :param spawn_positions: A list of dictionaries, each specifying a "seed" (int)
                            and a "position" ([x, z, y] list) for spawning.
    :type spawn_positions: List[dict]
    """

    def create_from_conf(source):
        """Creates a HardResetCallback instance from a configuration source.

        Loads data from the given configuration (file path or dict) and
        initializes a HardResetCallback if 'spawn_positions' is present.

        :param source: The configuration source.
        :type source: any
        :returns: A HardResetCallback instance or None.
        :rtype: Optional[HardResetCallback]
        """
        data = MinecraftCallback.load_data_from_conf(source)
        if 'spawn_positions' in data:
            raw_commands = data.get("custom_init_commands") or data.get("commands") or []
            has_command_authored_scene = bool(isinstance(raw_commands, list) and len(raw_commands) > 0)
            procedural_layout = data.get("procedural_layout")
            spawn_support_pad = data.get("spawn_support_pad")
            ensure_spawn_support = bool(has_command_authored_scene or isinstance(procedural_layout, dict))
            if isinstance(spawn_support_pad, dict) and spawn_support_pad.get("enabled") is not None:
                ensure_spawn_support = bool(spawn_support_pad.get("enabled"))
            return HardResetCallback(
                data['spawn_positions'],
                ensure_spawn_support=ensure_spawn_support,
                spawn_support_pad=spawn_support_pad if isinstance(spawn_support_pad, dict) else None,
            )
        else:
            return None

    def __init__(
        self,
        spawn_positions: List,
        *,
        ensure_spawn_support: bool = False,
        spawn_support_pad: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the HardResetCallback.

        :param spawn_positions: A list of potential spawn configurations.
                                Each configuration is a dict with "seed" and "position".
                                e.g., [{"seed": 123, "position": [0, 64, 0]}]
        :type spawn_positions: List[dict]
        """
        super().__init__()
        """
        position is a list of {
            "seed": int,
            "position": [x, z, y], 
        }
        """
        self.spawn_positions = spawn_positions
        pad_conf = dict(spawn_support_pad or {})
        if pad_conf.get("enabled") is not None:
            ensure_spawn_support = bool(pad_conf.pop("enabled"))
        self.ensure_spawn_support = bool(ensure_spawn_support)
        self.spawn_support_block = str(pad_conf.get("block_type", "minecraft:barrier") or "minecraft:barrier")
        self.spawn_support_cleanup_after_build = bool(pad_conf.get("cleanup_after_build", False))
        try:
            self.spawn_support_radius = max(0, int(pad_conf.get("radius", 1)))
        except Exception:
            self.spawn_support_radius = 1
        try:
            self.spawn_head_clearance = max(1, int(pad_conf.get("clearance_height", 2)))
        except Exception:
            self.spawn_head_clearance = 2

    def _spawn_pose(self):
        x, z, y = self.position["position"]
        yaw = self.position.get("yaw", None)
        pitch = self.position.get("pitch", None)
        yaw = None if yaw is None else float(yaw)
        pitch = None if pitch is None else float(pitch)
        return float(x), float(z), float(y), yaw, pitch

    def _teleport_player(self, sim):
        x, z, y, yaw, pitch = self._spawn_pose()
        if yaw is None and pitch is None:
            return sim.env.execute_cmd(f"/tp @a {x} {z} {y}")
        yaw = 0.0 if yaw is None else float(yaw)
        pitch = 0.0 if pitch is None else float(pitch)
        return sim.env.execute_cmd(f"/tp @a {x} {z} {y} {yaw} {pitch}")

    def _apply_spawn_support_pad(self, sim):
        if not self.ensure_spawn_support:
            return
        x, z, y, _, _ = self._spawn_pose()
        pad_y = int(np.floor(float(z))) - 1
        center_x = int(np.floor(float(x)))
        center_z = int(np.floor(float(y)))
        radius = int(self.spawn_support_radius)
        x0 = center_x - radius
        x1 = center_x + radius
        z0 = center_z - radius
        z1 = center_z + radius
        sim.env.execute_cmd(
            f"/fill {x0} {pad_y} {z0} {x1} {pad_y} {z1} {self.spawn_support_block} keep"
        )
        clearance_top = pad_y + int(self.spawn_head_clearance) + 1
        sim.env.execute_cmd(
            f"/fill {center_x} {pad_y + 1} {center_z} {center_x} {clearance_top} {center_z} minecraft:air"
        )

    def _cleanup_spawn_support_pad(self, sim):
        if not self.ensure_spawn_support or not self.spawn_support_cleanup_after_build:
            return
        x, z, y, _, _ = self._spawn_pose()
        pad_y = int(np.floor(float(z))) - 1
        center_x = int(np.floor(float(x)))
        center_z = int(np.floor(float(y)))
        radius = int(self.spawn_support_radius)
        x0 = center_x - radius
        x1 = center_x + radius
        z0 = center_z - radius
        z1 = center_z + radius
        sim.env.execute_cmd(
            f"/fill {x0} {pad_y} {z0} {x1} {pad_y} {z1} minecraft:air replace {self.spawn_support_block}"
        )

    def before_reset(self, sim, reset_flag):
        """Selects a spawn position and sets the environment seed before reset.

        Randomly chooses one of the provided `spawn_positions`, sets the
        environment's seed to the chosen seed, and forces a reset.

        :param sim: The simulator instance.
        :param reset_flag: The current reset flag status.
        :returns: True, to indicate that a reset should occur.
        :rtype: bool
        """
        self.position = random.choice(self.spawn_positions)
        sim.env.seed(self.position['seed'])
        return True

    def after_reset(self, sim, obs, info):
        """Teleports the player and allows the environment to settle after reset.

        After the environment resets, this method teleports the player to the
        selected x, z, y coordinates and then executes a number of no-op actions
        to allow the game world to stabilize.

        :param sim: The simulator instance.
        :param obs: The initial observation after reset.
        :param info: The initial info dictionary after reset.
        :returns: The modified observation and info.
        :rtype: tuple[dict, dict]
        """
        self._apply_spawn_support_pad(sim)
        obs, _, done, info = self._teleport_player(sim)
        # Build any command-authored scene before the settle no-ops. Otherwise the
        # player can fall out of the intended spawn area during the warmup steps.
        try:
            from minestudio.simulator.callbacks.commands import CommandsCallback

            for callback in getattr(sim, "callbacks", []):
                if not isinstance(callback, CommandsCallback):
                    continue
                if not getattr(callback, "commands", None):
                    continue
                for command in callback.commands:
                    obs, _, done, info = sim.env.execute_cmd(command)
                callback._skip_once = True
        except Exception:
            pass
        if self.spawn_support_cleanup_after_build:
            self._cleanup_spawn_support_pad(sim)
        else:
            self._apply_spawn_support_pad(sim)
        obs, _, done, info = self._teleport_player(sim)
        for _ in range(50): 
            action = sim.env.action_space.no_op()
            obs, reward, done, info = sim.env.step(action)
        if self.spawn_support_cleanup_after_build:
            self._cleanup_spawn_support_pad(sim)
        else:
            self._apply_spawn_support_pad(sim)
        obs, _, done, info = self._teleport_player(sim)
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info
