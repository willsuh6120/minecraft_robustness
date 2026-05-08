.. _inference-steve:

Tutorial: Inference with STEVE-1
---------------------------------

To inference with STEVE-1, you first need to download pretrained checkpoints.
The example code is provided in ``minestudio/tutorials/inference/evaluate_steve/main.py``.

.. dropdown:: Evaluating STEVE-1

    .. code-block:: python

        from minestudio.simulator.callbacks import MinecraftCallback
        from minestudio.models import SteveOnePolicy
        from minestudio.simulator import MinecraftSim
        from minestudio.simulator.callbacks import SpeedTestCallback, load_callbacks_from_config
        from minestudio.inference import EpisodePipeline, MineGenerator, InfoBaseFilter
        from minestudio.benchmark import prepare_task_configs

        import ray
        from functools import partial
        from rich import print

        class CommandCallback(MinecraftCallback):
            """
            To use SteveOnePolicy, you need to contain a condition in the observation.
            """
            def __init__(self, command, cond_scale = 4.0):
                self.command = command
                self.cond_scale = cond_scale

            def after_reset(self, sim, obs, info):
                self.timestep = 0
                obs["condition"] = {
                    "cond_scale": self.cond_scale,
                    "text": self.command
                }
                return obs, info
            
            def after_step(self, sim, obs, reward, terminated, truncated, info):
                obs["condition"] = {
                    "cond_scale": self.cond_scale,
                    "text": self.command
                }
                return obs, reward, terminated, truncated, info


        if __name__ == '__main__':
            ray.init()
            task_configs = prepare_task_configs("simple", path="CraftJarvis/MineStudio_task_group.simple")
            config_file = task_configs["collect_wood"] 
            # you can try: survive_plant, collect_wood, build_pillar, ... ; make sure the config file contains `reference_video` field 
            print(config_file)

            env_generator = partial(
                MinecraftSim,
                obs_size = (224, 224),
                preferred_spawn_biome = "forest", 
                callbacks = [
                    SpeedTestCallback(50),
                    CommandCallback("mine log", cond_scale=4.0),  # Add a command callback for SteveOnePolicy
                ] + load_callbacks_from_config(config_file)
            )

            agent_generator = lambda: SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official")

            worker_kwargs = dict(
                env_generator=env_generator, 
                agent_generator=agent_generator,
                num_max_steps=600,
                num_episodes=1,
                tmpdir="./output",
                image_media="h264",
            )

            pipeline = EpisodePipeline(
                episode_generator=MineGenerator(
                    num_workers=1, 
                    num_gpus=0.25,
                    max_restarts=3,
                    **worker_kwargs, 
                ), 
                episode_filter=InfoBaseFilter(
                    key="mine_block",
                    regex=".*log.*",
                    num=1,
                ),
            )
            summary = pipeline.run()
            print(summary)

Since STEVE-1 is a text-conditioned policy, we need to provide textual commands to guide the agent's behavior.
Supported tasks and configs can be found in ``minestudio/benchmark/task_configs`` and a detailed explanation can be found in the benchmarking tutorial.

To pass text commands to STEVE-1, we implement a ``CommandCallback`` for the environment.
The ``CommandCallback`` adds a condition field to the observation that contains:
    - ``cond_scale``: A scaling factor for the conditioning (default: 4.0)
    - ``text``: The textual command describing the desired behavior

After the environment is initialized, the text command will be passed to the ``'condition'`` field of the observation and then be used to guide the agent's actions.
The command is applied to every observation throughout the episode, providing consistent guidance to the agent.

For the inference pipeline parameters, we need to specify:
    - task, configs and text command for the ``env_generator``.
    - pretrained checkpoint for the ``agent_generator``.
    - rollout steps, number of episodes, output path for ``worker_kwargs``.
    - number of gpus and workers for ``MineGenerator``.
    - An ``episode_filter`` to filter the episode based on the key and value of the observation.

In the above example, we test the STEVE-1 model on the task of collecting wood with the command "mine log" and 1 episode with 600 steps.
1 worker is used with 0.25 GPU per worker.
The episode will be filtered based on the key ``mine_block`` and regex pattern ``.*log.*``.

For common text commands for different tasks, you should refer to the original STEVE-1 paper [1]_. 

The conditioning scale (``cond_scale``) controls how strongly the text command influences the agent's behavior:
    - Higher values (e.g., 6.0-8.0) make the agent follow commands more strictly
    - Lower values (e.g., 2.0-4.0) allow more exploration while still following the general command
    - The default value of 4.0 provides a good balance for most tasks

The summary of the pipeline will be printed to the console, showing the success rate and the number of episodes.
After the pipeline is finished, the console will print the summary of the pipeline like the following:

.. code-block:: python

    ...    

    (Worker pid=922019) Episode 0 saved at output/episode_0.mp4
    (Worker pid=922019) Speed Test Status:
    (Worker pid=922019) Average Time: 0.04
    (Worker pid=922019) Average FPS: 24.28
    (Worker pid=922019) Total Steps: 600
    {'num_yes': 1, 'num_episodes': 1, 'yes_rate': '100.00%'}

.. [1] Lifshitz S, Paster K, Chan H, et al. Steve-1: A generative model for text-to-behavior in minecraft[J]. Advances in Neural Information Processing Systems, 2024, 36.