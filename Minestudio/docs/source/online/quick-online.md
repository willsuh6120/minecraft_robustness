<!-- # Quick Online Training with VPT -->

We provide a simple example in `online/run`. You can fine-tune VPT to complete the task of killing cows by directly running:

```bash
cd online/run
bash run.sh
```

Specifically, this process includes several important configurations:

**Policy Generator**

which does not accept parameters and directly returns `MinePolicy`.As an example：
```python
def policy_generator():
    from minestudio.models.openai_vpt.body import load_openai_policy
    model_path = 'pretrained/foundation-model-2x.model'
    weights_path = 'pretrained/bc-from-early-game-2x.weights'
    policy = load_openai_policy(model_path, weights_path)
    return policy
```

**Environment Generator**

which does not accept parameters and directly returns `MinePolicy`. As an example:

```python
def env_generator():
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        SummonMobsCallback, 
        MaskActionsCallback, 
        RewardsCallback, 
        CommandsCallback, 
        JudgeResetCallback,
        FastResetCallback
    )
    sim = MinecraftSim(
        obs_size=(128, 128), 
        preferred_spawn_biome="plains", 
        action_type = "agent",
        timestep_limit=1000,
        callbacks=[
            SummonMobsCallback([{'name': 'cow', 'number': 10, 'range_x': [-5, 5], 'range_z': [-5, 5]}]),
            MaskActionsCallback(inventory=0), 
            RewardsCallback([{
                'event': 'kill_entity', 
                'objects': ['cow'], 
                'reward': 1.0, 
                'identity': 'chop_tree', 
                'max_reward_times': 30, 
            }]),
            CommandsCallback(commands=[
                '/give @p minecraft:iron_sword 1',
                '/give @p minecraft:diamond 64',
                '/effect @p 5 9999 255 true',
            ]),
            FastResetCallback(
                biomes=['plains'],
                random_tp_range=1000,
            ),
            JudgeResetCallback(600),
        ]
    )
    return sim
```

**Config** which provide the hyper-parameters for online training:
```python
online_dict = {
 "trainer_name": "PPOTrainer",
    "detach_rollout_manager": True,
    "rollout_config": {
        "num_rollout_workers": 2,
        "num_gpus_per_worker": 1.0,
        "num_cpus_per_worker": 1,
        "fragment_length": 256,
        "to_send_queue_size": 6,
        "worker_config": {
            "num_envs": 12,
            "batch_size": 6,
            "restart_interval": 3600,  # 1h
            "video_fps": 20,
            "video_output_dir": "output/videos",
        },
        "replay_buffer_config": {
            "max_chunks": 4800,
            "max_reuse": 2,
            "max_staleness": 2,
            "fragments_per_report": 40,
            "fragments_per_chunk": 1,
            "database_config": {
                "path": "output/replay_buffer_cache",
                "num_shards": 8,
            },
        },
        "episode_statistics_config": {},
    },
    "train_config": {
        "num_workers": 2,
        "num_gpus_per_worker": 1.0,
        "num_iterations": 4000,
        "vf_warmup": 0,
        "learning_rate": 0.00002,
        "anneal_lr_linearly": 
        "weight_decay": 0.04,
        "adam_eps": 1e-8,
        "batch_size_per_gpu": 1,
        "batches_per_iteration": 200,
        "gradient_accumulation": 10, 
        "epochs_per_iteration": 1, 
        "context_length": 64,
        "discount": 0.999,
        "gae_lambda": 0.95,
        "ppo_clip": 0.2,
        "clip_vloss": False, 
        "max_grad_norm": 5, 
        "zero_initial_vf": True,
        "ppo_policy_coef": 1.0,
        "ppo_vf_coef": 0.5, 
        "kl_divergence_coef_rho": 0.2,
        "entropy_bonus_coef": 0.0,
        "coef_rho_decay": 0.9995,
        "log_ratio_range": 50,  
        "normalize_advantage_full_batch": True, 
        "use_normalized_vf": True,
        "num_readers": 4,
        "num_cpus_per_reader": 0.1,
        "prefetch_batches": 2,
        "save_interval": 10,
        "keep_interval": 40,
        "record_video_interval": 2,
        "fix_decoder": False,
        "resume": None, 
        "resume_optimizer": True,
        "save_path": "output"
    },

    "logger_config": {
        "project": "minestudio_online",
        "name": "bow_cow"
    },
}

online_config = OmegaConf.create(online_dict)
```

After preparing all the above content, run:

```
from minestudio.online.rollout.start_manager import start_rolloutmanager
from minestudio.online.trainer.start_trainer import start_trainer
start_rolloutmanager(policy_generator, env_generator, online_cfg)
start_trainer(policy_generator, env_generator, online_cfg)
```

to start online training.

We use `wandb` to log and monitor the progress of the run. The corresponding parameters passed to `wandb` are in `config.logger_config`. When `save_path` is `None`, the checkpoint will be saved to Ray's working directory at `~/ray_results`.