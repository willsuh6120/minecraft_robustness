<!-- filepath: /Users/muzhancun/workspace/MineStudio/docs/source/online/config.md -->
# Configuring Online Training Sessions

Effective online training in MineStudio hinges on a well-defined configuration. This document details the structure and parameters of the `online_dict` configuration object, which is essential for launching and managing your training runs. This configuration is passed alongside serializable `env_generator` and `policy_generator` functions to `online.rollout.start_manager.start_rolloutmanager` and `minestudio.online.trainer.start_trainer.start_trainer`, serving as the blueprint for your agent's learning journey.

The `online_dict` orchestrates various aspects of the training process, including:
- The type of trainer to use (e.g., PPO).
- How experience data is collected (rollout).
- How the agent's policy is optimized (training).
- How experiments are logged.

## Example Configuration: A Quick Look

Below is an illustrative example of an `online_dict` to give you a practical overview. Detailed explanations of each parameter follow in the subsequent sections.

```python
{
  "trainer_name": "PPOTrainer",
  "detach_rollout_manager": True,
  "rollout_config": {
      "num_rollout_workers": 2,
      "num_gpus_per_worker": 1.0,
      "num_cpus_per_worker": 1,
      "fragment_length": 256,
      "to_send_queue_size": 1, // Consider increasing if data transfer is a bottleneck
      "worker_config": {
          "num_envs": 2, // Number of parallel environments per rollout worker
          "batch_size": 1, // Inference batch size within each rollout worker
          "restart_interval": 3600,  # 1 hour, for worker stability
          "video_fps": 20,
          "video_output_dir": "output/videos",
      },
      "replay_buffer_config": {
          "max_chunks": 4800,
          "max_reuse": 2, // How many times a chunk can be sampled
          "max_staleness": 2, // Max model versions old a chunk can be
          "fragments_per_report": 40,
          "fragments_per_chunk": 1,
          "database_config": {
              "path": "output/replay_buffer_cache",
              "num_shards": 8,
          },
      },
      "episode_statistics_config": {}, // Configuration for episode statistics actor
  },
  "train_config": {
      "num_workers": 2, // Number of training workers (for distributed training)
      "num_gpus_per_worker": 1.0,
      "num_iterations": 4000,
      "vf_warmup": 0, // Iterations for value function warmup
      "learning_rate": 0.00002,
      "anneal_lr_linearly": False, // Linearly decay learning rate to 0
      "weight_decay": 0.04,
      "adam_eps": 1e-8,
      "batch_size_per_gpu": 1, // Training batch size on each GPU
      "batches_per_iteration": 10, // Number of batches processed per training iteration
      "gradient_accumulation": 10,  // Accumulate gradients over N batches
      "epochs_per_iteration": 1,  // Number of epochs over the data per iteration
      "context_length": 64, // Sequence length for policy input
      "discount": 0.999, // Discount factor (gamma)
      "gae_lambda": 0.95, // GAE lambda parameter
      "ppo_clip": 0.2, // PPO clipping parameter
      "clip_vloss": False,  // Whether to clip value loss
      "max_grad_norm": 5,  // Maximum gradient norm for clipping
      "zero_initial_vf": True, // Initialize value function output to zero
      "ppo_policy_coef": 1.0, // Coefficient for policy loss
      "ppo_vf_coef": 0.5,  // Coefficient for value function loss
      "kl_divergence_coef_rho": 0.0, // KL divergence coefficient
      "entropy_bonus_coef": 0.0, // Entropy bonus coefficient
      "coef_rho_decay": 0.9995, // Decay rate for KL divergence coefficient
      "log_ratio_range": 50,  // Range for log probability ratio (numerical stability)
      "normalize_advantage_full_batch": True,  // Normalize advantages over the full batch
      "use_normalized_vf": True, // Use normalized value function estimates
      "num_readers": 4, // Number of data reader threads for replay buffer
      "num_cpus_per_reader": 0.1, // CPUs allocated per reader
      "prefetch_batches": 2, // Batches to prefetch during training
      "save_interval": 10, // Iterations between model checkpoints
      "keep_interval": 40, // Iterations between permanent model saves (older ones might be deleted)
      "record_video_interval": 2, // Iterations between recording videos (if enabled)
      "enable_ref_update": False, // Enable reference model update (for certain algorithms)
      "resume": None, // Path to checkpoint for resuming training, or null/None
      "resume_optimizer": True, // Whether to resume optimizer state
      "save_path": "/scratch/hekaichen/workspace/MineStudio/minestudio/online/run/output" // Can be relative to Ray's log dir or absolute
  },
  "logger_config": {
      "project": "minestudio_online", // WandB project name
      "name": "bow_cow" // WandB run name
  },
}
```

------------

## Detailed Configuration Parameters

This section provides a comprehensive breakdown of all configurable parameters within the `online_dict`.

### Ⅰ. Top-Level Configuration

These parameters define the overall training setup.

#### `trainer_name`
- **Type**: `String`
- **Description**: Specifies the core training algorithm to be used. This name corresponds to a registered trainer class within the MineStudio framework.
- **Default**: `"PPOTrainer"`
- **Example**: `"PPOTrainer"`

#### `detach_rollout_manager`
- **Type**: `Boolean`
- **Description**: If `True`, the rollout manager will run in a detached mode, meaning its lifecycle is not strictly tied to the script that launched it. This can be useful for long-running experiments or when managing Ray actors independently.
- **Default**: `True`
- **Example**: `True`

---

### Ⅱ. Rollout Configuration (`rollout_config`)

This section governs how the agent interacts with the environment to collect experience data.

#### `num_rollout_workers`
- **Type**: `Integer`
- **Description**: The number of parallel rollout worker actors to create. Each worker manages its own set of environments and collects data independently.
- **Example**: `2`

#### `num_gpus_per_worker`
- **Type**: `Float`
- **Description**: The fraction of a GPU allocated to each rollout worker. Can be fractional if GPUs are shared. Set to `0` if no GPUs are needed for rollout (e.g., if the policy is lightweight or runs on CPU).
- **Example**: `1.0` (for one dedicated GPU per worker), `0.25` (if one GPU is shared by 4 workers)

#### `num_cpus_per_worker`
- **Type**: `Integer`
- **Description**: The number of CPU cores allocated to each rollout worker.
- **Example**: `1`

#### `fragment_length`
- **Type**: `Integer`
- **Description**: The number of environment steps to collect in each `SampleFragment` before sending it from the rollout worker. This is a fundamental unit of experience.
- **Example**: `256`

#### `to_send_queue_size`
- **Type**: `Integer`
- **Description**: The size of the queue within each `RolloutWorkerWrapper` that buffers `SampleFragment`s before they are pushed to the replay buffer interface. A larger queue can help smooth out data flow but consumes more memory.
- **Default**: `1`
- **Example**: `4`
- **Note**: If you observe that rollout workers are frequently blocked waiting to send data, consider increasing this value.

#### A. Rollout Worker Settings (`rollout_config.worker_config`)
Configuration specific to the behavior of individual rollout workers.

##### `num_envs`
- **Type**: `Integer`
- **Description**: The number of parallel environment instances each `RolloutWorker` will manage.
- **Example**: `16`

##### `batch_size`
- **Type**: `Integer`
- **Description**: The batch size for model inference within each `RolloutWorker`. Observations from `batch_size` environments are grouped together for a single forward pass of the policy model.
- **Example**: `2` (if `num_envs` is 16, this implies 8 inference calls per step across all envs)

##### `restart_interval`
- **Type**: `Integer` (seconds)
- **Description**: The interval, in seconds, after which a rollout worker might be automatically restarted. This can help mitigate issues from long-running processes, like memory leaks in the environment. Set to `0` or a very large number to disable.
- **Example**: `3600` (1 hour)

##### `video_fps`
- **Type**: `Integer`
- **Description**: Frames per second for videos recorded from the environments (if video recording is enabled, typically controlled by `train_config.record_video_interval`).
- **Example**: `20`

##### `video_output_dir`
- **Type**: `String`
- **Description**: Directory where recorded videos will be saved. This path is typically relative to Ray's run-specific log directory (e.g., `~/ray_results/<experiment_name>/<run_id>/`).
- **Example**: `"output/videos"`

#### B. Replay Buffer Settings (`rollout_config.replay_buffer_config`)
Parameters controlling the behavior of the replay buffer that stores and serves `SampleFragment`s.

##### `max_chunks`
- **Type**: `Integer`
- **Description**: The maximum number of data chunks the replay buffer can store. A "chunk" is a unit of storage in the buffer, often corresponding to one or more `SampleFragment`s.
- **Example**: `4800`

##### `max_reuse`
- **Type**: `Integer`
- **Description**: The maximum number of times a single data chunk can be sampled for training before it's potentially evicted or considered stale.
- **Default**: `2`
- **Example**: `2`

##### `max_staleness`
- **Type**: `Integer`
- **Description**: The maximum difference in model versions (trainer iterations) between the model that generated a `SampleFragment` and the current training model version for that fragment to still be considered fresh enough for sampling.
- **Default**: `2`
- **Example**: `2`

##### `fragments_per_report`
- **Type**: `Integer`
- **Description**: The number of `SampleFragment`s that should be processed or reported by the replay buffer system in certain logging or status update intervals.
- **Example**: `40`

##### `fragments_per_chunk`
- **Type**: `Integer`
- **Description**: The number of `SampleFragment`s that are grouped together to form a single "chunk" in the replay buffer's storage.
- **Default**: `1`
- **Example**: `1`

##### `database_config`
  - **Description**: Configuration for the underlying database used by the replay buffer (e.g., LMDB).
    - `path`: Path to the directory where the replay buffer database files will be stored. This is often relative to Ray's run-specific log directory.
      - **Type**: `String`
      - **Example**: `"output/replay_buffer_cache"`
    - `num_shards`: The number of database shards to use. Sharding can improve performance for large buffers.
      - **Type**: `Integer`
      - **Example**: `8`

#### C. Episode Statistics Settings (`rollout_config.episode_statistics_config`)
- **Description**: Configuration for the `EpisodeStatistics` actor, which tracks metrics like episode rewards and lengths.
- **Note**: This configuration section is often empty (`{}`) if default settings for the statistics actor are sufficient. Specific parameters would depend on the `EpisodeStatistics` implementation.

---

### Ⅲ. Training Configuration (`train_config`)

This section details parameters that control the model optimization process.

#### `num_workers`
- **Type**: `Integer`
- **Description**: The number of parallel training worker actors if using distributed training (e.g., Ray Train). For a single trainer instance, this is typically `0` or `1` depending on the framework.
- **Example**: `2` (for distributed training), `0` or `1` (for a single trainer)

#### `num_gpus_per_worker`
- **Type**: `Float`
- **Description**: The fraction of a GPU allocated to each training worker. For a single trainer, this is the GPU allocation for that trainer.
- **Example**: `1.0`

#### `num_iterations`
- **Type**: `Integer`
- **Description**: The total number of training iterations to perform. An iteration typically involves sampling data, processing it, and performing one or more gradient updates.
- **Example**: `4000`

#### `vf_warmup`
- **Type**: `Integer`
- **Description**: Number of initial training iterations during which only the value function (VF) might be trained, or trained with different parameters. This can help stabilize early training.
- **Default**: `0` (no warmup)
- **Example**: `100`

#### `learning_rate`
- **Type**: `Float`
- **Description**: The initial learning rate for the optimizer.
- **Example**: `0.00002` (or `2e-5`)

#### `anneal_lr_linearly`
- **Type**: `Boolean`
- **Description**: If `True`, the learning rate will be linearly decayed from its initial value to `0` over the course of `num_iterations`.
- **Default**: `False`
- **Example**: `True`

#### `weight_decay`
- **Type**: `Float`
- **Description**: The L2 regularization factor (weight decay) applied by the optimizer.
- **Example**: `0.04`

#### `adam_eps`
- **Type**: `Float`
- **Description**: The epsilon term for the Adam optimizer, added to the denominator for numerical stability.
- **Example**: `1e-8`

#### `batch_size_per_gpu`
- **Type**: `Integer`
- **Description**: The number of samples processed in a single batch on each GPU during training. The total effective batch size is `batch_size_per_gpu * num_gpus_per_worker * num_workers` (if distributed) or `batch_size_per_gpu * num_gpus_per_worker` (if single trainer).
- **Example**: `1`

#### `batches_per_iteration`
- **Type**: `Integer`
- **Description**: The number of training batches to process in each training iteration.
- **Example**: `10`

#### `gradient_accumulation`
- **Type**: `Integer`
- **Description**: The number of batches over which to accumulate gradients before performing an optimizer step. This allows for effectively larger batch sizes than what might fit in GPU memory at once. `effective_batch_size = batch_size_per_gpu * gradient_accumulation`.
- **Default**: `1` (no accumulation)
- **Example**: `10`

#### `epochs_per_iteration`
- **Type**: `Integer`
- **Description**: The number of times the trainer will iterate over the data collected for the current training iteration (or a portion of the replay buffer).
- **Default**: `1`
- **Example**: `1`

#### `context_length`
- **Type**: `Integer`
- **Description**: The sequence length or context window (number of time steps) provided as input to the policy model, especially relevant for recurrent policies or transformers.
- **Example**: `64`

#### `discount`
- **Type**: `Float`
- **Description**: The discount factor (gamma, γ) for future rewards in reinforcement learning. Values closer to `1` give more weight to future rewards.
- **Example**: `0.999`

#### `gae_lambda`
- **Type**: `Float`
- **Description**: The lambda (λ) parameter for Generalized Advantage Estimation (GAE). Controls the bias-variance trade-off for advantage estimates. `1.0` is high variance (Monte Carlo), `0.0` is high bias (TD(0)).
- **Example**: `0.95`

#### `ppo_clip`
- **Type**: `Float`
- **Description**: The clipping parameter (epsilon, ε) for Proximal Policy Optimization (PPO). Defines the range `[1-ε, 1+ε]` within which the probability ratio is clipped.
- **Example**: `0.2`

#### `clip_vloss`
- **Type**: `Boolean`
- **Description**: If `True`, the value function loss will also be clipped, similar to how the policy objective is clipped in PPO.
- **Default**: `False`
- **Example**: `True`
- **Note**: The exact mechanism for value loss clipping can vary; consult the specific PPO trainer implementation.

#### `max_grad_norm`
- **Type**: `Float`
- **Description**: The maximum L2 norm for gradients. Gradients will be clipped if their norm exceeds this value, preventing excessively large updates.
- **Example**: `5.0`

#### `zero_initial_vf`
- **Type**: `Boolean`
- **Description**: If `True`, the value function's output layer might be initialized in such a way that its initial predictions are close to zero. This can sometimes help with stability at the start of training.
- **Default**: `True`
- **Example**: `True`

#### `ppo_policy_coef`
- **Type**: `Float`
- **Description**: The coefficient (weight) for the PPO policy loss term in the total loss function.
- **Default**: `1.0`
- **Example**: `1.0`

#### `ppo_vf_coef`
- **Type**: `Float`
- **Description**: The coefficient (weight) for the value function loss term in the total loss function.
- **Default**: `0.5`
- **Example**: `0.5`

#### `kl_divergence_coef_rho`
- **Type**: `Float`
- **Description**: The coefficient (rho, ρ) for an optional KL divergence penalty term in the loss function. This is often used in PPO variants to adaptively control the step size or as an auxiliary loss.
- **Default**: `0.0` (disabled)
- **Example**: `0.01`

#### `entropy_bonus_coef`
- **Type**: `Float`
- **Description**: The coefficient for an entropy bonus term added to the loss. Encourages exploration by penalizing overly deterministic policies.
- **Default**: `0.0` (disabled)
- **Example**: `0.01`

#### `coef_rho_decay`
- **Type**: `Float`
- **Description**: If `kl_divergence_coef_rho` is adaptive or scheduled, this might be a decay factor applied to it over iterations.
- **Example**: `0.9995` (implies slow decay)

#### `log_ratio_range`
- **Type**: `Float`
- **Description**: A value used to clip the log of the probability ratio `log(π_new / π_old)` for numerical stability, preventing extreme values that could lead to NaNs or Infs.
- **Example**: `50`

#### `normalize_advantage_full_batch`
- **Type**: `Boolean`
- **Description**: If `True`, advantage estimates will be normalized (mean-centered and divided by standard deviation) across the entire batch of data collected for the current iteration.
- **Default**: `True`
- **Example**: `True`

#### `use_normalized_vf`
- **Type**: `Boolean`
- **Description**: If `True`, the value function targets or estimates might be normalized. This can interact with `normalize_advantage_full_batch`.
- **Default**: `True`
- **Example**: `True`

#### `num_readers`
- **Type**: `Integer`
- **Description**: The number of parallel data reader threads or actors used to fetch data from the replay buffer for training.
- **Example**: `4`

#### `num_cpus_per_reader`
- **Type**: `Float`
- **Description**: The number of CPU cores allocated to each data reader.
- **Example**: `0.1` (can be fractional if readers are lightweight threads)

#### `prefetch_batches`
- **Type**: `Integer`
- **Description**: The number of batches the data readers will prefetch and keep ready for the trainer. This helps hide data loading latency.
- **Example**: `2`

#### `save_interval`
- **Type**: `Integer`
- **Description**: The interval (in training iterations) at which to save a model checkpoint.
- **Example**: `10`

#### `keep_interval`
- **Type**: `Integer`
- **Description**: The interval (in training iterations) for saving "permanent" checkpoints. Checkpoints saved at other `save_interval` steps might be deleted over time to save space, but those at `keep_interval` steps are usually retained.
- **Example**: `40` (e.g., keep every 4th checkpoint if `save_interval` is 10)

#### `record_video_interval`
- **Type**: `Integer`
- **Description**: The interval (in training iterations) at which to trigger video recording from the rollout environments. Set to `0` or a very large number to disable frequent recording.
- **Example**: `2` (record videos every 2 iterations)

#### `enable_ref_update`
- **Type**: `Boolean`
- **Description**: If `True`, enables the update of a reference model. This is relevant for certain algorithms (e.g., some self-play or population-based methods) where a slower-moving or averaged version of the policy is maintained.
- **Default**: `False`
- **Example**: `False`

#### `resume`
- **Type**: `String` or `None`
- **Description**: Path to a previously saved model checkpoint from which to resume training. If `None` or an empty string, training starts from scratch.
- **Default**: `None`
- **Example**: `"/path/to/my_checkpoint_dir/checkpoint_000100"`

#### `resume_optimizer`
- **Type**: `Boolean`
- **Description**: If `True` and `resume` is specified, the state of the optimizer (e.g., learning rate, momentum) will also be restored from the checkpoint.
- **Default**: `True`
- **Example**: `True`

#### `save_path`
- **Type**: `String`
- **Description**: The base directory where model checkpoints and other training outputs will be saved.
    - If a relative path is given (e.g., `"output"`), it's typically interpreted as relative to Ray's default results directory for the current experiment (e.g., `~/ray_results/<experiment_name>/<run_id>/output`).
    - An absolute path can also be provided (e.g., `"/scratch/user/my_experiment_saves"`).
- **Example**: `"output"`, `"/scratch/hekaichen/workspace/MineStudio/minestudio/online/run/output"`

---

### Ⅳ. Logger Configuration (`logger_config`)

Parameters for configuring experiment logging, typically with Weights & Biases (`wandb`).

#### `project`
- **Type**: `String`
- **Description**: The name of the project in Weights & Biases where this run's logs will be stored.
- **Example**: `"minestudio_online"`

#### `name`
- **Type**: `String`
- **Description**: The specific name for this training run within the WandB project. This helps differentiate between various experiments or trials.
- **Example**: `"bow_cow"` (perhaps indicating agent type and environment)
- **Note**: It's good practice to make run names informative, possibly including key hyperparameters or timestamps.