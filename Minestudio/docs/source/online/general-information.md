<!-- filepath: /Users/muzhancun/workspace/MineStudio/docs/source/online/general-information.md -->
# Online Training Module: General Information

Welcome to the MineStudio Online Training Module! This section of the documentation provides a high-level overview of its architecture, core components, and the underlying design philosophy. The online module is engineered to facilitate the training of agents directly within the interactive Minecraft environment, allowing them to learn and adapt through continuous experience.

## Core Philosophy: Learning by Doing, at Scale

The online training pipeline in MineStudio is built with scalability and efficiency in mind. It leverages the power of [Ray](https://www.ray.io/) for distributed computation, enabling you to train agents on complex tasks that may require significant computational resources and vast amounts of interaction data. The central idea is to have agents (policies) that learn by actively engaging with the environment, collecting experiences, and updating their decision-making processes in near real-time.

## Architectural Overview: Key Components

The online training module is primarily organized into three interconnected sub-modules, each residing in its respective subfolder within `minestudio/online/`:

1.  **`run`**: This is the entry point for initiating and managing an online training session. It's responsible for parsing configurations, initializing the necessary Ray actors, and orchestrating the overall workflow. Think of it as the conductor of the online training orchestra.
    *   *For more details, see the [Run](./run.md) documentation.*

2.  **`rollout`**: This component is dedicated to the crucial task of experience collection. It manages a fleet of workers that interact with multiple instances of the Minecraft environment in parallel. These workers use the current agent policy to decide actions, observe outcomes, and gather the raw data (observations, actions, rewards, etc.) that forms the basis of learning.
    *   *For more details, see the [Rollout](./rollout.md) documentation.*

3.  **`trainer`**: This is where the learning happens. The trainer takes the experiences collected by the `rollout` workers and uses them to optimize the agent's policy. MineStudio primarily features a `PPOTrainer` (Proximal Policy Optimization), a robust and widely-used reinforcement learning algorithm.
    *   *For more details, see the [Trainer](./trainer.md) documentation.*

4.  **`utils`**: This directory houses a collection of shared utilities, data structures, and helper functions that support both the `rollout` and `trainer` components. This promotes code reusability and consistency.
    *   *For more details, see the [Utils](./utils.md) documentation.*

## Interplay of Components: A Simplified Data Flow

While the detailed interactions are covered in the specific documentation for each component, here's a simplified view of how they work together:

1.  The **`run`** script starts the process, initializing the **`RolloutManager`** (from the `rollout` module) and the **`Trainer`** (e.g., `PPOTrainer`).
2.  The **`RolloutManager`** deploys multiple **`RolloutWorker`** actors. Each `RolloutWorker` in turn manages several **`EnvWorker`** instances, which are the actual Minecraft environment simulations.
3.  `EnvWorker`s send observations to their `RolloutWorker`.
4.  The `RolloutWorker` uses its local copy of the current policy (periodically updated by the `Trainer`) to select actions for each of its `EnvWorker`s.
5.  Actions are applied in the `EnvWorker`s, and the resulting new observations, rewards, and done states (collectively, a "step" of experience) are sent back to the `RolloutWorker`.
6.  The `RolloutWorker` groups these steps into `SampleFragment`s (chunks of trajectory data).
7.  These `SampleFragment`s are then sent, often via a `RolloutWorkerWrapper` and an internal queue, to a **Replay Buffer** (which can be part of the `RolloutManager` or a separate entity it manages).
8.  The **`Trainer`** fetches batches of `SampleFragment`s from the Replay Buffer.
9.  The `Trainer` computes advantages (e.g., using GAE) and then performs optimization steps (e.g., PPO updates) to improve the policy and value function models.
10. Periodically, the `Trainer` sends the updated model weights to the `RolloutManager`, which then broadcasts them to all `RolloutWorker`s, ensuring they use the latest policy for subsequent data collection.
11. This cycle of data collection and training continues, allowing the agent to progressively learn and improve its performance.

## Getting Started

To dive deeper into specific aspects:

*   Understand how to configure your training runs in the [Config](./config.md) section.
*   For a quick guide on launching a training session, refer to the [Quick Start](./quick-online.md).
*   If you're interested in extending or modifying the existing trainers or policies, the [Customization](./customization.md) page will be your guide.

This modular and distributed architecture is designed to be flexible and scalable, catering to a wide range of research and development needs in the exciting domain of learning agents for Minecraft.

