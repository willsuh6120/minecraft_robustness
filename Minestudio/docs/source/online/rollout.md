# Rollout

This document delves into the rollout process within the online training module, a critical phase where the agent interacts with the environment to gather experiences.

## Design Principle

The rollout module is engineered for efficient, large-scale experience collection, leveraging the power of `Ray` for distributed execution. Its design revolves around a few key components and philosophies.

At the most granular level, the **`EnvWorker`** (Environment Worker) is responsible for running an individual instance of the Minecraft simulation. Multiple `EnvWorker`s operate in parallel, each generating a stream of raw interactions.

Orchestrating these `EnvWorker`s is the **`RolloutWorker`**. This component manages a collection of environments and houses the current policy model. Its primary duties include spawning and overseeing its `EnvWorker`s, performing inference (i.e., using the policy model to decide actions based on observations from `EnvWorker`s), stepping the environments forward by relaying these actions, and collecting the resultant data (observations, actions, rewards, and done states). The `RolloutWorker` also communicates with an `EpisodeStatistics` actor to report metrics. For flexibility, a `progress_handler` can be specified; this callback is invoked with step-wise data and is typically used to channel the collected experiences towards a replay buffer.

To manage the lifecycle and distributed nature of `RolloutWorker`s, they are wrapped by **`RolloutWorkerWrapper`** Ray actors. Each `RolloutWorkerWrapper` is responsible for a single `RolloutWorker`. It handles model updates pushed from the trainer and manages communication with the replay buffer. A key function of the wrapper is to buffer the steps collected by its `RolloutWorker` and assemble them into `SampleFragment`s – coherent chunks of trajectory data.

Overseeing the entire distributed fleet of `RolloutWorkerWrapper`s is the **`RolloutManager`**, itself a Ray actor. This central coordinator manages the creation and supervision of all `RolloutWorkerWrapper`s. It plays a crucial role in **Model Synchronization**: when the trainer produces an updated model, it's sent to the `RolloutManager`, which then efficiently distributes these new model weights to all its `RolloutWorkerWrapper`s. These wrappers, in turn, update their local `RolloutWorker` instances. The system also manages data staleness, ensuring that workers don't continue to generate data with overly outdated policies. The `RolloutManager` also interfaces with the `ReplayBuffer` (often via a `ReplayBufferInterface`) and the `EpisodeStatistics` actor.

The **Data Flow** is designed for parallelism and efficiency: `EnvWorker`s run environment instances and send observations to their `RolloutWorker`. The `RolloutWorker` performs inference, sends actions back, and collects step data. This data is passed to its `RolloutWorkerWrapper` via the `progress_handler`. The `RolloutWorkerWrapper` buffers these steps, assembles them into `SampleFragment`s, and then typically uses a separate internal queue and sender thread to push these fragments to the `ReplayBufferInterface`, ready for the trainer.

## Logic

The rollout process unfolds through a sequence of coordinated actions, primarily orchestrated by Ray actors:

### 1. Initialization: Setting the Stage

The process begins with the creation of the **`RolloutManager`** Ray actor. This manager immediately takes on the task of spawning multiple **`RolloutWorkerWrapper`** Ray actors. Each of these wrappers, upon its own initialization, creates and configures a local **`RolloutWorker`** instance. Finally, each `RolloutWorker` launches several **`EnvWorker`** processes. Each `EnvWorker` is an independent simulation of the Minecraft environment, ready to interact with an agent.

```{note}
This hierarchical setup (`RolloutManager` -> `RolloutWorkerWrapper` (actor) -> `RolloutWorker` -> `EnvWorker` (process)) allows for scalable and resilient distributed data collection. Ray's actor model handles the complexities of inter-process communication and scheduling across potentially many machines.
```

### 2. The Experience Collection Loop: Agents in Action

The core of the rollout happens within each `RolloutWorker`. This is a continuous loop:
The `RolloutWorker` actively **polls** its associated `EnvWorker`s for new observations from the game. Once enough observations are gathered to form an inference batch (the size of which is configurable), the `RolloutWorker` uses its current policy model to **perform inference**, generating actions for each environment. These actions are then dispatched back to the respective `EnvWorker`s, which apply them to their Minecraft instances and step the simulation forward.

The resulting data from each step – the new observation, the action taken, the reward received, and whether the episode terminated or was truncated – is then passed by the `RolloutWorker` to its `progress_handler`. As mentioned, this handler is typically a method within the parent `RolloutWorkerWrapper`.

### 3. Data Buffering and Fragment Creation: Packaging Experiences

Inside the `RolloutWorkerWrapper`, the `progress_handler` (called by the `RolloutWorker`) appends the incoming step data to an internal buffer, usually specific to each `EnvWorker` it manages. When the buffer for a particular environment accumulates enough steps (typically `fragment_length + 1`, where `fragment_length` is the desired size of a data chunk), a `SampleFragment` is created. This fragment encapsulates the first `fragment_length` steps, forming a coherent piece of trajectory.
This newly minted `SampleFragment` is then placed into an internal **queue** within the `RolloutWorkerWrapper`.

```{tip}
Using an internal queue and a separate sender thread within the `RolloutWorkerWrapper` decouples the data collection (which is often synchronous with the environment steps) from the data transmission to the replay buffer. This can improve throughput and prevent the environment simulation from stalling if the replay buffer is temporarily slow to ingest data.
```
A dedicated **sender thread** running within the `RolloutWorkerWrapper` continuously monitors this queue. It takes `SampleFragment`s from the queue and adds them to the `ReplayBufferInterface`, making them available for the training process.

### 4. Model Updates: Staying Current

To ensure the agent learns from the most effective behaviors, its policy model must be kept up-to-date. The `Trainer` component, after performing its optimization steps, periodically sends new model weights to the `RolloutManager`.
The `RolloutManager` then efficiently **broadcasts** these updated weights to all its managed `RolloutWorkerWrapper` actors. Upon receiving new weights, each `RolloutWorkerWrapper` instructs its local `RolloutWorker` to load them into its policy model.

A `max_staleness` parameter is often used to manage data quality. If a `RolloutWorkerWrapper` finds that the data it's collecting is based on a model version that is too old compared to the latest version received, it might discard that data or reset its buffers to avoid polluting the replay buffer with experiences from a significantly outdated policy.

### 5. Monitoring Performance and Progress: Statistics

Keeping track of the rollout process is vital. Two main components handle this:
The **`EpisodeStatistics`** actor, a central Ray actor, is responsible for tracking and logging episode-level metrics. As `EnvWorker`s complete episodes (via their `RolloutWorker`), they report information like total rewards and episode lengths to `EpisodeStatistics`. This provides a high-level view of agent performance.

At a more granular level, the **`PipelineMonitor`** (often residing within each `RolloutWorker`) tracks performance metrics of the data collection pipeline itself. This can include timings for different stages, such as the time spent receiving observations from environments, the duration of the inference step, and the time taken to send actions back. This information is invaluable for identifying bottlenecks and optimizing the rollout throughput.

In essence, the `RolloutManager` acts as the conductor of this distributed orchestra of workers and actors. It ensures that all components are working in concert, using up-to-date policies, and that the valuable experience data generated by the agent's interactions is efficiently collected, packaged, and stored for learning.