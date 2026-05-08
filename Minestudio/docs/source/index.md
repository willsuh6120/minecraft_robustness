---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for MineStudio, with links to the rest
      of the site.
html_theme.sidebar_secondary.remove: true
---

<div align="center">
<img src="./_static/banner.png" width="60%" alt="MineStudio" />
</div>

<div align="center">
	<a href="https://craftjarvis.github.io/"><img alt="Homepage" src="https://img.shields.io/badge/%20CraftJarvis-HomePage-ffc107?color=blue&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
	<a href="https://huggingface.co/CraftJarvis"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CraftJarvis-ffc107?color=3b65ab&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
	<a href="https://github.com/CraftJarvis/MineStudio/blob/master/LICENSE"><img src="https://img.shields.io/badge/Code%20License-MIT-blue"/></a>
</div>
<div align="center">	
	<a href="https://arxiv.org/abs/2412.18293"><img src="https://img.shields.io/badge/arXiv-2412.18293-b31b1b.svg"></a>
	<a href="https://craftjarvis.github.io/MineStudio/"><img src="https://img.shields.io/badge/Doc-Sphinx-yellow"/></a>
    	<a href="https://pypi.org/project/minestudio/"><img src="https://img.shields.io/pypi/v/minestudio.svg"/></a>
	<a href="https://huggingface.co/CraftJarvis"><img src="https://img.shields.io/badge/Dataset-Released-orange"/></a>
	<a href="https://github.com/CraftJarvis/MineStudio/tree/master/minestudio/tutorials"><img alt="Static Badge" src="https://img.shields.io/badge/Tutorials-easy-brightgreen"></a>
	<a href="https://github.com/CraftJarvis/MineStudio"><img src="https://visitor-badge.laobi.icu/badge?page_id=CraftJarvis.MineStudio"/></a>
	<a href="https://github.com/CraftJarvis/MineStudio"><img src="https://img.shields.io/github/stars/CraftJarvis/MineStudio"/></a>
</div>

<hr>

# Welcome to MineStudio!

MineStudio contains a series of tools and APIs that can help you quickly develop Minecraft AI agents.

```{gallery-grid}
:grid-columns: 1 2 2 4
:center-rows:

- header: "{fas}`cube;pst-color-primary` Simulator"
  content: "Easily customizable Minecraft simulator based on [MineRL](https://github.com/minerllabs/minerl)."
  link: "simulator/index.html"
- header: "{fas}`database;pst-color-primary` Data"
  content: "A trajectory data structure for efficiently storing and retrieving arbitray trajectory segment."
  link: "data/index.html"
- header: "{fas}`brain;pst-color-primary` Models"
  content: "A template for Minecraft policy model and a gallery of baseline models."
  link: "models/index.html"
- header: "{fas}`cogs;pst-color-primary` Offline Training"
  content: "A straightforward pipeline for pre-training Minecraft agents with offline data."
  link: "offline/index.html"
- header: "{fas}`gamepad;pst-color-primary` Online Training"
  content: "Efficient RL implementation supporting memory-based policies and simulator crash recovery."
  link: "online/index.html"
- header: "{fas}`rocket;pst-color-primary` Inference"
  content: "Pallarelized and distributed inference framework based on [Ray](https://docs.ray.io/en/latest/index.html)."
  link: "inference/index.html"
- header: "{fas}`tasks;pst-color-primary` Benchmark"
  content: "Automating and batch-testing of diverse Minecraft task with [MCU](https://craftjarvis.github.io/MCU/)"
  link: "benchmark/index.html"
- header: "{fas}`code;pst-color-primary` API"
  content: "A comprehensive API reference for MineStudio, including all modules and classes."
  link: "api/index.html"
```

```{toctree}
:maxdepth: 2
:caption: Contents:
:hidden:

Overview <overview/index>
Getting Started <overview/getting-started>
Simulator <simulator/index>
Data <data/index>
Models <models/index>
Offline <offline/index>
Online <online/index>
Inference <inference/index>
Benchmark <benchmark/index>
API <api/index>
```

**This repository is under development.** We welcome any contributions and suggestions.

## News

- 2025/05/28 - We have released a big update of MineStudio (v1.1.4) with the following changes:
  - Refactored the [data](https://craftjarvis.github.io/MineStudio/data/index.html) component to support more flexible data loading and processing, all the trajectory modals are now decoupled. Users are able to [customize](https://craftjarvis.github.io/MineStudio/data/callbacks.html) their own data processing methods. 
  - Added detailed code comments and docstrings to all the modules, making it easier to understand and use the code.
  - Improved the documentation with more [examples](https://github.com/CraftJarvis/MineStudio/tree/master/tests), tutorials, and a new [API](https://craftjarvis.github.io/MineStudio/api/index.html) reference section.

## Quick Start

``````{tab-set}

`````{tab-item} Installation

```console
conda create -n minestudio --channel=conda-forge python=3.10 openjdk=8 -y
conda activate minestudio
pip install minestudio
# use xvfb for example
sudo apt install -y xvfb mesa-utils libegl1-mesa libgl1-mesa-dev libglu1-mesa-dev

python -m minestudio.simulator.entry
```

The simulator can be started with `xvfb` or `virtualGL`. See [Installation](https://craftjarvis.github.io/MineStudio/overview/getting-started) for more details. We also provide a [Docker image](https://github.com/CraftJarvis/MineStudio/blob/master/README.md#docker) to run MineStudio in a container.

`````

`````{tab-item} Simulator

```python
from minestudio.simulator import MinecraftSim
# Import callbacks from minestudio.simulator.callbacks if needed
sim = MinecraftSim(
    obs_size=(224, 224), render_size=(640, 360),
    callbacks = [...]
)
obs, info = sim.reset()
```

We provide a sets of [callbacks](https://craftjarvis.github.io/MineStudio/simulator/index.html#using-callbacks) to customize the simulator behavior, such as monitoring FPS, sending Minecraft commands, and more. You can also create your own callbacks by inheriting from `BaseCallback`.

`````

`````{tab-item} Data

```python
# Create a raw dataset from 6xx dataset
from minestudio.data import RawDataset
from minestudio.data.minecraft.callbacks import ImageKernelCallback, ActionKernelCallback

dataset = RawDataset(
    dataset_dirs=[
        '/nfs-shared-2/data/contractors/dataset_6xx', 
    ],
    modal_kernel_callbacks=[
        ImageKernelCallback(frame_width=224, frame_height=224, enable_video_aug=False), 
        ActionKernelCallback(enable_prev_action=True, win_bias=1, read_bias=-1),
    ],
    win_len=128, 
    split_ratio=0.9,
    shuffle_episodes=True,
)
item = dataset[0]
print(item.keys())
```
We classify and save the data according to its corresponding modality, e.g. `image`, `action`, etc. You can incorporate your own data modalities via defining custom [callbacks](https://craftjarvis.github.io/MineStudio/data/callbacks.html).

`````

`````{tab-item} Models

```python
from minestudio.models import VPTPolicy
policy = VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.rl_from_early_game_2x").to("cuda")
policy.eval()
memory = None

# Interacting with the simulator
action, memory = policy.get_action(obs, memory, input_shape='*')
obs, reward, terminated, truncated, info = env.step(action)
```

We implemented a template for Minecraft policy model, which can be used to build your own models or load pre-trained models. We also provide a gallery of baseline models, including VPT, Groot, and more. You can find the pre-trained models on [Hugging Face](https://huggingface.co/CraftJarvis).

`````

`````{tab-item} Offline Training

```python
from minestudio.data import MineDataModule
from minestudio.offline import MineLightning
from minestudio.offline.mine_callbacks import BehaviorCloneCallback
mine_lightning = MineLightning(
    mine_policy=policy,
    learning_rate=lr,
    warmup_steps=warmup_steps,
    weight_decay=decay,
    callbacks=[BehaviorCloneCallback(weight=bc_weight)]
)
mine_data = MineDataModule(
    data_params=data_params_dict,
    # additional parameters
)
L.Trainer(
    callbacks=[
        # callbacks for training, e.g. SmartCheckpointCallback, LearningRateMonitor, etc.
    ]
).fit(model=mine_lightning, datamodule=mine_data)
```
We make it easy to train Minecraft agents with offline data. 
Users can define their own data processing methods and training callbacks. The training pipeline is based on [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/), which allows users to easily customize the training process.

`````

`````{tab-item} Online Training

```python
import OmegaConf
from minestudio.online.rollout.start_manager import start_rolloutmanager
from minestudio.online.trainer.start_trainer import start_trainer
def policy_generator():
    return policy
def env_generator():
    # customize env with reward and task callbacks
    return sim
online_config = OmegaConf.create(online_dict) # online_dict is training configuration
start_rolloutmanager(policy_generator, env_generator, online_cfg)
start_trainer(policy_generator, env_generator, online_cfg)
```

We implemented a distributed online training framework based on [Ray](https://docs.ray.io/en/latest/index.html). It supports memory-based policies and simulator crash recovery. Users can customize the training process by defining their own policy and environment generators, as well as the training configuration.

`````

`````{tab-item} Inference

```python
from minestudio.inference import EpisodePipeline, MineGenerator, InfoBaseFilter
import ray

ray.init()
env_generator = partial(MinecraftSim, ...)
agent_generator = lambda: ... # load your policy model here
worker_kwargs = dict(
    env_generator = env_generator, 
    agent_generator = agent_generator,
) # additional parameters for the worker
pipeline = EpisodePipeline(
    episode_generator = MineGenerator(
        num_workers = 8, # the number of workers
        num_gpus = 0.25, # the number of gpus
        max_restarts = 3, # the maximum number of restarts for failed workers
        **worker_kwargs, 
    ),
    episode_filter = InfoBaseFilter(
        key="mine_block",
        regex=".*log.*",
        num=1,
    ), # InfoBaseFilter will label episodes mine more than 1 *.log
)
summary = pipeline.run()
print(summary)
```
The distributed inference framework is designed to efficiently evaluate Minecraft agents in parallel. It allows users to filter episodes based on specific criteria. Users can customize the episode generator and filter to suit their needs.

`````

`````{tab-item} Benchmark

```yaml
custom_init_commands: 
- /give @s minecraft:water_bucket 3
- /give @s minecraft:stone 64
- /give @s minecraft:dirt 64
- /give @s minecraft:shovel{Enchantments:[{id:"minecraft:efficiency",lvl:1}]} 1
text: Build a waterfall in your Minecraft world.
```
We provide a set of benchmark tasks for Minecraft agents, which can be used to evaluate the performance of the agents. The tasks are defined in YAML format, and users can easily add their own tasks. The benchmark tasks are designed to be compatible with [MCU](https://craftjarvis.github.io/MCU/), a Minecraft task automation framework.

`````

``````

## Gallery

### Datasets

We converted the [Contractor Data](https://github.com/openai/Video-Pre-Training?tab=readme-ov-file#contractor-demonstrations) the OpenAI VPT project provided to our trajectory structure and released them to the Hugging Face. (The old dataset is only available in v1.0.6 and earlier versions. From v1.1.0, we have changed the dataset structure to support more flexible data loading and processing.)

<table class="booktabs-table">
  <caption>OpenAI Contractor Datasets</caption>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Description</th>
      <th>Copyright</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/CraftJarvis/minestudio-data-6xx-v110">6xx</a></td>
      <td>Free Gameplay</td>
      <td>OpenAI</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/CraftJarvis/minestudio-data-7xx-v110">7xx</a></td>
      <td>Early Game</td>
      <td>OpenAI</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/CraftJarvis/minestudio-data-8xx-v110">8xx</a></td>
      <td>House Building from Scratch</td>
      <td>OpenAI</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/CraftJarvis/minestudio-data-9xx-v110">9xx</a></td>
      <td>House Building from Random Materials</td>
      <td>OpenAI</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/CraftJarvis/minestudio-data-10xx-v110">10xx</a></td>
      <td>Obtain Diamond Pickaxe</td>
      <td>OpenAI</td>
    </tr>
  </tbody>
</table>

### Models

We have included a gallery of SOTA pre-trained agents in Minecraft, such as VPTs, GROOT, STEVE-1, ROCKETs. These models are trained by us or other researchers and are available on [Hugging Face](https://huggingface.co/CraftJarvis). You can use them directly in your projects or as a starting point for further training and fair comparison.

<table class="booktabs-table">
  <caption>SOTA Minecraft Agents</caption>
  <thead>
    <tr>
      <th>Model</th>
      <th>Description</th>
      <th>Copyright</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/CraftJarvis/MineStudio_VPT.foundation_model_1x" target="_blank" rel="noopener noreferrer">VPT Foundation Model 1x</a></td>
      <td>Behavior cloning on all contractor data, 71M parameters</td>
      <td><a href="https://github.com/openai/Video-Pre-Training" target="_blank" rel="noopener noreferrer">OpenAI</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/CraftJarvis/MineStudio_VPT.foundation_model_2x" target="_blank" rel="noopener noreferrer">VPT Foundation Model 2x</a></td>
      <td>Behavior cloning on all contractor data, 248M parameters</td>
      <td><a href="https://github.com/openai/Video-Pre-Training" target="_blank" rel="noopener noreferrer">OpenAI</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/CraftJarvis/MineStudio_VPT.foundation_model_3x" target="_blank" rel="noopener noreferrer">VPT Foundation Model 3x</a></td>
      <td>Behavior cloning on all contractor data, 0.5B parameters</td>
      <td><a href="https://github.com/openai/Video-Pre-Training" target="_blank" rel="noopener noreferrer">OpenAI</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/CraftJarvis/MineStudio_VPT.bc_early_game_2x" target="_blank" rel="noopener noreferrer">VPT-BC Early Game 2x</a></td>
      <td>Behavior cloning on early game data</td>
      <td><a href="https://github.com/openai/Video-Pre-Training" target="_blank" rel="noopener noreferrer">OpenAI</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/CraftJarvis/MineStudio_VPT.rl_from_house_2x" target="_blank" rel="noopener noreferrer">VPT-RL from House 2x</a></td>
      <td>RL from VPT-BC House</td>
      <td><a href="https://github.com/openai/Video-Pre-Training" target="_blank" rel="noopener noreferrer">OpenAI</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/CraftJarvis/MineStudio_VPT.rl_from_early_game_2x" target="_blank" rel="noopener noreferrer">VPT-RL from Early Game 2x</a></td>
      <td>RL from VPT-BC Early Game</td>
      <td><a href="https://github.com/openai/Video-Pre-Training" target="_blank" rel="noopener noreferrer">OpenAI</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/CraftJarvis/MineStudio_VPT.bc_house_3x" target="_blank" rel="noopener noreferrer">VPT-BC House 3x</a></td>
      <td>Behavior cloning from house building data</td>
      <td><a href="https://github.com/openai/Video-Pre-Training" target="_blank" rel="noopener noreferrer">OpenAI</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/CraftJarvis/MineStudio_VPT.bc_early_game_3x" target="_blank" rel="noopener noreferrer">VPT-BC Early Game 3x</a></td>
      <td>Behavior cloning from early game data</td>
      <td><a href="https://github.com/openai/Video-Pre-Training" target="_blank" rel="noopener noreferrer">OpenAI</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/CraftJarvis/MineStudio_VPT.rl_for_shoot_animals_2x" target="_blank" rel="noopener noreferrer">VPT-RL Shoot Animals 2x</a></td>
      <td>RL for shooting animals</td>
      <td><a href="https://craftjarvis.github.io/" target="_blank" rel="noopener noreferrer">CraftJarvis</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/CraftJarvis/MineStudio_VPT.rl_for_build_portal_2x" target="_blank" rel="noopener noreferrer">VPT-RL Build Portal 2x</a></td>
      <td>RL for building nether portal</td>
      <td><a href="https://craftjarvis.github.io/" target="_blank" rel="noopener noreferrer">CraftJarvis</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/CraftJarvis/MineStudio_GROOT.18w_EMA" target="_blank" rel="noopener noreferrer">GROOT</a></td>
      <td>Self-supervised training to follow demonstration videos</td>
      <td><a href="https://craftjarvis.github.io/" target="_blank" rel="noopener noreferrer">CraftJarvis</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/CraftJarvis/MineStudio_STEVE-1.official" target="_blank" rel="noopener noreferrer">STEVE-1</a></td>
      <td>Language/image-conditioned Minecraft agent</td>
      <td><a href="https://github.com/Shalev-Lifshitz/STEVE-1" target="_blank" rel="noopener noreferrer">STEVE-1</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/CraftJarvis/MineStudio_ROCKET-1.12w_EMA" target="_blank" rel="noopener noreferrer">ROCKET-1</a></td>
      <td>Segment-conditioned agent powered by SAM-2 and VLMs</td>
      <td><a href="https://craftjarvis.github.io/" target="_blank" rel="noopener noreferrer">CraftJarvis</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/phython96/ROCKET-2-1x-22w" target="_blank" rel="noopener noreferrer">ROCKET-2 1x</a></td>
      <td>Segment-conditioned agent capable of handling cross-view instructions</td>
      <td><a href="https://craftjarvis.github.io/" target="_blank" rel="noopener noreferrer">CraftJarvis</a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/phython96/ROCKET-2-1.5x-17w" target="_blank" rel="noopener noreferrer">ROCKET-2 1.5x</a></td>
      <td>Segment-conditioned agent capable of handling cross-view instructions</td>
      <td><a href="https://craftjarvis.github.io/" target="_blank" rel="noopener noreferrer">CraftJarvis</a></td>
    </tr>
  </tbody>
</table>

## Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=CraftJarvis/MineStudio&type=Date)](https://www.star-history.com/#CraftJarvis/MineStudio&Date)

</div>
