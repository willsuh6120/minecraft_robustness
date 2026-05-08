
# Tutorial: Pre-Training ROCKET-1 from Scratch


Let’s break down the parameters required to train ROCKET-1 step by step.

We begin with the introduction of dependencies:
```python
import hydra
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from einops import rearrange
from typing import Dict, Any, Tuple

from minestudio.data import MineDataModule
from minestudio.models import RocketPolicy
from minestudio.offline import MineLightning
from minestudio.offline.utils import convert_to_normal
from minestudio.offline.mine_callbacks import BehaviorCloneCallback
from minestudio.offline.lightning_callbacks import SmartCheckpointCallback, SpeedMonitorCallback, EMA
```
```{note}
We observed that the dependencies include `SmartCheckpointCallback` and `EMA`. The `SmartCheckpointCallback` is responsible for saving model weights and checkpoints, while `EMA` implements Exponential Moving Average. EMA is typically used during training to smooth model weights, enhancing the model's generalization capability. It also maintains an additional copy of the model.
```


Next, we proceed to construct the policy model for ROCKET-1:
```python
rocket_policy = RocketPolicy(
    backbone="efficientnet_b0.ra_in1k",
    hiddim=1024,
    num_heads=8,
    num_layers=4,
    timesteps=128,
    mem_len=128,
)
```
```{note}
The `timesteps` parameter represents the maximum sequence length within a batch during training. The `mem_len` parameter specifies the memory length supported by [TransformerXL](https://arxiv.org/abs/1901.02860), which stores cached `key` and `value` data during both inference and training. These cached values are directly involved in the attention computation. 
```

Next, configure the offline lightning module:
```python
mine_lightning = MineLightning(
    mine_policy=rocket_policy, 
    log_freq=20,
    learning_rate=0.00004,
    weight_decay=0.001,
    warmup_steps=2000,
    callbacks=[
        BehaviorCloneCallback(weight=args.objective_weight),
    ], 
    hyperparameters=convert_to_normal(args),
)
```
```{note}
The `learning_rate` and `weight_decay` parameters are empirically set. The `warmup_steps` parameter specifies the number of warmup steps for the learning rate, which is crucial when training a Transformer model from scratch. 

The `callbacks` parameter defines the optimization objectives. The MineStudio offline training module supports setting multiple optimization objectives. For example, `BehaviorCloneCallback` is used to specify the optimization objective for behavior cloning.

The `hyperparameters` parameter is used to define various hyperparameters. The `convert_to_normal` function converts parameters from a Hydra configuration file into a standard dictionary format, which is then logged in `wandb`.
```

Next, configure the dataloader:
```python
mine_data = MineDataModule(
    data_params=dict(
        mode='raw',
        dataset_dirs=[
            '/nfs-shared-2/data/contractors/dataset_6xx',
            '/nfs-shared-2/data/contractors/dataset_7xx',
            '/nfs-shared-2/data/contractors/dataset_8xx',
            '/nfs-shared-2/data/contractors/dataset_9xx',
            '/nfs-shared-2/data/contractors/dataset_10xx'
        ],
        frame_width=224,
        frame_height=224,
        win_len=128,
        enable_segment=True,
    ),
    batch_size=8,
    num_workers=8,
    prefetch_factor=4,
    split_ratio=0.95, 
    shuffle_episodes=True,
    episode_continuous_batch=True,
)
```

```{note}
The `win_len` parameter defines the sequence length within a batch, which should be set to the same value as `timesteps`. 

Setting `enable_segment=True` enables the use of semantic segmentation information from the data, which provides the conditional information required by the ROCKET-1 model.

The `shuffle_episodes=True` parameter indicates that the trajectory arrangement in the original data will be shuffled, affecting the `train-val` split.

Setting `episode_continuous_batch=True` ensures that continuous episode segments are used as batches, which influences the sampling strategy of the dataloader.
```

Configure the `Trainer` and initiate training:
```python
callbacks=[
    LearningRateMonitor(logging_interval='step'), 
    SpeedMonitorCallback(),
    SmartCheckpointCallback(
        dirpath='./weights', filename='weight-{epoch}-{step}', save_top_k=-1, 
        every_n_train_steps=args.save_freq, save_weights_only=True,
    ), 
    SmartCheckpointCallback(
        dirpath='./checkpoints', filename='ckpt-{epoch}-{step}', save_top_k=1, 
        every_n_train_steps=args.save_freq+1, save_weights_only=False,
    ), 
    EMA(
        decay=0.999, 
        validate_original_weights=True, 
        every_n_steps=8, 
        cpu_offload=False, 
    )
]

L.Trainer(
    logger=WandbLogger(project="minestudio"), 
    devices=8, 
    precision='bf16', 
    strategy='ddp_find_unused_parameters_true', 
    use_distributed_sampler=not episode_continuous_batch,
    callbacks=callbacks, 
    gradient_clip_val=1.0, 
).fit(
    model=mine_lightning, 
    datamodule=mine_data, 
    ckpt_path=ckpt_path,
)
```

To monitor the training process effectively, we use multiple callback functions. 

- `LearningRateMonitor` records changes in the learning rate.  
- `SpeedMonitorCallback` tracks the training speed.  
- `SmartCheckpointCallback` saves model weights and checkpoints.  
- `EMA` implements Exponential Moving Average.

The parameter `use_distributed_sampler=not episode_continuous_batch` indicates that when `episode_continuous_batch=True`, the dataloader will automatically use our distributed batch sampler. While configuring the `Trainer`, we must set `use_distributed_sampler=False`.

Finally, we start the training process using the `fit` function.  
- The `model` parameter corresponds to the configured `mine_lightning`.  
- The `datamodule` parameter refers to the configured `mine_data`.  
- The `ckpt_path` parameter specifies the checkpoint path. To resume training from a specific checkpoint, set `ckpt_path` to the path of that checkpoint. Otherwise, set it to `None`.