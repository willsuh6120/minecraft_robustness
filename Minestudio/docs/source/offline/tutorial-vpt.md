
# Tutorial: Fine-tuning VPT to a Hunter

Fine-tune a VPT policy in MineStudio is really simple. 

The following code snippet shows how to finetune a VPT policy to hunt animals in Minecraft using offline data.

1. Import some dependencies:
    ```python
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import LearningRateMonitor
    # below are MineStudio dependencies
    from minestudio.data import MineDataModule
    from minestudio.offline import MineLightning
    from minestudio.models import load_vpt_policy, VPTPolicy
    from minestudio.offline.mine_callbacks import BehaviorCloneCallback
    from minestudio.offline.lightning_callbacks import SmartCheckpointCallback, SpeedMonitorCallback
    ```

2. Configure the policy model and the training process:
    ```python
    policy = VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.foundation_model_2x")
    mine_lightning = MineLightning(
        mine_policy=policy,
        learning_rate=0.00004,
        warmup_steps=2000,
        weight_decay=0.000181,
        callbacks=[BehaviorCloneCallback(weight=0.01)]
    )
    ```

3. Configure the data module that contains all the `kill_entity` trajectory segments:
    ```python
    episode_continuous_batch = True
    mine_data = MineDataModule(
        data_params=dict(
            mode='event', 
            dataset_dirs=['10xx'],
            win_len=128,
            frame_width=128,
            frame_height=128,
            event_regex="minecraft.kill_entity:.*",
            bias=16,
            min_nearby=64,
        ),
        batch_size=8,
        num_workers=8,
        prefetch_factor=4,
        split_ratio=0.9, 
        shuffle_episodes=True,
        episode_continuous_batch=episode_continuous_batch,
    )
    ```

    ```{warning}
    If `episode_continuous_batch=True`, then the dataloader will automatically use our distributed batch sampler. When configuring the `trainer`, we need to set `use_distributed_sampler=False`. 
    ```

4. Configure the `trainer` with your preferred [PyTorch Lightning](https://lightning.ai/) callbacks:
    ```python
    L.Trainer(
        logger=WandbLogger(project="minestudio-vpt"), 
        devices=8, 
        precision="bf16", 
        strategy='ddp_find_unused_parameters_true', 
        use_distributed_sampler=not episode_continuous_batch, 
        gradient_clip_val=1.0, 
        callbacks=[
            LearningRateMonitor(logging_interval='step'), 
            SpeedMonitorCallback(),
            SmartCheckpointCallback(
                dirpath='./weights', filename='weight-{epoch}-{step}', save_top_k=-1, 
                every_n_train_steps=2000, save_weights_only=True,
            ), 
            SmartCheckpointCallback(
                dirpath='./checkpoints', filename='ckpt-{epoch}-{step}', save_top_k=1, 
                every_n_train_steps=2000+1, save_weights_only=False,
            )
        ]
    ).fit(model=mine_lightning, datamodule=mine_data)
    ```