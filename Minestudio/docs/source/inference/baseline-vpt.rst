
.. _inferece-vpt:

Tutorial: Inference with VPT
----------------------------

We can easily perform batch evaluations on a VPT model. A simple example is evaluating the success rate of a policy model fine-tuned by OpenAI using reinforcement learning for the diamond-mining task: 

First, import the necessary dependencies. Since our episode generator, ``MineGenerator``, is implemented based on ``ray``, it is essential to initialize ``ray`` beforehand. 

.. code-block:: python

    import ray
    from functools import partial
    
    from minestudio.inference import EpisodePipeline, MineGenerator, InfoBaseFilter
    from minestudio.models import load_vpt_policy
    from minestudio.simulator import MinecraftSim

    ray.init()


Next, we create the ``env_generator`` and ``agent_generator`` separately to enable workers to generate resources. 

.. code-block:: python

    env_generator = partial(
        MinecraftSim, 
        obs_size=(128, 128), 
        preferred_spawn_biome="forest", 
    )
    agent_generator = lambda: VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.rl_from_early_game_2x")

Next, we configure the worker parameters, including:  
    - A maximum of 12,000 steps per episode,  
    - Each worker generating 2 episodes,  
    - The output folder set to ``./output``,  
    - The output video format set to ``h264``.

.. code-block:: python

    worker_kwargs = dict(
        env_generator=env_generator, 
        agent_generator=agent_generator,
        num_max_steps=12000,
        num_episodes=2,
        tmpdir="./output",
        image_media="h264",
    )

Finally, we create an ``EpisodePipeline`` object, passing ``MineGenerator`` as the episode generator and ``InfoBaseFilter`` as the episode filter.

.. code-block:: python

    pipeline = EpisodePipeline(
        episode_generator=MineGenerator(
            num_workers=8,
            num_gpus=0.25,
            max_restarts=3,
            **worker_kwargs, 
        ), 
        episode_filter=InfoBaseFilter(
            key="mine_block",
            regex=".*diamond_ore.*",
            num=1,
        ),
    )
    summary = pipeline.run()
    print(summary)

.. note::

    We initialized 8 workers, with each worker utilizing 0.25 of a GPU and allowing up to 3 restarts.

    We used the built-in ``InfoBaseFilter`` to process the generated episodes, including detecting whether a ``mine_block`` event occurred with the ``val`` set to ``diamond_ore``.

The summary of the pipeline will be printed to the console, showing the success rate and the number of episode.
After the pipeline is finished, the console will print the summary of the pipeline like the following:

.. code-block:: python

    ... ...
    {'num_yes': 4, 'num_episodes': 16, 'yes_rate': '25.00%'}
    (Worker pid=1011772) Speed Test Status: 
    (Worker pid=1011772) Average Time: 0.02 
    (Worker pid=1011772) Average FPS: 56.11 