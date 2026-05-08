Built-in Models: GROOT
======================================================================

.. admonition:: Quick Facts
    
    - GROOT is an open-world controller that follows open-ended instructions by using reference videos as expressive goal specifications, eliminating the need for text annotations. 
    - GROOT leverages a causal transformer-based encoder-decoder architecture to self-supervise the learning of a structured goal space from gameplay videos.

Insights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To develop an effective instruction-following controller, defining a robust goal representation is essential. Unlike previous approaches, such as using language descriptions or future images (e.g., Steve-1), GROOT leverages reference videos as goal representations. These gameplay videos serve as a rich and expressive source of information, enabling the agent to learn complex behaviors effectively. The paper frames the learning process as future state prediction, allowing the agent to follow demonstrations seamlessly.

Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Formally, the future state prediction problem is defined as maximizing the log-likelihood of future states given past ones: :math:\log p_{\theta}(s_{t+1:T} | s_{0:t}). By introducing :math:`g` as a latent variable conditioned on past states, the evidence lower bound (ELBO) can be expressed as:

.. math::

    \log p_{\theta}(s_{t+1:T} | s_{0:t}) &= \log \sum_g p_{\theta}(s_{t+1:T}, g | s_{0:t}) \\
    &\geq \mathbb{E}_{g \sim q_\phi(\cdot | s_{0:t})} \left[ \log p_{\theta}(s_{t+1:T} | g, s_{0:t}) \right] - D_{\text{KL}}(q_\phi(g | s_{0:T}) || p_\theta(g|s_{0:t})),

where :math:`D_{\text{KL}}` is the Kullback-Leibler divergence, and :math:`q_\phi` represents the variational posterior.

This objective can be further simplified using the transition function :math:`p_{\theta}(s_{t+1}|s_{0:t},a_t)` and a goal-conditioned policy (to be learned) :math:`\pi_{\theta}(a_t|s_{0:t},g)`:

.. math::

    \log p_{\theta}(s_{t+1:T} | s_{0:t}) \geq \sum_{\tau = t}^{T - 1} \mathbb{E}_{g \sim q_\phi(\cdot | s_{0:T}), a_\tau \sim p_{\theta}(\cdot | s_{0:\tau+1})} \left[ \log \pi_{\theta}(a_{\tau} | s_{0:\tau}, g) \right] - D_{\text{KL}}(q_\phi(g | s_{0:T}) || p_\theta(g|s_{0:t})),

where :math:`q_\phi(\cdot|s_{0:T})` is implemented as a video encoder, and :math:`p_{\theta}(\cdot|s_{0:\tau+1})` represents the Inverse Dynamic Model (IDM), which predicts actions to transition to the next state and is typically a pretrained model.
Please refer to the `paper <https://arxiv.org/pdf/2310.08235>`_ for more details.

Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../_static/image/groot_architecture.png
    :width: 800
    :align: center

    GROOT agent architecture.

The GROOT agent consists of a video encoder and a policy decoder.
The video encoder is a non-causal transformer that extracts semantic information and generates goal embeddings.
The policy is a causal transformer
decoder that receives the goal embeddings as the instruction and autoregressively translates the state
sequence into a sequence of actions.

For more details, a vision backbone is used to extract features from the video frames, which are then fed into the transformer encoder.
The non-causal transformer outputs a set of summary tokens :math:`\{\hat{c}_{1:N}\}`, which are used to sample a set of embeddings :math:`\{g_{1:N}\}` using the reparameterization trick: :math:`g_t \sim \mathcal{N}(\mu(\hat{c}_t), \sigma(\hat{c}_t))`.
The decoder then takes the goal embeddings and the state sequence as input and autoregressively predicts the action sequence.
To see a detailed architecture, please refer to the `paper <https://arxiv.org/pdf/2310.08235>`_. and the `official repository <https://github.com/CraftJarvis/GROOT>`_.

Our Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our implementation of GROOT mainly consists of 5 components: ``LatentSpace``, ``VideoEncoder``, ``ImageEncoder``, ``Decoder``, and ``GrootPolicy`` in ``minestudio/models/groot_one/body.py``.

.. dropdown:: Click to see the arguments for each component of GROOT
    :icon: unlock

    .. list-table::
        :widths: 25 25 25
        :header-rows: 1
    
        * - `Argument Name`
          - `Description`
          - `Component Type`
        * - ``hiddim: int=1024``
          - The dimension of the hidden state.
          - All components
        * - ``num_spatial_layers: int=2``
          - The number of spatial layers in the pooling transformer.
          - ``VideoEncoder``
        * - ``num_temporal_layers: int=2``
          - The number of temporal layers in the video encoder.
          - ``VideoEncoder``
        * - ``num_heads: int=8``
          - The number of heads in the multi-head attention.
          - ``VideoEncoder``, ``ImageEncoder``, ``Decoder``
        * - ``dropout: float=0.1``
          - The dropout rate.
          - ``VideoEncoder``, ``ImageEncoder``
        * - ``num_layers: int=2``
          - The number of layers in the transformer.
          - ``ImageEncoder``, ``Decoder``
        * - ``timesteps: int=128``
          - The number of timesteps for an input sequence.
          - ``Decoder``
        * - ``mem_len: int=128``
          - The memory length for the Transformer XL.
          - ``Decoder``
        * - ``backbone: str='efficientnet_b0.ra_in1k'``
          - The vision backbone for feature extraction.
          - ``GrootPolicy``
        * - ``freeze_backbone: bool=True``
          - Whether to freeze the backbone weights.
          - ``GrootPolicy``
        * - ``video_encoder_kwargs: Dict={}``
          - The keyword arguments for the video encoder.
          - ``GrootPolicy``
        * - ``image_encoder_kwargs: Dict={}``
          - The keyword arguments for the image encoder.
          - ``GrootPolicy``
        * - ``decoder_kwargs: Dict={}``
          - The keyword arguments for the decoder.
          - ``GrootPolicy``
        * - ``action_space=None``
          - The action space for the environment.
          - ``GrootPolicy``

Here we provide a brief overview and workflow of the components:

.. dropdown:: Click to see the workflow of GROOT
    :icon: unlock

    1. The ``forward`` method of `GrootPolicy` takes arguments ``input: Dict`` and ``memory: Optional[List[torch.Tensor]] = None``.
    2. The ``input['image`]`` firstly get rearranged and transformed for ``self.backbone``. Then image features are extracted using the backbone and get updimensioned.
    3. If ``reference`` is in the input, which means a demonstration video is provided, the reference video is encoded the same way as the input image. Otherwise, reference video is the input sequence itself for self-supervised learning.
    4. The posterior distribution is calculated using the video encoder, and the goal embeddings are sampled.
    5. The prior distribution is calculated using the image encoder with only the first frame.
    6. The image features and goal embeddings are concatenated and fused to form the input for the decoder.
    7. The decoder autoregressively predicts the action logits as well as generates next memory.

Training GROOT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To implement the training objective of GROOT, we add a ``kl_divergence`` callback in ``minestudio/train/mine_callbacks``. This callback calculates the KL divergence between the posterior and prior distributions and adds it to the loss.

To train GROOT, we provide a configuration file ``minestudio/tutorials/train/3_pretrain_groots/groot_config.yaml``.
Specify this file path with hydra to start training:

.. code-block:: bash

   cd minestudio/tutorials/train/3_pretrain_groots
   python main.py

Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example of how to evaluate the trained GROOT model. Provide it with a reference video and let it run!

.. code-block:: python

    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback
    from minestudio.models import GrootPolicy, load_groot_policy
    import numpy as np
    import av

    if __name__ == '__main__':
        
        policy = load_groot_policy(
            ckpt_path = # specify the checkpoint path here,
        ).to("cuda")
        
        resolution = (224, 224) # specify the observation size

        env = MinecraftSim(
            obs_size = resolution,
            preferred_spawn_biome = "forest", 
            callbacks = [
                RecordCallback(record_path = "./output", fps = 30, frame_type="pov"),
                SpeedTestCallback(50),
            ]
        )

        ref_video_path = # specify the reference video path here

        memory = None
        obs, info = env.reset()
        obs["ref_video_path"] = ref_video_path

        for i in range(1200):
            action, memory = policy.get_action(obs, memory, input_shape='*')
            obs, reward, terminated, truncated, info = env.step(action)

        env.close()

.. note::

    We provide a set of reference videos in `huggingface <https://huggingface.co/datasets/CraftJarvis/MinecraftReferenceVideos>`_.

An example of inference code using our framework can be found in :any:`inferece-groot`.