Built-in Models: ROCKET-1
======================================================================

`ROCKET-1: Mastering Open-World Interaction with Visual-Temporal Context Prompting <https://arxiv.org/abs/2410.17856>`_

.. admonition:: Quick Facts

    ROCKET-1 is actually a segmentation-conditioned policy. Human or other high-level reasoners can pinpoint an object by segmenting it from the background. ROCKET-1 leverages this ability to interact with the Minecraft environment by using a segmentation mask as a prompt.

Insights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hierarchical agent architectures have become a popular approach to addressing open-world interaction challenges. These architectures leverage the reasoning capabilities of large language models (LLMs) to decompose tasks into subgoals, which are then communicated to the low-level controller through language. However, this method struggles to convey precise spatial details. This paper proposes visual-temporal context prompting, a novel technique that enables the high-level reasoner and low-level controller to communicate interaction intents more effectively by utilizing semantic segmentation of the current visual frames. This approach significantly improves the transmission of spatial details, enhancing the agent’s interaction efficiency. 

Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../_static/image/rocket-pipeline.png
    :width: 800
    :align: center

    ROCKET-1 policy architecture

To train ROCKET1, we need to prepare interaction trajectory data in the format: :math:`\tau = (o_{1:T} , a_{1:T} , 𝑚_{1:T} , c_{1:T} )`, where :math:`o_t \in \mathbb{R}^{3\times H \times W}` is the visual observation at time :math:`t`, :math:`m_t \in \{0, 1\}_{1 \times H \times W}` is a binary mask highlighting the object in :math:`o_t` for future interaction, :math:`c_t \in \mathbb{N}` denotes the interaction type, and :math:`a_t` is the action. If both :math:`m_t` and :math:`c_t` are zeros, no region is highlighted at
:math:`o_t`. 

.. hint::
    
    All these trajectory data, including segmentation masks, are provided by our ``minestudio.data`` part. 

The optimizing objective is to maximize the log-likelihood of the interaction trajectory data:

.. math::

    \mathcal{L} = -\sum_{t=1}^{|\tau|} \log \pi(a_t | o_{1:t}, m_{1:t} \odot w_{1:t}, c_{1:t} \odot w_{1:t})

where :math:`w_t \sim \text{Bernoulli}(1-p)` represents a mask, with :math:`p` denoting the dropping probability, :math:`\odot` denotes the product operation over time dimension. 



Our Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We utilize EfficientNet as the visual backbone, modifying its input channels from 3 to 4 to accommodate semantic segmentation masks. Additionally, we employ PyTorch’s built-in ``TransformerEncoder`` for self-attention pooling of visual patches. The visual features and interaction types are then alternately arranged into a token sequence, which is processed using the TransformerXL module from the `VPT repository <https://github.com/openai/Video-Pre-Training>`_. For the action head, we reuse the hierarchical action head implementation from the VPT repository. 

.. note::
    
    You can find our implementation in the module ``minestudio.models.rocket_one``.


Train ROCKET-1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide a simple example in the tutorials ``minestudio.tutorials.train.2_pretrain_vpts``. 

You can simply copy the directory and change dir to the copied directory. Then, run the following command: 

.. code-block:: console

    $ python train.py --config rocket_config.yaml



Evaluate ROCKET-1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluating the trained ROCKET-1 in your own scripts is easy:

.. code-block:: python

    import torch
    from minestudio.models import load_rocket_policy, RocketPolicy

    model = load_rocket_policy('/path/to/rocket.ckpt').to('cuda')
    # or 
    model = RocketPolicy.from_pretrained("CraftJarvis/MineStudio_ROCKET-1.12w_EMA").to("cuda")
    model.eval()
    input = {
        'image': torch.zeros(224, 224, 3).to("cuda"), 
        'segment': {
            'obj_id': torch.tensor(0).to("cuda"),
            'obj_mask': torch.zeros(224, 224).to("cuda"),
        }  
    }
    memory = None
    output, memory = model.get_action(input, memory, input_shape='*')


We provide a interactive gradio page to evaluate the ROCKET-1 model. You can run the following command to start the server:

.. code-block:: console

    $ python -m minestudio.tutorials.inference.evaluate_rocket.rocket_gradio \
        --port 7862 \
        --model-path '/path/to/rocket.ckpt' \
        --sam-path '/path/to/sam2

Then, open your browser and go to ``http://localhost:7862`` to see the evaluation page.

.. figure:: ../_static/image/gradio-rocket.png
    :width: 800
    :align: center

    ROCKET-1 evaluation page

