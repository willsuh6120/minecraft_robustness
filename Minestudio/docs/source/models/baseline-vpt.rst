
Built-in Models: VPT
======================================================================

`Video Pretraining (vpt): Learning to Act by Watching Unlabeled Online Videos <https://arxiv.org/abs/2206.11795>`_

.. admonition:: Quick Facts

    The paper utilizes `next token prediction` to pre-train a foundational policy model ``VPT`` on the Minecraft trajectory dataset and demonstrates that it can be efficiently fine-tuned for other complex tasks (such as mining diamonds from scratch). 

Insights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To develop foundational policy models in the Minecraft domain, it is crucial to pretrain on a large-scale dataset of human expert trajectories. This enables the model to learn a diverse range of behaviors and account for as many edge cases as possible. However, acquiring such trajectory data—which includes sequences of both states and actions—can be challenging, whereas video data from platforms like YouTube is relatively easier to obtain. To address this issue, this paper proposes leveraging a small amount of manually labeled trajectory data to train an Inverse Dynamics Model (IDM), which is then used to generate pseudo-labels for video data. This approach significantly scales up the trajectory dataset available for training. Inspired by GPT’s pretraining paradigm, VPT (Video Pretrained Transformer) trains a foundational policy model by predicting the next action, and subsequently fine-tunes it on specific tasks using reinforcement learning, achieving remarkable performance. 

.. figure:: ../_static/image/vpt.png
    :width: 800
    :align: center

Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simple behavior cloning is used to train the foundation policy.

.. math::

    \text{min}_\theta \sum_{t \in [1 \cdots T ]} - \log \pi_\theta (a_t | o_1, \cdots, o_t), \ \text{where} \ a_t \in p_{\text{IDM}} (a_t | o_1, \cdots, o_t, \cdots, o_T)






Our Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We mostly used the code provided by the `original repo <https://github.com/openai/Video-Pre-Training>`_ and encapsulated it using the Policy Template ``MinePolicy`` designed in MineStudio.

.. code-block:: python

    @Registers.model.register
    class VPTPolicy(MinePolicy):

        def __init__(self, policy_kwargs, action_space=None):
            super().__init__(hiddim=policy_kwargs["hidsize"], action_space=action_space)
            self.net = MinecraftPolicy(**policy_kwargs)
            self.cached_init_states = dict()

        def initial_state(self, batch_size: int=None):
            if batch_size is None:
                return [t.squeeze(0).to(self.device) for t in self.net.initial_state(1)]
            else:
                if batch_size not in self.cached_init_states:
                    self.cached_init_states[batch_size] = [t.to(self.device) for t in self.net.initial_state(batch_size)]
                return self.cached_init_states[batch_size]

        def forward(self, input, state_in, **kwargs):
            B, T = input["image"].shape[:2]
            first = torch.tensor([[False]], device=self.device).repeat(B, T)
            state_in = self.initial_state(B) if state_in is None else state_in
            (pi_h, v_h), state_out = self.net(input, state_in, context={"first": first})
            pi_logits = self.pi_head(pi_h)
            vpred = self.value_head(v_h)
            latents = {'pi_logits': pi_logits, 'vpred': vpred}
            return latents, state_out


.. hint::

    ``MinecraftPolicy`` comes from the original implementation of the `VPT repository <https://github.com/openai/Video-Pre-Training>`_.  

Train VPT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide a simple example in the tutorials ``minestudio.tutorials.train.1_finetune_vpts``. 

You can simply copy the directory and change dir to the copied directory. Then, run the following command: 

.. code-block:: console

    $ python train.py --config vpt_config.yaml


Evaluate VPT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example of how to evaluate the trained VPT model. Provide a model path and weights path and let it run! 

.. code-block:: python

    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import RecordCallback
    from minestudio.models import load_vpt_policy

    policy = load_vpt_policy(
        model_path="/path/to/foundation-model-2x.model", 
        weights_path="/path/to/foundation-model-2x.weights"
    ).to("cuda")
    policy.eval()

    env = MinecraftSim(
        obs_size=(128, 128), 
        callbacks=[RecordCallback(record_path="./output", fps=30, frame_type="pov")]
    )
    memory = None
    obs, info = env.reset()
    for i in range(1200):
        action, memory = policy.get_action(obs, memory, input_shape='*')
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()


