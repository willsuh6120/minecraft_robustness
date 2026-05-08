Built-in Models: STEVE-1
======================================================================
`STEVE-1: A Generative Model for Text-to-Behavior in Minecraft <https://arxiv.org/abs/2306.00937>`_

.. admonition:: Quick Facts

    STEVE-1 [1]_ finetunes VPT to follow short-horizon open-ended text and visual instructions without the need of costly human annotations.

Insights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pre-trained foundation models demonstrates suprising ability to be efficiently fine-tuned for becoming instruction-following.
In sequential decision-making domains, two foundation models in Minecraft are released: VPT [2]_ and MineCLIP [3]_, opening intriguing possibilities for exploring the finetuning of instruction-awared decision-making agents.

The authors draw insights from unCLIP [4]_ to propose a two-stage learning framework for training STEVE-1, eliminating laborious human annotations.

Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../_static/image/steve.png
    :width: 800
    :align: center

    STEVE-1 architecture. Image credit: [1]_

To create a policy in Minecraft conditioned text instructions :math:`y`, they ultilize a dataset of (partially) annotated trajectories :math:`[(\tau_1, y_1), (\tau_2, y_2), \dots, (\tau_n, \emptyset)]`
They employ MineCLIP which is capable of generating aligned latents :math:`z_{\tau_{t:t+16}}` and :math:`z_y`, where :math:`z_{\tau_{\text{goal}}} = z_{\tau_{t:t+16}}` is an embedding of 16 consecutive frames.

The instruction-following model is composed of a ``policy`` and a ``prior``:

.. math::

    p(\tau | y) = p(\tau, z_{\tau_{\text{goal}}} | y) = p(\tau | z_{\tau_{\text{goal}}} ) p(z_{\tau_{\text{goal}}} | y),

where the policy generates a trajectory :math:`\tau` conditioned on the aligned latents :math:`z_{\tau_{\text{goal}}}` and the prior generates :math:`z_{\tau_{\text{goal}}}` conditioned on the instruction :math:`y`.

To train the policy, they use a modification of hindsight relabeling to generate goals for each trajectory:

.. figure:: ../_static/image/steve_hindsight.png
    :width: 800
    :align: center

    They randomly select timesteps from episodes and use hindsight relabeling to set the intermediate goals for the trajectory segments to those visual
    MineCLIP embeddings. Image credit: [1]_

By finetuning VPT on this dataset, the policy learns to reach given goal states (visual goals).

To also learn to follow text instructions, they train a conditioned variational autoencoder (CVAE) with Gaussian prior and posterior to translate from a text embedding :math:`z_y` to a visual embedding :math:`z_{\tau_{\text{goal}}}`.
The training objective is a standard ELBO loss:

.. math::
    \mathcal{L}_{\text{prior}}(\phi) = \mathbb{E}_{(z_{\tau_{\text{goal}}}, z_y) \sim \mathcal{D}_{\text{labels}}} \left[ \text{KL}(q_{\phi}(z_{\tau_{\text{goal}}}|z_y)||p(z_{\tau_{\text{goal}}})) - \mathbb{E}_{c \sim q_\phi(z_{\tau_{\text{goal}}}|z_y)}[\log p_\phi(z_{\tau_{\text{goal}}}|c,z_y)] \right].

They ultilize classifier-free guidance to train the policy, where the goal embedding is occasionally droped out during training.
During inference, they compute a combination of logits with and without combination to generate the final trajectory.


Citations
---------

.. [1] Lifshitz S, Paster K, Chan H, et al. Steve-1: A generative model for text-to-behavior in minecraft[J]. Advances in Neural Information Processing Systems, 2024, 36.
.. [2] Baker B, Akkaya I, Zhokov P, et al. Video pretraining (vpt): Learning to act by watching unlabeled online videos[J]. Advances in Neural Information Processing Systems, 2022, 35: 24639-24654.
.. [3] Fan L, Wang G, Jiang Y, et al. Minedojo: Building open-ended embodied agents with internet-scale knowledge[J]. Advances in Neural Information Processing Systems, 2022, 35: 18343-18362.
.. [4] Ramesh A, Dhariwal P, Nichol A, et al. Hierarchical text-conditional image generation with clip latents[J]. arXiv preprint arXiv:2204.06125, 2022, 1(2): 3.