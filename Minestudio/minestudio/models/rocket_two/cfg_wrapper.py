import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple

from minestudio.models.base_policy import dict_map, recursive_tensor_op


class CFGWrapper:
    def __init__(self, model, k: float = 1.0, base_model=None):
        self.model = model
        self.base_model = base_model
        self.k = float(k)
        self.cache_latents = {}

    def initial_state(self):
        base_model = self.base_model if self.base_model is not None else self.model
        return (self.model.initial_state(), base_model.initial_state())

    @staticmethod
    def _squeeze_latents(latents):
        return dict_map(lambda tensor: tensor[0][0], latents)

    @staticmethod
    def _build_unconditioned_input(input: Dict[str, Any]) -> Dict[str, Any]:
        base_input = input.copy()
        base_input["cross_view"] = dict(base_input["cross_view"])
        base_input["cross_view"]["cross_view_image"] = np.zeros_like(base_input["cross_view"]["cross_view_image"])
        base_input["cross_view"]["cross_view_obj_id"] = torch.zeros_like(base_input["cross_view"]["cross_view_obj_id"]) - 1
        base_input["cross_view"]["cross_view_obj_mask"] = torch.zeros_like(base_input["cross_view"]["cross_view_obj_mask"])
        return base_input

    @torch.inference_mode()
    def get_action(
        self,
        input: Dict[str, Any],
        state_in: Optional[List[torch.Tensor]],
        deterministic: bool = False,
        input_shape: str = "BT*",
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        if state_in is None:
            state_in = (None, None)
        cond_state_in, base_state_in = state_in

        cond_input = input.copy()
        cond_latents, cond_state_out = self.get_action_once(
            self.model,
            cond_input,
            cond_state_in,
            deterministic,
            input_shape,
            **kwargs,
        )

        base_model = self.base_model if self.base_model is not None else self.model
        base_input = self._build_unconditioned_input(input)
        base_latents, base_state_out = self.get_action_once(
            base_model,
            base_input,
            base_state_in,
            deterministic,
            input_shape,
            **kwargs,
        )

        pi_logits = {
            "buttons": (1 + self.k) * cond_latents["pi_logits"]["buttons"] - self.k * base_latents["pi_logits"]["buttons"],
            "camera": (1 + self.k) * cond_latents["pi_logits"]["camera"] - self.k * base_latents["pi_logits"]["camera"],
        }
        self.cache_latents = self._squeeze_latents(cond_latents)
        self.cache_latents["cond_pi_logits"] = self._squeeze_latents(cond_latents["pi_logits"])
        self.cache_latents["base_pi_logits"] = self._squeeze_latents(base_latents["pi_logits"])
        self.cache_latents["pi_logits"] = self._squeeze_latents(pi_logits)
        action = self.model.pi_head.sample(pi_logits, deterministic)
        state_out = (
            recursive_tensor_op(lambda x: x[0], cond_state_out),
            recursive_tensor_op(lambda x: x[0], base_state_out),
        )
        return dict_map(lambda tensor: tensor[0][0], action), state_out

    @torch.inference_mode()
    def get_action_once(
        self,
        model,
        input: Dict[str, Any],
        state_in: Optional[List[torch.Tensor]],
        deterministic: bool = False,
        input_shape: str = "BT*",
        **kwargs,
    ):
        assert input_shape == "*"
        input = dict_map(model._batchify, input)
        if state_in is not None:
            state_in = recursive_tensor_op(lambda x: x.unsqueeze(0), state_in)
        latents, state_out = model.forward(input, state_in, **kwargs)
        return latents, state_out
