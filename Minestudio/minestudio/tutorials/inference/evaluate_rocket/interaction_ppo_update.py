import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

try:
    from minestudio.tutorials.inference.evaluate_rocket.corridor_signal import load_corridor_causal_mask
    _CORRIDOR_OK = True
except ImportError:
    try:
        from corridor_signal import load_corridor_causal_mask  # running directly from directory
        _CORRIDOR_OK = True
    except ImportError:
        _CORRIDOR_OK = False

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from minestudio.models import CrossViewRocket, RocketPolicy, load_cross_view_rocket, load_rocket_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="interaction_posttrain run directory containing episode_fragment.pt files")
    parser.add_argument("--model-path", type=str, default="CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
    parser.add_argument("--kl-anchor-model-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluate_rocket/interaction_ppo_update")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--policy-coef", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--update-fragment-batch-size", type=int, default=1)
    parser.add_argument(
        "--loss-focus-mode",
        type=str,
        default="uniform",
        choices=["uniform", "suffix_success", "prefix_success", "advantage_top_k", "lateral_top_k", "corridor_commit"],
    )
    parser.add_argument("--loss-focus-suffix-len", type=int, default=32)
    parser.add_argument("--loss-focus-context-len", type=int, default=96)
    parser.add_argument("--loss-focus-weight", type=float, default=4.0)
    parser.add_argument("--loss-focus-top-k", type=int, default=32,
                        help="For advantage_top_k / lateral_top_k: number of steps to upweight per successful episode.")
    parser.add_argument("--clip-vloss", action="store_true")
    parser.add_argument("--normalize-advantage", action="store_true")
    parser.add_argument("--max-fragments", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-kind", type=str, default="rocket1", choices=["rocket1", "rocket2"])
    parser.add_argument("--cfg-coef", type=float, default=0.0)
    parser.add_argument(
        "--cfg-policy-mode",
        type=str,
        default="full",
        choices=["full", "frozen_base"],
    )
    parser.add_argument("--cfg-base-ref-model-path", type=str, default="")
    parser.add_argument(
        "--trainable-scope",
        type=str,
        default="heads",
        choices=["value", "heads", "heads_last", "crossview_small", "full_except_vision"],
    )
    return parser.parse_args()


def to_device_tree(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return {str(key): to_device_tree(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [to_device_tree(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(to_device_tree(value, device) for value in data)
    return data


def resolve_cfg_base_ref_model_path(model_path: str, cfg_policy_mode: str, cfg_base_ref_model_path: str) -> str:
    if str(cfg_policy_mode or "full") != "frozen_base":
        return ""
    explicit_path = str(cfg_base_ref_model_path or "").strip()
    if explicit_path:
        return explicit_path
    return str(model_path)


def resolve_kl_anchor_model_path(model_path: str, kl_anchor_model_path: str) -> str:
    explicit_path = str(kl_anchor_model_path or "").strip()
    if explicit_path:
        return explicit_path
    return str(model_path)


def unsqueeze_tree(data, dim: int = 0):
    if isinstance(data, torch.Tensor):
        return data.unsqueeze(dim)
    if isinstance(data, Mapping):
        return {str(key): unsqueeze_tree(value, dim) for key, value in data.items()}
    if isinstance(data, list):
        return [unsqueeze_tree(value, dim) for value in data]
    if isinstance(data, tuple):
        return tuple(unsqueeze_tree(value, dim) for value in data)
    return data


def normalize_prev_action_tree(data):
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            return data.item()
        return torch.from_numpy(data)
    if isinstance(data, Mapping):
        normalized = {}
        for key, value in data.items():
            key_str = str(key)
            normalized_value = normalize_prev_action_tree(value)
            if isinstance(normalized_value, torch.Tensor):
                if key_str == "camera":
                    normalized_value = normalized_value.to(dtype=torch.float32)
                else:
                    normalized_value = normalized_value.to(dtype=torch.long)
            normalized[key_str] = normalized_value
        return normalized
    if isinstance(data, tuple):
        data = list(data)
    if isinstance(data, list):
        if not data:
            return torch.empty(0, dtype=torch.float32)
        normalized_items = [normalize_prev_action_tree(item) for item in data]
        if all(isinstance(item, torch.Tensor) for item in normalized_items):
            return torch.stack([item if item.ndim > 0 else item.reshape(()) for item in normalized_items], dim=0)
        if all(not isinstance(item, (Mapping, list, tuple)) for item in normalized_items):
            try:
                array = np.asarray(normalized_items)
                if array.dtype != object:
                    return torch.as_tensor(array)
            except Exception:
                pass
        try:
            tensor = torch.as_tensor(data)
        except Exception:
            tensor = torch.tensor(data)
        return tensor
    return torch.as_tensor(data)


def save_rocket_checkpoint(model, output_path: Path, metadata: Dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_config = {}
    if isinstance(model, CrossViewRocket):
        model_config = {
            "hiddim": int(model.view_cls_tokens.shape[-1]),
            "num_layers": int(len(model.view_resampler.layers)),
            "use_prev_action": bool(getattr(model, "use_prev_action", False)),
            "num_view_tokens": int(getattr(model, "num_view_tokens", 1)),
        }
    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_config": model_config,
            "state_dict": state_dict,
            "metadata": metadata,
        },
        output_path,
    )


def load_policy_from_source(model_path: str, model_kind: str):
    if model_kind == "rocket2":
        if Path(model_path).exists():
            return load_cross_view_rocket(model_path)
        if model_path.startswith("hf:"):
            return CrossViewRocket.from_pretrained(model_path.split(":", 1)[1])
        return CrossViewRocket.from_pretrained(model_path)
    if Path(model_path).exists():
        return load_rocket_policy(model_path)
    return RocketPolicy.from_pretrained(model_path)


def set_trainable_scope(model, scope: str) -> List[str]:
    for param in model.parameters():
        param.requires_grad = False

    unfrozen_modules: List[str] = []

    def unfreeze_module(module_name: str):
        module = getattr(model, module_name, None)
        if module is None:
            return
        found = False
        for param in module.parameters():
            param.requires_grad = True
            found = True
        if found:
            unfrozen_modules.append(module_name)

    if scope == "value":
        unfreeze_module("value_head")
    elif scope == "heads":
        unfreeze_module("pi_head")
        unfreeze_module("value_head")
    elif scope == "heads_last":
        unfreeze_module("pi_head")
        unfreeze_module("value_head")
        unfreeze_module("lastlayer")
        unfreeze_module("final_ln")
    elif scope == "crossview_small":
        unfreeze_module("pi_head")
        unfreeze_module("value_head")
        unfreeze_module("lastlayer")
        unfreeze_module("final_ln")
        unfreeze_module("updim_cross")
    elif scope == "full_except_vision":
        for name, param in model.named_parameters():
            if not name.startswith("view_backbone."):
                param.requires_grad = True
        unfrozen_modules.append("full_except_vision")
    else:
        raise ValueError(f"Unknown trainable scope: {scope}")
    return unfrozen_modules


def count_parameters_by_trainability(model) -> Tuple[int, int]:
    trainable = 0
    frozen = 0
    for param in model.parameters():
        count = int(param.numel())
        if param.requires_grad:
            trainable += count
        else:
            frozen += count
    return trainable, frozen


def find_fragment_paths(run_dir: Path) -> List[Path]:
    return sorted(run_dir.rglob("episode_fragment.pt"))


def compute_gae_from_fragment(fragment: Dict, denormalizer, gamma: float, gae_lambda: float) -> Tuple[torch.Tensor, torch.Tensor]:
    rewards = fragment["reward"].float()
    old_vpred = fragment["old_value"].float()
    denormalizer_device = old_vpred.device
    try:
        denormalizer_device = next(denormalizer.__self__.parameters()).device
    except Exception:
        try:
            denormalizer_device = next(denormalizer.__self__.buffers()).device
        except Exception:
            denormalizer_device = old_vpred.device
    with torch.no_grad():
        old_values = denormalizer(old_vpred.to(denormalizer_device)).detach().reshape(-1).cpu()

    terminated_flags = fragment.get("terminated")
    truncated_flags = fragment.get("truncated")
    if terminated_flags is None:
        terminated_flags = fragment.get("env_done", fragment["done"]).bool()
    else:
        terminated_flags = terminated_flags.bool()
    if truncated_flags is None:
        truncated_flags = torch.zeros_like(terminated_flags, dtype=torch.bool)
    else:
        truncated_flags = truncated_flags.bool()
    bootstrap_value_raw = fragment.get("bootstrap_value", 0.0)
    if isinstance(bootstrap_value_raw, torch.Tensor):
        bootstrap_value = float(bootstrap_value_raw.detach().reshape(-1)[0].item())
    else:
        bootstrap_value = float(bootstrap_value_raw)
    bootstrap_valid_raw = fragment.get("bootstrap_valid", False)
    if isinstance(bootstrap_valid_raw, torch.Tensor):
        bootstrap_valid = bool(bootstrap_valid_raw.detach().reshape(-1)[0].item())
    else:
        bootstrap_valid = bool(bootstrap_valid_raw)
    T = int(rewards.shape[0])
    advantages = torch.zeros(T, dtype=torch.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            if bool(terminated_flags[t]):
                next_nonterminal = 0.0
                next_value = 0.0
            elif bootstrap_valid:
                next_nonterminal = 1.0
                next_value = bootstrap_value
            else:
                next_nonterminal = 0.0
                next_value = 0.0
        else:
            if bool(terminated_flags[t]) or bool(truncated_flags[t]):
                next_nonterminal = 0.0
                next_value = 0.0
            else:
                next_nonterminal = 1.0
                next_value = float(old_values[t + 1].item())
        delta = float(rewards[t].item()) + gamma * next_value * next_nonterminal - float(old_values[t].item())
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + old_values.float()
    return advantages, returns


def load_lateral_signal(fragment_path: str, sequence_length: int) -> Optional[torch.Tensor]:
    """Per-step absolute change in lateral offset from the direct start→target line.

    Lateral offset = signed perpendicular distance of agent position from the straight line
    connecting the episode start to target_world_center (XZ plane only).

    signal[t] = |lateral_offset[t] - lateral_offset[t-1]|

    This fires at lane-commitment moments (agent moves sideways to enter a detour lane),
    funnel/door alignment steps, and obstacle recovery turns — the path-choice decisions
    that drive P-maze success. It does NOT fire heavily during the straight final approach
    after the maze structure is cleared, unlike raw distance improvement.

    Returns None if trajectory.jsonl or goal_spec.json are absent or malformed.
    """
    fpath = Path(str(fragment_path))
    traj_path = fpath.parent / "trajectory.jsonl"
    goal_path = fpath.parent / "goal_spec.json"
    if not traj_path.exists() or not goal_path.exists():
        return None
    try:
        goal = json.loads(goal_path.read_text(encoding="utf-8"))
        target = goal.get("target_world_center")
        if not (isinstance(target, (list, tuple)) and len(target) >= 3):
            return None
        tx, tz = float(target[0]), float(target[2])

        positions: List[Tuple[float, float]] = []
        with traj_path.open(encoding="utf-8") as fh:
            for line in fh:
                pos = json.loads(line).get("player_pos") or {}
                px = float(pos.get("x", tx))
                pz = float(pos.get("z", tz))
                positions.append((px, pz))

        if len(positions) < 2:
            return None

        positions = positions[:sequence_length]
        start_px, start_pz = positions[0]
        d_x, d_z = tx - start_px, tz - start_pz
        d_len = math.sqrt(d_x ** 2 + d_z ** 2)
        if d_len < 1e-6:
            return None
        # Unit vector along the direct start→target line (XZ plane)
        td_x, td_z = d_x / d_len, d_z / d_len

        # Signed lateral offset via 2D cross product: (rel × td).y = rel_x*td_z - rel_z*td_x
        lateral_offsets: List[float] = []
        for px, pz in positions:
            rel_x, rel_z = px - start_px, pz - start_pz
            lateral_offsets.append(rel_x * td_z - rel_z * td_x)

        # Signal = absolute change in lateral offset (captures both entry and exit of detours)
        signal = [0.0] + [abs(lateral_offsets[t] - lateral_offsets[t - 1]) for t in range(1, len(lateral_offsets))]
        while len(signal) < sequence_length:
            signal.append(0.0)

        return torch.tensor(signal[:sequence_length], dtype=torch.float32)
    except Exception:
        return None


def load_training_fragments(run_dir: Path, reference_model, gamma: float, gae_lambda: float, max_fragments: int = 0) -> List[Dict]:
    fragment_paths = find_fragment_paths(run_dir)
    rows: List[Dict] = []
    for fragment_path in fragment_paths:
        # PPO fragments are trusted local files produced by our own rollout code.
        # PyTorch 2.6 defaults torch.load(..., weights_only=True), which rejects
        # these dict payloads because they contain numpy-backed tensors/arrays.
        fragment = torch.load(fragment_path, map_location="cpu", weights_only=False)
        if not bool(fragment.get("trainable", True)):
            continue
        advantages, returns = compute_gae_from_fragment(
            fragment=fragment,
            denormalizer=reference_model.value_head.denormalize,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        fragment["advantages"] = advantages
        fragment["returns"] = returns
        fragment["fragment_path"] = str(fragment_path.resolve())
        seq_len = int(fragment.get("sequence_length", fragment["reward"].shape[0]))
        fragment["lateral_signal"] = load_lateral_signal(fragment["fragment_path"], seq_len)
        if _CORRIDOR_OK:
            fragment["corridor_causal_mask"] = load_corridor_causal_mask(fragment["fragment_path"], seq_len)
        else:
            fragment["corridor_causal_mask"] = None
        rows.append(fragment)
        if max_fragments > 0 and len(rows) >= max_fragments:
            break
    return rows


def aggregate_advantage_stats(fragments: List[Dict]) -> Tuple[float, float]:
    if not fragments:
        return 0.0, 1.0
    flat = torch.cat([fragment["advantages"].reshape(-1) for fragment in fragments], dim=0)
    mean = float(flat.mean().item())
    std = float(flat.std(unbiased=False).item())
    if std < 1e-8:
        std = 1.0
    return mean, std


@torch.no_grad()
def measure_logprob_identity(
    model,
    fragments: List[Dict],
    device: str,
    ppo_clip: float,
    cfg_coef: float,
    cfg_policy_mode: str,
    base_model,
) -> Dict[str, float]:
    if not fragments:
        return {
            "num_fragments_checked": 0,
            "mean_abs_logprob_diff": 0.0,
            "max_abs_logprob_diff": 0.0,
            "mean_approx_kl": 0.0,
            "mean_clip_fraction": 0.0,
        }

    total_mean_abs = 0.0
    total_approx_kl = 0.0
    total_clip_fraction = 0.0
    max_abs = 0.0
    count = 0

    for fragment in fragments:
        obs = build_model_input(fragment, device)
        action = unsqueeze_tree(to_device_tree(fragment["action"], device), 0)
        old_logprob = fragment["old_logprob"].unsqueeze(0).to(device)

        pi_logits, _ = forward_policy_outputs(
            model,
            obs,
            cfg_coef,
            cfg_policy_mode=cfg_policy_mode,
            base_model=base_model,
        )
        new_logprob = model.pi_head.logprob(action, pi_logits)
        log_ratio = new_logprob - old_logprob
        ratio = log_ratio.exp()

        total_mean_abs += float(log_ratio.abs().mean().item())
        max_abs = max(max_abs, float(log_ratio.abs().max().item()))
        total_approx_kl += float((((ratio - 1.0) - log_ratio).mean()).item())
        total_clip_fraction += float((((ratio - 1.0).abs() > ppo_clip).float().mean()).item())
        count += 1

    return {
        "num_fragments_checked": int(count),
        "mean_abs_logprob_diff": total_mean_abs / float(count),
        "max_abs_logprob_diff": max_abs,
        "mean_approx_kl": total_approx_kl / float(count),
        "mean_clip_fraction": total_clip_fraction / float(count),
    }


def build_model_input(fragment: Dict, device: str) -> Dict:
    model_input = {
        "image": fragment["image"].unsqueeze(0).to(device),
        "segment": {
            "obj_mask": fragment["obj_mask"].unsqueeze(0).to(device),
            "obj_id": fragment["obj_id"].unsqueeze(0).to(device),
        },
    }
    if "cross_view_image" in fragment:
        model_input["cross_view"] = {
            "cross_view_image": fragment["cross_view_image"].unsqueeze(0).to(device),
            "cross_view_obj_mask": fragment["cross_view_obj_mask"].unsqueeze(0).to(device),
            "cross_view_obj_id": fragment["cross_view_obj_id"].unsqueeze(0).to(device),
        }
    if "env_prev_action" in fragment:
        prev_action = normalize_prev_action_tree(fragment["env_prev_action"])
        model_input["env_prev_action"] = to_device_tree(unsqueeze_tree(prev_action, 0), device)
    return model_input


def build_unconditioned_model_input(model_input: Dict) -> Dict:
    base_input = dict(model_input)
    cross_view = model_input.get("cross_view")
    if cross_view is None:
        return base_input
    base_cross_view = dict(cross_view)
    base_cross_view["cross_view_image"] = torch.zeros_like(base_cross_view["cross_view_image"])
    base_cross_view["cross_view_obj_id"] = torch.zeros_like(base_cross_view["cross_view_obj_id"]) - 1
    base_cross_view["cross_view_obj_mask"] = torch.zeros_like(base_cross_view["cross_view_obj_mask"])
    base_input["cross_view"] = base_cross_view
    return base_input


def mix_cfg_pi_logits(cond_pi_logits: Dict[str, torch.Tensor], base_pi_logits: Dict[str, torch.Tensor], cfg_coef: float) -> Dict[str, torch.Tensor]:
    return {
        "buttons": (1.0 + cfg_coef) * cond_pi_logits["buttons"] - cfg_coef * base_pi_logits["buttons"],
        "camera": (1.0 + cfg_coef) * cond_pi_logits["camera"] - cfg_coef * base_pi_logits["camera"],
    }


def slice_time_tree(data, start_idx: int, end_idx: int, time_dim: int):
    if isinstance(data, torch.Tensor):
        slicer = [slice(None)] * data.ndim
        slicer[time_dim] = slice(start_idx, end_idx)
        return data[tuple(slicer)]
    if isinstance(data, Mapping):
        return {str(key): slice_time_tree(value, start_idx, end_idx, time_dim) for key, value in data.items()}
    if isinstance(data, list):
        return data[start_idx:end_idx]
    if isinstance(data, tuple):
        return tuple(data[start_idx:end_idx])
    return data


def build_focus_fragment_view(
    fragment: Dict,
    *,
    loss_focus_mode: str,
    loss_focus_suffix_len: int,
    loss_focus_context_len: int,
) -> Tuple[Dict, int, int]:
    sequence_length = int(fragment["sequence_length"])
    if (
        str(loss_focus_mode) == "prefix_success"
        and str(fragment.get("stop_reason", "")) == "success"
        and int(loss_focus_context_len) > 0
    ):
        prefix_len = min(sequence_length, max(1, int(loss_focus_context_len)))
        return fragment, prefix_len, sequence_length
    if (
        str(loss_focus_mode) != "suffix_success"
        or str(fragment.get("stop_reason", "")) != "success"
        or int(loss_focus_suffix_len) <= 0
    ):
        return fragment, sequence_length, sequence_length

    suffix_len = max(1, int(loss_focus_suffix_len))
    context_len = max(0, int(loss_focus_context_len))
    window_start = max(0, sequence_length - (context_len + suffix_len))
    window_end = sequence_length
    if window_start <= 0:
        focus_start = max(0, sequence_length - suffix_len)
        return fragment, focus_start, sequence_length

    window_fragment = dict(fragment)
    time_dim0_keys = [
        "image",
        "obj_mask",
        "obj_id",
        "action",
        "old_logprob",
        "old_value",
        "reward",
        "done",
        "env_done",
        "first",
        "subtask_index",
        "interaction_type",
        "prompt_text",
        "segment_area",
        "cross_view_image",
        "cross_view_obj_mask",
        "cross_view_obj_id",
        "env_prev_action",
        "advantages",
        "returns",
        "lateral_signal",
        "corridor_causal_mask",
    ]
    for key in time_dim0_keys:
        if key in fragment:
            window_fragment[key] = slice_time_tree(fragment[key], window_start, window_end, time_dim=0)
    if "ref_pi_logits" in fragment:
        window_fragment["ref_pi_logits"] = slice_time_tree(fragment["ref_pi_logits"], window_start, window_end, time_dim=1)
    window_fragment["sequence_length"] = int(window_end - window_start)
    focus_start = max(0, int(window_fragment["sequence_length"]) - suffix_len)
    return window_fragment, focus_start, int(window_fragment["sequence_length"])


def build_loss_weights(
    sequence_length: int,
    focus_start: int,
    focus_weight: float,
    device: str,
    loss_focus_mode: str = "suffix_success",
    advantages: torch.Tensor = None,
    lateral_signal: torch.Tensor = None,
    corridor_causal_mask: torch.Tensor = None,
    loss_focus_top_k: int = 32,
) -> torch.Tensor:
    weights = torch.ones((1, int(sequence_length)), dtype=torch.float32, device=device)
    focus_weight = float(max(1.0, focus_weight))
    k = min(max(1, int(loss_focus_top_k)), int(sequence_length))

    if loss_focus_mode == "advantage_top_k" and advantages is not None:
        signal = advantages.reshape(-1).to(device)
        top_k_indices = torch.topk(signal, k=k, largest=True).indices
        weights[0, top_k_indices] = focus_weight
    elif loss_focus_mode == "prefix_success":
        prefix_len = min(max(0, int(focus_start)), int(sequence_length))
        if prefix_len > 0:
            weights[:, :prefix_len] = focus_weight
    elif loss_focus_mode == "lateral_top_k" and lateral_signal is not None:
        signal = lateral_signal.reshape(-1).to(device)
        top_k_indices = torch.topk(signal, k=k, largest=True).indices
        weights[0, top_k_indices] = focus_weight
    elif loss_focus_mode == "corridor_commit" and corridor_causal_mask is not None:
        # Binary causal mask: True = in commit-to-approach window; apply focus_weight there
        mask = corridor_causal_mask.reshape(-1).float().to(device)[:int(sequence_length)]
        if mask.shape[0] < int(sequence_length):
            mask = torch.nn.functional.pad(mask, (0, int(sequence_length) - mask.shape[0]))
        weights[0] = 1.0 + (focus_weight - 1.0) * mask
    elif focus_start < int(sequence_length):
        weights[:, int(focus_start):] = focus_weight
    return weights


def weighted_time_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    values = values.float()
    expanded_weights = weights
    while expanded_weights.ndim < values.ndim:
        expanded_weights = expanded_weights.unsqueeze(-1)
    weighted_sum = (values * expanded_weights).sum()
    normalizer = weights.sum().clamp_min(1e-8)
    return weighted_sum / normalizer


def forward_policy_outputs(
    model,
    obs: Dict,
    cfg_coef: float,
    *,
    cfg_policy_mode: str = "full",
    base_model=None,
):
    cond_latents, _ = model(obs, model.initial_state(1))
    pi_logits = cond_latents["pi_logits"]
    if float(cfg_coef) > 0.0 and "cross_view" in obs:
        base_obs = build_unconditioned_model_input(obs)
        if str(cfg_policy_mode or "full") == "frozen_base":
            effective_base_model = base_model if base_model is not None else model
            with torch.no_grad():
                base_latents, _ = effective_base_model(base_obs, effective_base_model.initial_state(1))
        else:
            base_latents, _ = model(base_obs, model.initial_state(1))
        pi_logits = mix_cfg_pi_logits(cond_latents["pi_logits"], base_latents["pi_logits"], float(cfg_coef))
    return pi_logits, cond_latents["vpred"]


def compute_fragment_loss(
    model,
    ref_model,
    base_model,
    fragment: Dict,
    device: str,
    cfg_coef: float,
    cfg_policy_mode: str,
    ppo_clip: float,
    vf_coef: float,
    policy_coef: float,
    entropy_coef: float,
    kl_coef: float,
    clip_vloss: bool,
    normalize_advantage: bool,
    advantage_mean: float,
    advantage_std: float,
    loss_focus_mode: str,
    loss_focus_suffix_len: int,
    loss_focus_context_len: int,
    loss_focus_weight: float,
    loss_focus_top_k: int = 32,
) -> Dict[str, torch.Tensor]:
    fragment_view, focus_start, sequence_length = build_focus_fragment_view(
        fragment,
        loss_focus_mode=loss_focus_mode,
        loss_focus_suffix_len=loss_focus_suffix_len,
        loss_focus_context_len=loss_focus_context_len,
    )
    obs = build_model_input(fragment_view, device)
    action = unsqueeze_tree(to_device_tree(fragment_view["action"], device), 0)
    old_logprob = fragment_view["old_logprob"].unsqueeze(0).to(device)
    old_vpred = fragment_view["old_value"].unsqueeze(0).unsqueeze(-1).to(device)
    returns = fragment_view["returns"].unsqueeze(0).unsqueeze(-1).to(device)
    advantages = fragment_view["advantages"].unsqueeze(0).to(device)
    raw_advantages = fragment_view["advantages"]
    if normalize_advantage:
        advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)
    is_successful = str(fragment.get("stop_reason", "")) == "success"
    adv_for_weights = raw_advantages if (loss_focus_mode == "advantage_top_k" and is_successful) else None
    lat_for_weights = fragment_view.get("lateral_signal") if (loss_focus_mode == "lateral_top_k" and is_successful) else None
    corridor_mask_for_weights = fragment_view.get("corridor_causal_mask") if (loss_focus_mode == "corridor_commit" and is_successful) else None
    loss_weights = build_loss_weights(
        sequence_length, focus_start, loss_focus_weight, device,
        loss_focus_mode=loss_focus_mode,
        advantages=adv_for_weights,
        lateral_signal=lat_for_weights,
        corridor_causal_mask=corridor_mask_for_weights,
        loss_focus_top_k=loss_focus_top_k,
    )

    pi_logits, vpred = forward_policy_outputs(
        model,
        obs,
        cfg_coef,
        cfg_policy_mode=cfg_policy_mode,
        base_model=base_model,
    )
    vpred = vpred.reshape(1, -1, 1)

    new_logprob = model.pi_head.logprob(action, pi_logits)
    log_ratio = new_logprob - old_logprob
    ratio = log_ratio.exp()

    loss_policy_1 = -advantages * ratio
    loss_policy_2 = -advantages * torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
    policy_loss = weighted_time_mean(torch.max(loss_policy_1, loss_policy_2), loss_weights)

    vf_loss_unclipped = 0.5 * model.value_head.loss(vpred, returns, reduction="none")
    if clip_vloss:
        vpred_clipped = old_vpred + torch.clamp(vpred - old_vpred, -ppo_clip, ppo_clip)
        vf_loss_clipped = 0.5 * model.value_head.loss(vpred_clipped, returns, reduction="none")
        value_loss = weighted_time_mean(torch.max(vf_loss_unclipped, vf_loss_clipped), loss_weights)
    else:
        value_loss = weighted_time_mean(vf_loss_unclipped, loss_weights)

    entropy_bonus = weighted_time_mean(model.pi_head.entropy(pi_logits), loss_weights)

    if kl_coef > 0.0:
        ref_pi_logits = None
        if "ref_pi_logits" in fragment_view:
            ref_pi_logits = to_device_tree(fragment_view["ref_pi_logits"], device)
        elif ref_model is not None:
            with torch.no_grad():
                ref_pi_logits, _ = forward_policy_outputs(
                    ref_model,
                    obs,
                    cfg_coef,
                    cfg_policy_mode=cfg_policy_mode,
                    base_model=base_model,
                )
        if ref_pi_logits is not None:
            epsilon = 1e-8
            kl_divergence = model.pi_head.kl_divergence(
                {key: (pi_logits[key] + epsilon) for key in pi_logits},
                {key: (ref_pi_logits[key] + epsilon) for key in ref_pi_logits},
            )
            kl_divergence = weighted_time_mean(kl_divergence, loss_weights)
        else:
            kl_divergence = torch.tensor(0.0, device=device)
    else:
        kl_divergence = torch.tensor(0.0, device=device)

    total_loss = (
        policy_coef * policy_loss
        + vf_coef * value_loss
        - entropy_coef * entropy_bonus
        + kl_coef * kl_divergence
    )

    approx_kl = weighted_time_mean(((ratio - 1.0) - log_ratio), loss_weights)
    clip_fraction = weighted_time_mean(((ratio - 1.0).abs() > ppo_clip).float(), loss_weights)
    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy_bonus": entropy_bonus.detach(),
        "kl_divergence": kl_divergence.detach(),
        "approx_kl": approx_kl.detach(),
        "clip_fraction": clip_fraction.detach(),
        "sequence_length_used": torch.tensor(int(sequence_length), dtype=torch.float32, device=device),
        "focus_start": torch.tensor(int(focus_start), dtype=torch.float32, device=device),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_base_ref_model_path = resolve_cfg_base_ref_model_path(
        model_path=args.model_path,
        cfg_policy_mode=args.cfg_policy_mode,
        cfg_base_ref_model_path=args.cfg_base_ref_model_path,
    )
    kl_anchor_model_path = resolve_kl_anchor_model_path(
        model_path=args.model_path,
        kl_anchor_model_path=args.kl_anchor_model_path,
    )

    value_reference_model = load_policy_from_source(args.model_path, args.model_kind).to(device)
    value_reference_model.eval()
    base_ref_model = None
    if float(args.cfg_coef) > 0.0 and str(args.cfg_policy_mode) == "frozen_base":
        if str(cfg_base_ref_model_path) == str(args.model_path):
            base_ref_model = value_reference_model
        else:
            base_ref_model = load_policy_from_source(cfg_base_ref_model_path, args.model_kind).to(device)
            base_ref_model.eval()
        for param in base_ref_model.parameters():
            param.requires_grad = False

    fragments = load_training_fragments(
        run_dir=run_dir,
        reference_model=value_reference_model,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_fragments=args.max_fragments,
    )
    if not fragments:
        raise RuntimeError(f"No trainable episode_fragment.pt files found under {run_dir}")

    kl_anchor_model = None
    if args.kl_coef > 0.0:
        if str(kl_anchor_model_path) == str(args.model_path):
            kl_anchor_model = value_reference_model
        elif base_ref_model is not None and str(kl_anchor_model_path) == str(cfg_base_ref_model_path):
            kl_anchor_model = base_ref_model
        else:
            kl_anchor_model = load_policy_from_source(kl_anchor_model_path, args.model_kind).to(device)
            kl_anchor_model.eval()
        for fragment in fragments:
            obs = build_model_input(fragment, device)
            with torch.no_grad():
                ref_pi_logits, _ = forward_policy_outputs(
                    kl_anchor_model,
                    obs,
                    args.cfg_coef,
                    cfg_policy_mode=args.cfg_policy_mode,
                    base_model=base_ref_model,
                )
            fragment["ref_pi_logits"] = to_device_tree(ref_pi_logits, "cpu")

    if kl_anchor_model not in {None, value_reference_model, base_ref_model}:
        del kl_anchor_model
    if base_ref_model is not value_reference_model:
        del value_reference_model
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = load_policy_from_source(args.model_path, args.model_kind).to(device)
    model.eval()
    unfrozen_modules = set_trainable_scope(model, args.trainable_scope)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected for PPO update.")
    trainable_parameter_count, frozen_parameter_count = count_parameters_by_trainability(model)
    sanity_preupdate = measure_logprob_identity(
        model=model,
        fragments=fragments,
        device=device,
        ppo_clip=args.ppo_clip,
        cfg_coef=args.cfg_coef,
        cfg_policy_mode=args.cfg_policy_mode,
        base_model=base_ref_model,
    )
    print(json.dumps({"event": "sanity_preupdate", **sanity_preupdate}, ensure_ascii=False))
    ref_model = None
    if ref_model is not None:
        ref_model.eval()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    advantage_mean, advantage_std = aggregate_advantage_stats(fragments)
    history: List[Dict] = []
    update_fragment_batch_size = max(1, int(args.update_fragment_batch_size))

    for epoch in range(args.epochs):
        random.shuffle(fragments)
        epoch_stats = {
            "epoch": epoch + 1,
            "num_fragments": len(fragments),
            "mean_total_loss": 0.0,
            "mean_policy_loss": 0.0,
            "mean_value_loss": 0.0,
            "mean_entropy_bonus": 0.0,
            "mean_kl_divergence": 0.0,
            "mean_approx_kl": 0.0,
            "mean_clip_fraction": 0.0,
        }

        for batch_start in range(0, len(fragments), update_fragment_batch_size):
            fragment_batch = fragments[batch_start : batch_start + update_fragment_batch_size]
            batch_size = max(1, len(fragment_batch))
            optimizer.zero_grad(set_to_none=True)
            for fragment in fragment_batch:
                loss_dict = compute_fragment_loss(
                    model=model,
                    ref_model=ref_model,
                    base_model=base_ref_model,
                    fragment=fragment,
                    device=device,
                    cfg_coef=args.cfg_coef,
                    cfg_policy_mode=args.cfg_policy_mode,
                    ppo_clip=args.ppo_clip,
                    vf_coef=args.vf_coef,
                    policy_coef=args.policy_coef,
                    entropy_coef=args.entropy_coef,
                    kl_coef=args.kl_coef,
                    clip_vloss=args.clip_vloss,
                    normalize_advantage=args.normalize_advantage,
                    advantage_mean=advantage_mean,
                    advantage_std=advantage_std,
                    loss_focus_mode=args.loss_focus_mode,
                    loss_focus_suffix_len=args.loss_focus_suffix_len,
                    loss_focus_context_len=args.loss_focus_context_len,
                    loss_focus_weight=args.loss_focus_weight,
                    loss_focus_top_k=args.loss_focus_top_k,
                )
                (loss_dict["total_loss"] / float(batch_size)).backward()
                for key in list(epoch_stats.keys())[2:]:
                    stat_key = key.replace("mean_", "")
                    epoch_stats[key] += float(loss_dict[stat_key].item())
            clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()

        denom = float(len(fragments))
        for key in list(epoch_stats.keys())[2:]:
            epoch_stats[key] /= denom
        history.append(epoch_stats)
        print(json.dumps(epoch_stats, ensure_ascii=False))

    checkpoint_path = output_dir / "model.pt"
    save_rocket_checkpoint(
        model,
        checkpoint_path,
        metadata={
            "source_run_dir": str(run_dir.resolve()),
            "source_model_path": args.model_path,
            "kl_anchor_model_path": str(kl_anchor_model_path),
            "model_kind": args.model_kind,
            "cfg_coef": float(args.cfg_coef),
            "cfg_policy_mode": str(args.cfg_policy_mode),
            "cfg_base_ref_model_path": str(cfg_base_ref_model_path),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "max_grad_norm": float(args.max_grad_norm),
            "update_fragment_batch_size": int(update_fragment_batch_size),
            "loss_focus_mode": str(args.loss_focus_mode),
            "loss_focus_suffix_len": int(args.loss_focus_suffix_len),
            "loss_focus_context_len": int(args.loss_focus_context_len),
            "loss_focus_weight": float(args.loss_focus_weight),
            "loss_focus_top_k": int(args.loss_focus_top_k),
            "fragments": len(fragments),
            "trainable_scope": args.trainable_scope,
            "unfrozen_modules": unfrozen_modules,
            "trainable_parameter_count": trainable_parameter_count,
            "frozen_parameter_count": frozen_parameter_count,
            "sanity_preupdate": sanity_preupdate,
            "history": history,
        },
    )
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "sanity_preupdate.json").write_text(
        json.dumps(sanity_preupdate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "train_metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "model_path": args.model_path,
                "kl_anchor_model_path": str(kl_anchor_model_path),
                "model_kind": args.model_kind,
                "cfg_coef": float(args.cfg_coef),
                "cfg_policy_mode": str(args.cfg_policy_mode),
                "cfg_base_ref_model_path": str(cfg_base_ref_model_path),
                "output_checkpoint": str(checkpoint_path.resolve()),
                "device": device,
                "epochs": int(args.epochs),
                "learning_rate": float(args.learning_rate),
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "max_grad_norm": float(args.max_grad_norm),
                "update_fragment_batch_size": int(update_fragment_batch_size),
                "loss_focus_mode": str(args.loss_focus_mode),
                "loss_focus_suffix_len": int(args.loss_focus_suffix_len),
                "loss_focus_context_len": int(args.loss_focus_context_len),
                "loss_focus_weight": float(args.loss_focus_weight),
                "fragments": len(fragments),
                "advantage_mean": advantage_mean,
                "advantage_std": advantage_std,
                "trainable_scope": args.trainable_scope,
                "unfrozen_modules": unfrozen_modules,
                "trainable_parameter_count": trainable_parameter_count,
                "frozen_parameter_count": frozen_parameter_count,
                "sanity_preupdate": sanity_preupdate,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"saved updated model to {checkpoint_path}")


if __name__ == "__main__":
    main()
