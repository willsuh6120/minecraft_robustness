import os
import cv2
import time
import json
import socket
import signal
import subprocess
import gradio as gr
import torch
import numpy as np
import yaml
import copy
from pathlib import Path
from PIL import Image, ImageDraw

from minestudio.models import RocketPolicy, load_rocket_policy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import load_callbacks_from_config

from sam2.build_sam import build_sam2_camera_predictor
import requests
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from minestudio.tutorials.inference.evaluate_rocket.pointing_common import (
    build_localization_prompt,
    encode_image_base64,
    parse_coordinates,
    to_pixel_coordinates,
)

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), 
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 255, 255), (0, 0, 0), (128, 128, 128),
    (128, 0, 0), (128, 128, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (0, 0, 128),
]

SEGMENT_MAPPING = {
    "Hunt": 0, "Use": 3, "Mine": 2, "Interact": 3, "Craft": 4, "Switch": 5, "Approach": 6, "None": -1
}


# NOOP_ACTION = {
#     "ESC": 0,
#     "back": 0,
#     "drop": 0,
#     "forward": 0,
#     "hotbar.1": 0,
#     "hotbar.2": 0,
#     "hotbar.3": 0,
#     "hotbar.4": 0,
#     "hotbar.5": 0,
#     "hotbar.6": 0,
#     "hotbar.7": 0,
#     "hotbar.8": 0,
#     "hotbar.9": 0,
#     "inventory": 0,
#     "jump": 0,
#     "left": 0,
#     "right": 0,
#     "sneak": 0,
#     "sprint": 0,
#     "swapHands": 0,
#     "camera": np.array([0, 0]),
#     "attack": 0,
#     "use": 0,
#     "pickItem": 0,
# }

class Session:
    
    def __init__(self, model_path: str, sam_path: str, name_file_mapping: dict):
        start_image = np.zeros((360, 640, 3), dtype=np.uint8)
        self.current_image = np.array(start_image)
        self.model_path = model_path
        self.sam_path = sam_path
        self.name_file_mapping = name_file_mapping
        self.clear_points()
        
        self.sam_choice = 'base'
        self.load_sam()
        
        self.tracking_flag = True
        self.points = []
        self.points_label = []
        self.able_to_track = False
        self.segment_type = "Approach"
        self.obj_mask = np.zeros((224, 224), dtype=np.uint8)
        self.calling_rocket = False
        self.num_steps = 0
        self.last_action_summary = "none"
        self.plan_count = 0
        self.last_plan = {}
        self.plan_history = []
        self.world_seed = 0
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        self.last_policy_action = None
        self.last_policy_logprob = None
        self.last_policy_value = None
        self.last_policy_value_raw = None
        self.last_memory_in = None
        self.last_model_input = None
        self.last_segment_area = 0
        self.last_segment_raw_area = 0
        self.last_segment_fallback_area = 0
        self.last_segment_used_fallback = False
        self.last_segment_point_count = 0
    
    def clear_points(self):
        self.points = []
        self.points_label = []
    
    def clear_obj_mask(self):
        self.obj_mask = np.zeros((224, 224), dtype=np.uint8)
    
    def clear_agent_memory(self, reset_counters: bool = True):
        if reset_counters:
            self.num_steps = 0
            self.last_action_summary = "none"
        if hasattr(self, "agent"):
            self.state = self.agent.initial_state()
        self.last_policy_action = None
        self.last_policy_logprob = None
        self.last_policy_value = None
        self.last_policy_value_raw = None
        self.last_memory_in = None
        self.last_model_input = None

    def reset_planner_state(self):
        self.plan_count = 0
        self.last_plan = {}
        self.plan_history = []

    def summarize_agent_action(self, action):
        try:
            action_copy = copy.deepcopy(action)
            env_action = self.env.agent_action_to_env_action(action_copy)
            active_buttons = [
                key for key, value in env_action.items()
                if key != "camera" and value == 1
            ]
            camera = env_action.get("camera", np.array([0, 0]))
            camera = np.asarray(camera).tolist()
            if not active_buttons:
                active_buttons = ["noop"]
            return f"buttons={'+'.join(active_buttons)}, camera={camera}"
        except Exception as exc:
            return f"unavailable ({exc})"

    def _detach_action_for_logging(self, action):
        if isinstance(action, torch.Tensor):
            return action.detach().cpu().clone()
        if isinstance(action, dict):
            return {key: self._detach_action_for_logging(value) for key, value in action.items()}
        if isinstance(action, list):
            return [self._detach_action_for_logging(value) for value in action]
        if isinstance(action, tuple):
            return tuple(self._detach_action_for_logging(value) for value in action)
        return action

    def _detach_state_for_logging(self, state):
        if state is None:
            return None
        if isinstance(state, torch.Tensor):
            return state.detach().cpu().clone()
        if isinstance(state, list):
            return [self._detach_state_for_logging(value) for value in state]
        if isinstance(state, tuple):
            return tuple(self._detach_state_for_logging(value) for value in state)
        if isinstance(state, dict):
            return {key: self._detach_state_for_logging(value) for key, value in state.items()}
        return state

    def load_sam(self):
        
        ckpt_mapping = {
            'large': [os.path.join(self.sam_path, "sam2_hiera_large.pt"), "sam2_hiera_l.yaml"],
            'base': [os.path.join(self.sam_path, "sam2_hiera_base_plus.pt"), "sam2_hiera_b+.yaml"],
            'small': [os.path.join(self.sam_path, "sam2_hiera_small.pt"), "sam2_hiera_s.yaml"], 
            'tiny': [os.path.join(self.sam_path, "sam2_hiera_tiny.pt"), "sam2_hiera_t.yaml"]
        }
        sam_ckpt, model_cfg = ckpt_mapping[self.sam_choice]
        # first realease the old predictor
        if hasattr(self, "predictor"):
            del self.predictor
        self.predictor = build_sam2_camera_predictor(model_cfg, sam_ckpt)
        print(f"Successfully loaded SAM2 from {sam_ckpt}")
        self.able_to_track = False

    def segment(self):
        positive_points = []
        if len(self.points) > 0 and len(self.points_label) > 0:
            self.able_to_track = True
            positive_points = [
                (int(point[0]), int(point[1]))
                for point, label in zip(self.points, self.points_label)
                if int(label) > 0
            ]
            self.last_segment_point_count = len(positive_points)
            self.predictor.load_first_frame(self.current_image)
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0, 
                obj_id=0,
                points=self.points,
                labels=self.points_label,
            )
        else:
            self.last_segment_point_count = 0
            out_obj_ids, out_mask_logits = self.predictor.track(self.current_image)
        self.obj_mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy() # 360, 640
        raw_mask_uint8 = np.asarray(self.obj_mask, dtype=np.uint8)
        self.last_segment_raw_area = int(raw_mask_uint8.sum())
        self.last_segment_fallback_area = 0
        self.last_segment_used_fallback = False
        if positive_points:
            if int(raw_mask_uint8.sum()) < 300:
                fallback_mask = np.zeros_like(raw_mask_uint8, dtype=np.uint8)
                for x, y in positive_points:
                    cv2.circle(fallback_mask, (x, y), 18, 1, -1)
                self.last_segment_fallback_area = int(fallback_mask.sum())
                self.last_segment_used_fallback = True
                self.obj_mask = np.logical_or(raw_mask_uint8 > 0, fallback_mask > 0)
            else:
                self.obj_mask = raw_mask_uint8 > 0
        self.clear_points()
        self.last_segment_area = int(np.asarray(self.obj_mask, dtype=np.uint8).sum())
        return self.obj_mask
    
    def reset(self, env_name: str, world_seed=None, warmup_noop_steps: int = 30):
        self.image_history = []
        if world_seed not in (None, ""):
            self.world_seed = int(world_seed)
        if hasattr(self, "env"):
            self.env.close()
        with open(self.name_file_mapping[env_name], "r") as f:
            task_config = yaml.safe_load(f) or {}
        # ROCKET-1 does not consume reference videos; strip GROOT-style demo deps.
        task_config.pop("reference_video", None)
        self.env = MinecraftSim(
            seed=self.world_seed,
            preferred_spawn_biome="plains", 
            callbacks=load_callbacks_from_config(task_config),
        )
        self.obs, self.info = self.env.reset()
        self.last_reset_warmup_steps = int(max(0, warmup_noop_steps))
        for i in range(self.last_reset_warmup_steps): #! better init
            time.sleep(0.1)
            noop_action = self.env.noop_action()
            self.obs, self.reward, terminated, truncated, self.info = self.env.step(noop_action)
        
        self.reward = 0
        if os.path.exists(self.model_path):
            agent = load_rocket_policy(self.model_path)
        else:
            agent = RocketPolicy.from_pretrained(self.model_path)
        self.agent = agent.to("cuda")
        self.agent.eval()
        self.clear_agent_memory()
        self.reset_planner_state()
        self.current_image = self.info["pov"]
        self.image_history.append(self.current_image)
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        return self.current_image
    
    def apply_mask(self):
        image = self.current_image.copy()
        color = COLORS[ SEGMENT_MAPPING[self.segment_type] ]
        color = np.array(color).reshape(1, 1, 3)[:, :, ::-1]
        obj_mask = (self.obj_mask[..., None] * color).astype(np.uint8)
        image = cv2.addWeighted(image, 1.0, obj_mask, 0.5, 0.0)
        return image
    
    def step(self, input_action=None):
        if input_action is not None:
            action = input_action
            self.last_policy_action = None
            self.last_policy_logprob = None
            self.last_policy_value = None
            self.last_policy_value_raw = None
            self.last_memory_in = None
            self.last_model_input = None
        else:
            obj_id = torch.tensor( SEGMENT_MAPPING[self.segment_type] )
            obj_mask = self.obj_mask.astype(np.uint8)
            obj_mask = cv2.resize(obj_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            obj_mask = torch.tensor(obj_mask, dtype=torch.uint8)
            obs = {
                'image': self.obs['image'], 
                'segment': {
                    'obj_id': obj_id, 
                    'obj_mask': obj_mask, 
                }
            }
            self.last_model_input = {
                "image": np.array(self.obs["image"], copy=True),
                "obj_mask": obj_mask.detach().cpu().clone(),
                "obj_id": int(obj_id.item()),
            }
            self.last_memory_in = self._detach_state_for_logging(self.state)
            action, self.state = self.agent.get_action(obs, self.state, input_shape="*")
            self.last_policy_action = self._detach_action_for_logging(action)
            try:
                self.last_policy_logprob = float(self.agent.pi_head.logprob(action, self.agent.cache_latents["pi_logits"]).item())
            except Exception:
                self.last_policy_logprob = None
            try:
                cached_vpred = self.agent.cache_latents["vpred"]
                value_head_owner = getattr(self.agent, "model", self.agent)
                value_head = getattr(value_head_owner, "value_head", None)
                self.last_policy_value = float(cached_vpred.reshape(-1)[0].item())
                if value_head is not None:
                    with torch.no_grad():
                        raw_vpred = value_head.denormalize(cached_vpred)
                    self.last_policy_value_raw = float(raw_vpred.reshape(-1)[0].detach().cpu().item())
                else:
                    self.last_policy_value_raw = None
            except Exception:
                self.last_policy_value = None
                self.last_policy_value_raw = None

        self.last_action_summary = self.summarize_agent_action(action)
        self.obs, self.reward, terminated, truncated, self.info = self.env.step(action)
        self.last_reward = float(self.reward)
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)
        self.current_image = self.info["pov"]
        image = self.current_image
        if self.able_to_track and self.tracking_flag:
            self.segment()
            self.last_segment_area = int(np.asarray(self.obj_mask, dtype=np.uint8).sum())
            image = self.apply_mask()
        else:
            self.last_segment_area = int(np.asarray(self.obj_mask, dtype=np.uint8).sum())
            time.sleep(0.01)
        self.num_steps += 1
        self.image_history.append(image)
        return image

    def estimate_current_policy_value(self):
        if not hasattr(self, "agent") or not hasattr(self, "obs"):
            return None
        try:
            obj_id = torch.tensor(SEGMENT_MAPPING[self.segment_type])
            obj_mask = self.obj_mask.astype(np.uint8)
            obj_mask = cv2.resize(obj_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            obj_mask = torch.tensor(obj_mask, dtype=torch.uint8)
            obs = {
                'image': self.obs['image'],
                'segment': {
                    'obj_id': obj_id,
                    'obj_mask': obj_mask,
                }
            }
            cached_latents = getattr(self.agent, "cache_latents", None)
            cached_vpred = getattr(self.agent, "vpred", None)
            try:
                self.agent.get_action(obs, self.state, deterministic=True, input_shape="*")
                current_vpred = self.agent.cache_latents["vpred"]
                value_head_owner = getattr(self.agent, "model", self.agent)
                value_head = getattr(value_head_owner, "value_head", None)
                value_norm = float(current_vpred.reshape(-1)[0].item())
                if value_head is None:
                    return {
                        "value_norm": value_norm,
                        "value_raw": value_norm,
                    }
                with torch.no_grad():
                    value_raw = value_head.denormalize(current_vpred)
                return {
                    "value_norm": value_norm,
                    "value_raw": float(value_raw.reshape(-1)[0].detach().cpu().item()),
                }
            finally:
                if cached_latents is not None:
                    self.agent.cache_latents = cached_latents
                if cached_vpred is not None and hasattr(self.agent, "vpred"):
                    self.agent.vpred = cached_vpred
        except Exception:
            return None

    def close(self):
        if hasattr(self, "env"):
            self.env.close()
        if hasattr(self, "agent"):
            del self.agent
        if hasattr(self, "predictor"):
            del self.predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class Pointer:
    '''
    Pointer is a model based on molmo, which can output point location of the object in the image.
    '''
    def __init__(
        self,
        model_id,
        model_url: str = "http://127.0.0.1:9162/v1",
        api_key: str = "EMPTY",
        managed_env_name: str = "molmo-pointing",
        managed_host: str = "127.0.0.1",
        managed_port: int = 0,
        molmo_loader: str = "manual",
        molmo_torch_dtype: str = "float16",
        molmo_autocast_dtype: str = "float16",
    ):
        self.model_id = model_id
        self.model_url = model_url
        self.api_key = api_key
        self.client = None
        self.managed_env_name = managed_env_name
        self.managed_host = managed_host
        self.managed_port = int(managed_port)
        self.molmo_loader = molmo_loader
        self.molmo_torch_dtype = molmo_torch_dtype
        self.molmo_autocast_dtype = molmo_autocast_dtype
        self._managed_server_process = None
        self._managed_server_log_path = None
        self._managed_server_base_url = None
        self._managed_server_log_handle = None

    def _is_local_hf(self):
        return self.model_url is None or self.model_url == "huggingface"

    def _is_managed_pointing_server(self):
        return self.model_url == "managed-local"

    def _pointing_server_base_url(self):
        if self._is_managed_pointing_server():
            if not self._managed_server_base_url:
                raise RuntimeError("Managed pointing server has not been started yet.")
            return self._managed_server_base_url.rstrip("/")
        if self.model_url.startswith("pointing://"):
            return self.model_url[len("pointing://"):].rstrip("/")
        if self.model_url.startswith("server://"):
            return self.model_url[len("server://"):].rstrip("/")
        return self.model_url.rstrip("/")

    def _is_pointing_server(self):
        if self._is_local_hf():
            return False
        if self._is_managed_pointing_server():
            return True
        if self.model_url.startswith(("pointing://", "server://")):
            return True
        return not self.model_url.rstrip("/").endswith("/v1")

    def _pick_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.managed_host, 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return int(sock.getsockname()[1])

    def _start_managed_pointing_server(self):
        if self._managed_server_process is not None:
            return
        port = self.managed_port if self.managed_port > 0 else self._pick_free_port()
        self.managed_port = int(port)
        self._managed_server_base_url = f"http://{self.managed_host}:{self.managed_port}"
        repo_root = Path(__file__).resolve().parents[4]
        log_dir = Path("/tmp/minestudio_pointing_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._managed_server_log_path = str(log_dir / f"pointing_server_{self.managed_port}.log")
        log_handle = open(self._managed_server_log_path, "w", encoding="utf-8")
        self._managed_server_log_handle = log_handle
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        cmd = [
            "conda",
            "run",
            "-n",
            self.managed_env_name,
            "python",
            "-m",
            "minestudio.tutorials.inference.evaluate_rocket.pointing_server",
            "--host",
            self.managed_host,
            "--port",
            str(self.managed_port),
            "--backend",
            "molmo-local",
            "--model-id",
            self.model_id,
            "--molmo-loader",
            self.molmo_loader,
            "--molmo-torch-dtype",
            self.molmo_torch_dtype,
            "--molmo-autocast-dtype",
            self.molmo_autocast_dtype,
        ]
        self._managed_server_process = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        healthcheck_url = f"{self._managed_server_base_url}/health"
        deadline = time.time() + 180
        while time.time() < deadline:
            if self._managed_server_process.poll() is not None:
                raise RuntimeError(
                    "Managed pointing server exited before becoming ready. "
                    f"See log: {self._managed_server_log_path}"
                )
            try:
                response = requests.get(healthcheck_url, timeout=2)
                response.raise_for_status()
                print(f"Started managed pointing server: {healthcheck_url}")
                return
            except requests.RequestException:
                time.sleep(1)
        raise RuntimeError(
            "Timed out waiting for managed pointing server to become ready. "
            f"See log: {self._managed_server_log_path}"
        )

    def post_init(self):
        if self._is_local_hf():
            self.load_molmo_from_hf(self.model_id)
        elif self._is_managed_pointing_server():
            self._start_managed_pointing_server()
        elif self._is_pointing_server():
            healthcheck_url = f"{self._pointing_server_base_url()}/health"
            try:
                response = requests.get(healthcheck_url, timeout=5)
                response.raise_for_status()
                print(f"Connected to pointing server: {healthcheck_url}")
            except requests.RequestException as exc:
                raise RuntimeError(f"Failed to reach pointing server at {healthcheck_url}") from exc
        else:
            if OpenAI is None:
                raise ImportError(
                    "openai package is required when using an OpenAI-compatible Molmo endpoint. "
                    "Use --molmo-url huggingface to run Molmo locally without it, "
                    "or point --molmo-url to a custom pointing server."
                )
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.model_url,
            )
            models = client.models.list()
            print(models)
            model = models.data[0].id
            assert model == self.model_id, f"Model {self.model_id} not found in current model_url {self.model_url}"
            print(f"Using model {self.model_id} based on url {self.model_url}")
            self.client = client

    def _resolve_local_snapshot_path(self, model_id: str) -> str:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        model_dir = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
        ref_file = model_dir / "refs" / "main"
        if ref_file.exists():
            revision = ref_file.read_text().strip()
            snapshot_dir = model_dir / "snapshots" / revision
            if snapshot_dir.exists():
                return str(snapshot_dir)
        return model_id

    def load_molmo_from_hf(self, model_id):
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        local_model_path = self._resolve_local_snapshot_path(model_id)
        print(f"Loading Molmo from local path: {local_model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.GenerationConfig = GenerationConfig
        self.processor = AutoProcessor.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=self.model_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def _generate_with_openai_client(self, prompt: str, image) -> str:
        if self.client is None:
            print("Initializing OpenAI client")
            self.post_init()
        image_base64 = encode_image_base64(image)
        chat_completion_from_base64 = self.client.chat.completions.create(
            messages=[{
                "role":"user",
                "content": [{"type": "text", "text": prompt},
                        {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}],
            model=self.model_id,
        )
        return chat_completion_from_base64.choices[0].message.content

    def _locate_with_pointing_server(self, prompt: str, image):
        if not getattr(self, "_pointing_server_ready", False):
            print("Initializing pointing server client")
            self.post_init()
            self._pointing_server_ready = True
        response = requests.post(
            f"{self._pointing_server_base_url()}/locate",
            json={
                "prompt": prompt,
                "image_base64": encode_image_base64(image),
                "model_id": self.model_id,
            },
            timeout=180,
        )
        if not response.ok:
            try:
                payload = response.json()
                error_msg = payload.get("error", response.text)
                traceback_msg = payload.get("traceback", "")
                details = f"{error_msg}\n{traceback_msg}".strip()
            except ValueError:
                details = response.text
            raise RuntimeError(f"Pointing server locate failed: {details}")
        payload = response.json()
        self.molmo_result = payload.get("text", "")
        points = payload.get("points", [])
        return [(int(point[0]), int(point[1])) for point in points]

    def _generate_with_hf(self, prompt: str, image) -> str:
        if not hasattr(self, "model") or not hasattr(self, "processor"):
            print("Initializing Hugging Face Molmo")
            self.post_init()
        pil_image = Image.fromarray(image).convert("RGB")
        inputs = self.processor.process(
            images=[pil_image],
            text=prompt,
        )
        inputs = {
            key: value.to(self.device).unsqueeze(0)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                enabled=self.device.type == "cuda",
                dtype=torch.float16,
            ):
                output = self.model.generate_from_batch(
                    inputs,
                    self.GenerationConfig(
                        max_new_tokens=200,
                        stop_strings="<|endoftext|>",
                        use_cache=True,
                    ),
                    tokenizer=self.processor.tokenizer,
                )
        generated_tokens = output[0, inputs["input_ids"].size(1):]
        return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _build_localization_prompt(self, target: str, retry: bool = False) -> str:
        return build_localization_prompt(target, retry=retry)

    def generate_text(self, prompt: str, image) -> str:
        if self._is_local_hf():
            return self._generate_with_hf(prompt, image)
        return self._generate_with_openai_client(prompt, image)

    def parse_coordinates(self, text: str):
        return parse_coordinates(text)

    def _to_pixel_coordinates(self, points, image_shape):
        return to_pixel_coordinates(points, image_shape)

    # def gen_point(self, image:Image, object_name:str):
    def gen_point(self, image:Image, prompt:str):
        if self._is_pointing_server():
            points = self._locate_with_pointing_server(prompt, image)
            print("Pointing server result: ", self.molmo_result)
            return points
        self.molmo_result = self.generate_text(self._build_localization_prompt(prompt), image)
        print("Pointing result: ", self.molmo_result)
        if 'none' in self.molmo_result.lower():
            return []
        points = self.parse_coordinates(self.molmo_result)
        if not points:
            retry_result = self.generate_text(self._build_localization_prompt(prompt, retry=True), image)
            print("Retry pointing result: ", retry_result)
            self.molmo_result = retry_result
            if 'none' in retry_result.lower():
                return []
            points = self.parse_coordinates(retry_result)
        if not points:
            print("Failed to parse Molmo coordinates from output")
            return []
        return self._to_pixel_coordinates(points, image.shape)

    def close(self):
        if self._managed_server_process is not None:
            try:
                os.killpg(self._managed_server_process.pid, signal.SIGTERM)
                self._managed_server_process.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(self._managed_server_process.pid, signal.SIGKILL)
                except Exception:
                    pass
            self._managed_server_process = None
        if self._managed_server_log_handle is not None:
            try:
                self._managed_server_log_handle.close()
            except Exception:
                pass
            self._managed_server_log_handle = None
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        self.client = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Planner:
    def __init__(self, model_id, model_url: str = "http://127.0.0.1:9164", api_key: str = "EMPTY"):
        self.model_id = model_id
        self.model_url = model_url.rstrip("/")
        self.api_key = api_key
        self.last_result = {}

    def post_init(self):
        healthcheck_url = f"{self.model_url}/health"
        try:
            response = requests.get(healthcheck_url, timeout=5)
            response.raise_for_status()
            print(f"Connected to planner server: {healthcheck_url}")
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to reach planner server at {healthcheck_url}") from exc

    def plan(
        self,
        task_text: str,
        image,
        current_interaction: str = "",
        last_action_summary: str = "",
        plan_count: int = 0,
        history: str = "",
        last_plan=None,
    ):
        if not getattr(self, "_planner_server_ready", False):
            print("Initializing planner server client")
            self.post_init()
            self._planner_server_ready = True
        if isinstance(last_plan, dict):
            last_plan = json.dumps(last_plan, ensure_ascii=False)
        elif last_plan is None:
            last_plan = ""
        response = requests.post(
            f"{self.model_url}/plan",
            json={
                "task_text": task_text,
                "image_base64": encode_image_base64(image),
                "model_id": self.model_id,
                "current_interaction": current_interaction,
                "last_action_summary": last_action_summary,
                "plan_count": plan_count,
                "history": history,
                "last_plan": last_plan,
            },
            timeout=180,
        )
        response.raise_for_status()
        self.last_result = response.json()
        return self.last_result

def reset_fn(env_name, session):
    image = session.reset(env_name)
    return image, session

def step_fn(act_key, session):
    # action = NOOP_ACTION.copy()
    action = self.env.noop_action()
    if act_key != "null":
        action[act_key] = 1
    image = session.step(action)
    # image = Image.fromarray(image)
    return image, session

def loop_step_fn(steps, session):
    for i in range(steps):
        image = session.step()
        status = f"Running Agent `Rocket` steps: {i+1}/{steps}. "
        # image = Image.fromarray(image)
        yield image, session.num_steps, status, session

def clear_memory_fn(session):
    image = session.current_image
    session.clear_agent_memory()
    return image, "0", session

def get_points_with_draw(image, label, session, evt: gr.SelectData):
    points = session.points
    point_label = session.points_label
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 5, (0, 255, 0) if label == 'Add Points' else (255, 0, 0)
    points.append([x, y])
    point_label.append(1 if label == 'Add Points' else 0)
    cv2.circle(image, (x, y), point_radius, point_color, -1)
    return image, session

def clear_points_fn(session):
    session.clear_points()
    return session.current_image, session

def segment_fn(session):
    session.segment()
    image = session.apply_mask()
    return image, session

def clear_segment_fn(session):
    session.clear_obj_mask()
    session.tracking_flag = False
    return session.current_image, False, session

def set_tracking_mode(tracking_flag, session):
    session.tracking_flag = tracking_flag
    return session

def set_segment_type(segment_type, session):
    session.segment_type = segment_type
    return session

def play_fn(session):
    image = session.step()
    return image, session

def make_video_fn(session, make_video, save_video, progress=gr.Progress()):
    images = session.image_history
    if len(images) == 0:
        return session, make_video, save_video
    filepath = "rocket.mp4"
    h, w = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filepath, fourcc, 20.0, (w, h))
    for image in progress.tqdm(images):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)
    video.release()
    session.image_history = []
    return session, gr.Button("Make Video", visible=False), gr.DownloadButton("Save Video", value=filepath, visible=True)

def save_video_fn(session, make_video, save_video):
    return session, gr.Button("Make Video", visible=True), gr.DownloadButton("Save Video", visible=False)

def choose_sam_fn(sam_choice, session):
    session.sam_choice = sam_choice
    session.load_sam()
    return session
def molmo_fn(molmo_text, molmo_session,session):
    img = Image.fromarray(session.current_image)
    output_text = molmo_session.generate(img, molmo_text)
    return output_text

def extract_points(data):
    # 匹配 x 和 y 坐标的值，支持 <points> 和 <point> 标签
    pattern = r'x\d?="([-+]?\d*\.\d+|\d+)" y\d?="([-+]?\d*\.\d+|\d+)"'
    points = re.findall(pattern, data)
    # 将提取到的坐标转换为浮点数
    points = [(float(x)/100*640, float(y)/100*360) for x, y in points]
    
    return points

def add_points_fn(image, text, session):
    new_points = extract_points(text)
    points = session.points
    point_label = session.points_label
    for x, y in new_points:
        point_radius, point_color = 5, (0, 255, 0) 
        x,y = int(x),int(y)
        points.append([x, y])
        point_label.append(1)
        cv2.circle(image, (x, y), point_radius, point_color, -1)
    return image, session
