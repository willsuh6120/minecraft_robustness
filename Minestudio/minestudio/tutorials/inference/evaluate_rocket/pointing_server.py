import argparse
from contextlib import nullcontext
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

import requests

from minestudio.tutorials.inference.evaluate_rocket.pointing_common import (
    build_localization_prompt,
    decode_image_base64,
    parse_coordinates,
    to_pixel_coordinates,
)


class PointingBackend:
    def __init__(
        self,
        backend: str,
        backend_url: Optional[str],
        model_id: str,
        api_key: str,
        timeout: int,
        molmo_loader: str = "manual",
        molmo_torch_dtype: str = "float16",
        molmo_autocast_dtype: str = "float16",
        molmo_device_map: str = "",
    ):
        self.backend = backend
        self.backend_url = backend_url.rstrip("/") if backend_url else None
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout
        self.molmo_loader = molmo_loader
        self.molmo_torch_dtype = molmo_torch_dtype
        self.molmo_autocast_dtype = molmo_autocast_dtype
        self.molmo_device_map = molmo_device_map
        self._molmo_lock = threading.Lock()
        self._molmo_processor = None
        self._molmo_model = None
        self._molmo_generation_config = None
        self._molmo_device = None
        self._molmo_dtype = None
        self._molmo_loaded_model_id = None
        self._finalize_openai_settings()

    def _finalize_openai_settings(self):
        if self.backend == "openai":
            self.backend = "openai-compatible"
            if not self.backend_url:
                self.backend_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            if self.api_key == "EMPTY":
                self.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

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

    def _resolve_torch_dtype(self, dtype_name: str):
        import torch

        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype_name == "auto":
            return "auto"
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported Molmo dtype: {dtype_name}")
        return mapping[dtype_name]

    def _resolve_autocast_dtype(self, dtype_name: str):
        if dtype_name == "none":
            return None
        resolved = self._resolve_torch_dtype(dtype_name)
        if resolved == "auto":
            raise ValueError("molmo autocast dtype cannot be auto")
        return resolved

    def _load_molmo_manual(self, local_model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        self._molmo_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested_dtype = self._resolve_torch_dtype(self.molmo_torch_dtype)
        self._molmo_dtype = torch.float16 if requested_dtype == "auto" and self._molmo_device.type == "cuda" else (
            torch.float32 if requested_dtype == "auto" else requested_dtype
        )
        self._molmo_generation_config = GenerationConfig
        try:
            self._molmo_processor = AutoProcessor.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=True,
                use_fast=True,
            )
        except Exception as exc:
            print(f"Falling back to slow Molmo processor after fast tokenizer load failed: {exc}")
            self._molmo_processor = AutoProcessor.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=True,
                use_fast=False,
            )
        self._molmo_model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=self._molmo_dtype,
        )
        self._molmo_model.to(self._molmo_device)
        self._molmo_model.eval()

    def _load_molmo_official(self, local_model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        requested_dtype = self._resolve_torch_dtype(self.molmo_torch_dtype)
        device_map = self.molmo_device_map or "auto"
        processor_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if requested_dtype == "auto":
            processor_kwargs["torch_dtype"] = "auto"
        processor_kwargs["device_map"] = device_map
        self._molmo_processor = AutoProcessor.from_pretrained(
            local_model_path,
            **processor_kwargs,
        )
        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "device_map": device_map,
            "torch_dtype": "auto" if requested_dtype == "auto" else requested_dtype,
        }
        self._molmo_model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            **model_kwargs,
        )
        self._molmo_generation_config = GenerationConfig
        self._molmo_device = getattr(self._molmo_model, "device", None)
        if self._molmo_device is None:
            self._molmo_device = next(self._molmo_model.parameters()).device
        self._molmo_dtype = requested_dtype
        self._molmo_model.eval()

    def _ensure_local_molmo(self, model_id: str):
        if (
            self._molmo_model is not None
            and self._molmo_processor is not None
            and self._molmo_loaded_model_id == model_id
        ):
            return
        with self._molmo_lock:
            if (
                self._molmo_model is not None
                and self._molmo_processor is not None
                and self._molmo_loaded_model_id == model_id
            ):
                return

            local_model_path = self._resolve_local_snapshot_path(model_id)
            print(f"Loading Molmo from local path: {local_model_path}")
            self._molmo_processor = None
            self._molmo_model = None
            self._molmo_generation_config = None
            self._molmo_device = None
            self._molmo_dtype = None
            if self.molmo_loader == "official":
                self._load_molmo_official(local_model_path)
            else:
                self._load_molmo_manual(local_model_path)
            self._molmo_loaded_model_id = model_id

    def _generate_text_molmo_local(self, prompt: str, image, model_id: str) -> str:
        self._ensure_local_molmo(model_id)
        from PIL import Image

        pil_image = Image.fromarray(image).convert("RGB")
        inputs = self._molmo_processor.process(images=[pil_image], text=prompt)
        inputs = {
            key: value.to(self._molmo_device).unsqueeze(0)
            for key, value in inputs.items()
        }
        with self._molmo_lock:
            import torch

            autocast_dtype = self._resolve_autocast_dtype(self.molmo_autocast_dtype)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if self._molmo_device.type == "cuda" and autocast_dtype is not None
                else nullcontext()
            )
            with torch.inference_mode():
                with autocast_context:
                    output = self._molmo_model.generate_from_batch(
                        inputs,
                        self._molmo_generation_config(
                            max_new_tokens=200,
                            stop_strings="<|endoftext|>",
                            use_cache=True,
                        ),
                        tokenizer=self._molmo_processor.tokenizer,
                    )
        generated_tokens = output[0, inputs["input_ids"].size(1):]
        return self._molmo_processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _normalize_content(self, content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_chunks = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_chunks.append(item.get("text", ""))
            return "\n".join(chunk for chunk in text_chunks if chunk).strip()
        return str(content)

    def generate_text(self, prompt: str, image, image_base64: str, model_id: Optional[str] = None) -> str:
        model_id = model_id or self.model_id
        if self.backend == "mock":
            return '<points x1="50" y1="50"/>'
        if self.backend == "molmo-local":
            return self._generate_text_molmo_local(prompt, image, model_id)

        if self.backend != "openai-compatible":
            raise ValueError(f"Unsupported backend: {self.backend}")
        if not self.backend_url:
            raise ValueError("--backend-url is required for openai-compatible backend")

        response = requests.post(
            f"{self.backend_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": model_id,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]["content"]
        return self._normalize_content(message)

    def locate(self, prompt: str, image_base64: str, model_id: Optional[str] = None):
        image = decode_image_base64(image_base64)
        text = self.generate_text(build_localization_prompt(prompt), image, image_base64, model_id=model_id)
        points = [] if "none" in text.lower() else parse_coordinates(text)
        if not points:
            retry_text = self.generate_text(
                build_localization_prompt(prompt, retry=True),
                image,
                image_base64,
                model_id=model_id,
            )
            text = retry_text
            points = [] if "none" in retry_text.lower() else parse_coordinates(retry_text)
        pixel_points = to_pixel_coordinates(points, image.shape) if points else []
        return {
            "text": text,
            "points": pixel_points,
            "backend": self.backend,
            "model_id": model_id or self.model_id,
        }


def make_handler(backend: PointingBackend):
    class PointingHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "backend": backend.backend,
                        "model_id": backend.model_id,
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/locate":
                self._send_json(404, {"error": "not found"})
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
                prompt = payload["prompt"]
                image_base64 = payload["image_base64"]
                model_id = payload.get("model_id")
                result = backend.locate(prompt, image_base64, model_id=model_id)
                self._send_json(200, result)
            except Exception as exc:
                self._send_json(
                    500,
                    {
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )

        def log_message(self, fmt, *args):
            print(f"[pointing-server] {self.address_string()} - {fmt % args}")

    return PointingHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9163)
    parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        choices=["mock", "molmo-local", "openai-compatible", "openai"],
    )
    parser.add_argument("--backend-url", type=str, default="")
    parser.add_argument("--model-id", type=str, default="gpt-4o-mini")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--molmo-loader", type=str, default="manual", choices=["manual", "official"])
    parser.add_argument(
        "--molmo-torch-dtype",
        type=str,
        default="float16",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--molmo-autocast-dtype",
        type=str,
        default="float16",
        choices=["none", "float16", "bfloat16"],
    )
    parser.add_argument("--molmo-device-map", type=str, default="")
    args = parser.parse_args()

    backend = PointingBackend(
        backend=args.backend,
        backend_url=args.backend_url,
        model_id=args.model_id,
        api_key=args.api_key,
        timeout=args.timeout,
        molmo_loader=args.molmo_loader,
        molmo_torch_dtype=args.molmo_torch_dtype,
        molmo_autocast_dtype=args.molmo_autocast_dtype,
        molmo_device_map=args.molmo_device_map,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(backend))
    print(
        f"Pointing server listening on http://{args.host}:{args.port} "
        f"(backend={args.backend}, model_id={args.model_id})"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
