import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from minestudio.tutorials.inference.evaluate_rocket.pointing_common import encode_image_base64
from minestudio.tutorials.inference.evaluate_rocket.pointing_server import PointingBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="describe", choices=["describe", "locate"])
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--model-id", type=str, default="allenai/MolmoE-1B-0924")
    parser.add_argument("--molmo-loader", type=str, default="official", choices=["manual", "official"])
    parser.add_argument(
        "--molmo-torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--molmo-autocast-dtype",
        type=str,
        default="bfloat16",
        choices=["none", "float16", "bfloat16"],
    )
    parser.add_argument("--molmo-device-map", type=str, default="auto")
    args = parser.parse_args()

    image = np.array(Image.open(Path(args.image_path)).convert("RGB"))
    image_base64 = encode_image_base64(image)
    backend = PointingBackend(
        backend="molmo-local",
        backend_url=None,
        model_id=args.model_id,
        api_key="EMPTY",
        timeout=180,
        molmo_loader=args.molmo_loader,
        molmo_torch_dtype=args.molmo_torch_dtype,
        molmo_autocast_dtype=args.molmo_autocast_dtype,
        molmo_device_map=args.molmo_device_map,
    )

    if args.mode == "describe":
        text = backend.generate_text(args.prompt, image, image_base64)
        print(text)
        return

    result = backend.locate(args.prompt, image_base64)
    print(result)


if __name__ == "__main__":
    main()
