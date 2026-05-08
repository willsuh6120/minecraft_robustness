import base64
import re
from io import BytesIO
from typing import List, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image


PointList = List[Tuple[int, int]]


def encode_image_base64(image: np.ndarray) -> str:
    img = Image.fromarray(image)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")


def decode_image_base64(image_base64: str) -> np.ndarray:
    raw = base64.b64decode(image_base64)
    return np.array(Image.open(BytesIO(raw)).convert("RGB"))


def canonicalize_target_text(target: str) -> str:
    normalized = re.sub(r"\s+", " ", target.strip().lower())
    normalized = re.sub(r"^target object:\s*", "", normalized)
    normalized = re.sub(r"^the\s+", "", normalized)

    if any(token in normalized for token in ["oak tree trunk", "oak log", "wood", "tree trunk"]):
        return "oak tree trunk"
    if "crafting table" in normalized:
        return "crafting table"
    if "coal" in normalized:
        return "coal ore"
    if "iron" in normalized:
        return "iron ore"
    if "diamond" in normalized:
        return "diamond ore"
    if "obsidian" in normalized:
        return "obsidian block"
    if "sheep" in normalized:
        return "sheep"
    if "cow" in normalized:
        return "cow"
    if "zombie" in normalized:
        return "zombie"
    if "skeleton" in normalized:
        return "skeleton"
    if "spider" in normalized:
        return "spider"
    if "boat" in normalized:
        return "boat"
    if "bed" in normalized:
        return "bed"
    if "chest" in normalized:
        return "chest"
    if "door" in normalized or "gate" in normalized:
        return "door"
    if "fire" in normalized:
        return "fire"

    return normalized or target.strip()


def build_localization_prompt(target: str, retry: bool = False) -> str:
    raw_target = re.sub(r"\s+", " ", target.strip())
    lowered = raw_target.lower()
    if any(token in lowered for token in ["point to", "choose", "do not", "return exactly", "ignore the minecraft hud"]):
        instruction = raw_target
        if not instruction.endswith("."):
            instruction += "."
    else:
        target = canonicalize_target_text(raw_target)
        instruction = f"Point to the center of {target}."
    if retry:
        return (
            f"{instruction} Ignore the Minecraft HUD and text overlays. "
            'Return exactly one point in this format: <point x="12.3" y="45.6"/> '
            "using percentage coordinates from 0 to 100. "
            "Do not return a box. Do not return multiple points. "
            "If the target is absent, return exactly: None."
        )
    return (
        f"{instruction} Ignore the Minecraft HUD and text overlays. "
        'Return exactly one point only: <point x="12.3" y="45.6"/> '
        "using percentage coordinates from 0 to 100. "
        "No box. No multiple points. No explanation. "
        "If absent, return exactly: None."
    )


def parse_coordinates(text: str) -> List[Tuple[float, float]]:
    text = text.strip()
    coordinates: List[Tuple[float, float]] = []

    try:
        root = ET.fromstring(text)
        if root.tag.lower() == "point" and "x" in root.attrib and "y" in root.attrib:
            return [(float(root.attrib["x"]), float(root.attrib["y"]))]
        for attr_name, attr_value in root.attrib.items():
            if attr_name.startswith("x"):
                y_attr_name = "y" + attr_name[1:]
                if y_attr_name in root.attrib:
                    coordinates.append((float(attr_value), float(root.attrib[y_attr_name])))
        if len(coordinates) >= 2:
            x_coords = [point[0] for point in coordinates]
            y_coords = [point[1] for point in coordinates]
            return [((min(x_coords) + max(x_coords)) / 2, (min(y_coords) + max(y_coords)) / 2)]
        if coordinates:
            return coordinates
    except ET.ParseError:
        pass

    single_point_match = re.search(
        r'<point[^>]*x\s*=\s*["\']?([-+]?\d*\.\d+|\d+)["\']?[^>]*y\s*=\s*["\']?([-+]?\d*\.\d+|\d+)["\']?[^>]*/?>',
        text,
        flags=re.IGNORECASE,
    )
    if single_point_match:
        return [(float(single_point_match.group(1)), float(single_point_match.group(2)))]

    attr_matches = re.findall(
        r'x\d?\s*=\s*["\']?([-+]?\d*\.\d+|\d+)["\']?[^xy\n\r<>]*y\d?\s*=\s*["\']?([-+]?\d*\.\d+|\d+)["\']?',
        text,
        flags=re.IGNORECASE,
    )
    if attr_matches:
        if len(attr_matches) >= 2:
            x_coords = [float(x) for x, _ in attr_matches]
            y_coords = [float(y) for _, y in attr_matches]
            return [((min(x_coords) + max(x_coords)) / 2, (min(y_coords) + max(y_coords)) / 2)]
        return [(float(x), float(y)) for x, y in attr_matches]

    point_matches = re.findall(
        r'[\(\[]\s*([-+]?\d*\.\d+|\d+)\s*,\s*([-+]?\d*\.\d+|\d+)\s*[\)\]]',
        text,
    )
    if point_matches:
        return [(float(x), float(y)) for x, y in point_matches]

    labeled_matches = re.findall(
        r'x\d?\s*[:=]\s*([-+]?\d*\.\d+|\d+)\s*[,;\s]+y\d?\s*[:=]\s*([-+]?\d*\.\d+|\d+)',
        text,
        flags=re.IGNORECASE,
    )
    if labeled_matches:
        return [(float(x), float(y)) for x, y in labeled_matches]

    return []


def to_pixel_coordinates(points: Sequence[Tuple[float, float]], image_shape) -> PointList:
    height, width = image_shape[:2]
    normalized_points: PointList = []
    for x, y in points:
        if 0 <= x <= 1 and 0 <= y <= 1:
            px, py = x * width, y * height
        elif 0 <= x <= 100 and 0 <= y <= 100:
            px, py = x / 100 * width, y / 100 * height
        else:
            px, py = x, y
        normalized_points.append((int(round(px)), int(round(py))))
    return normalized_points
