from dataclasses import dataclass
from typing import Optional, Tuple

from minestudio.tutorials.inference.evaluate_rocket.interaction_benchmark_spec import ScriptedSubtask


@dataclass(frozen=True)
class PointVerificationResult:
    valid: bool
    reason: str
    normalized_point: Optional[Tuple[float, float]] = None


def verify_point_region(
    point: Optional[Tuple[int, int]],
    image_shape,
    region: str,
) -> PointVerificationResult:
    if point is None:
        return PointVerificationResult(valid=False, reason="missing_point")

    height, width = image_shape[:2]
    if width <= 0 or height <= 0:
        return PointVerificationResult(valid=False, reason="invalid_image_shape")

    x, y = point
    nx = float(x) / float(width)
    ny = float(y) / float(height)
    normalized = (round(nx, 4), round(ny, 4))

    if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
        return PointVerificationResult(valid=False, reason="out_of_bounds", normalized_point=normalized)

    region = (region or "any").strip().lower()
    if region in {"", "any", "none"}:
        return PointVerificationResult(valid=True, reason="unverified_region", normalized_point=normalized)
    if region == "left":
        return PointVerificationResult(valid=nx <= 0.45, reason="left_region", normalized_point=normalized)
    if region == "right":
        return PointVerificationResult(valid=nx >= 0.55, reason="right_region", normalized_point=normalized)
    if region == "center":
        return PointVerificationResult(valid=0.35 <= nx <= 0.65, reason="center_region", normalized_point=normalized)
    if region == "upper":
        return PointVerificationResult(valid=ny <= 0.45, reason="upper_region", normalized_point=normalized)
    if region == "lower":
        return PointVerificationResult(valid=ny >= 0.55, reason="lower_region", normalized_point=normalized)
    return PointVerificationResult(valid=True, reason=f"unknown_region:{region}", normalized_point=normalized)


def verify_subtask_point(
    subtask: ScriptedSubtask,
    point: Optional[Tuple[int, int]],
    image_shape,
) -> PointVerificationResult:
    return verify_point_region(point=point, image_shape=image_shape, region=subtask.verification_region)
