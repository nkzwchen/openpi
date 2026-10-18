"""Full front/wrist images, six measured positions and six absolute action targets."""
import dataclasses

import numpy as np

from openpi import transforms
from openpi.so101.contract import CAMERAS, squeeze_history, validate_image


@dataclasses.dataclass(frozen=True)
class AbsoluteInputs(transforms.DataTransformFn):
    def __call__(self, data):
        images = {}
        for camera, slot in zip(CAMERAS, ("base_0_rgb", "left_wrist_0_rgb"), strict=True):
            image = validate_image(data[f"observation/{camera}"])
            if image.dtype != np.uint8:
                raise ValueError("Expected RGB uint8 image")
            images[slot] = image
        images["right_wrist_0_rgb"] = np.zeros_like(images["base_0_rgb"])
        result = {"state": squeeze_history(data["observation/state"], (6,)).astype(np.float32),
                  "image": images, "image_mask": {"base_0_rgb": True, "left_wrist_0_rgb": True,
                                                   "right_wrist_0_rgb": False}, "prompt": data["prompt"]}
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 6 or not np.isfinite(actions).all():
                raise ValueError("Expected (horizon, 6) absolute actions")
            result["actions"] = actions
        return result


@dataclasses.dataclass(frozen=True)
class AbsoluteOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] < 6 or not np.isfinite(actions).all():
            raise ValueError("Invalid model action chunk")
        return {"actions": actions[:, :6]}
