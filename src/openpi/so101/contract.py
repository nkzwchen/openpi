"""LeRobot SO101 absolute positions, full front/wrist frames, no frozen joints."""
import numpy as np

VERSION = "so101_openpi_absolute6_v1"
PROMPT = "pick blocks from table"
FPS = 10
CAMERAS = ("cam_main", "cam_wrist")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
STATE_ORDER = JOINTS
# Explicit LeRobot use_degrees=False: five calibrated [-100,100] arm positions,
# and calibrated [0,100] gripper position. These are absolute targets, not deltas.
LOW = np.array([-100.] * 5 + [0.], dtype=np.float32)
HIGH = np.full(6, 100., dtype=np.float32)
UNITS = "lerobot_range_m100_100_arm_range_0_100_gripper"


def squeeze_history(value, shape):
    array = np.asarray(value)
    if array.shape == (1, *shape):
        array = array[0]
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Expected finite {shape} or {(1, *shape)}, got {array.shape}")
    return array


def validate_action(value):
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (6,) or not np.isfinite(action).all() or np.any(action < LOW) or np.any(action > HIGH):
        raise ValueError("Expected six absolute LeRobot positions: arm [-100,100], gripper [0,100]")
    return action


def validate_image(value):
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
        raise ValueError("Expected full RGB uint8 HWC image")
    return image


def observation_to_input(obs, prompt=PROMPT):
    # Measured follower position is retained as observation.state, as in LeRobot.
    result = {"observation/state": validate_action(squeeze_history(obs["state"], (6,))), "prompt": prompt}
    for camera in CAMERAS:
        result[f"observation/{camera}"] = validate_image(obs[camera]).copy()
    return result


def positions_to_array(values):
    return validate_action([values[f"{joint}.pos"] for joint in JOINTS])


def array_to_positions(values):
    return dict(zip((f"{joint}.pos" for joint in JOINTS), map(float, validate_action(values)), strict=True))
