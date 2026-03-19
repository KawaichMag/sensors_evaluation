import numpy as np
import math


def parse_sensor_parameter(
    value: float | list[float] | tuple[float, ...] | np.ndarray,
    sensor_count: int,
    name: str,
) -> np.ndarray:
    if np.isscalar(value):
        return np.full(sensor_count, float(value), dtype=np.float32)  # type: ignore

    parsed = np.asarray(value, dtype=np.float32)
    if parsed.shape != (sensor_count,):
        raise ValueError(
            f"{name} must be a scalar or have exactly {sensor_count} values, got shape {parsed.shape}."
        )
    return parsed


def parse_angle_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return None
    return [math.radians(float(part)) for part in parts]
