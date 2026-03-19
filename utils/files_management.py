from pathlib import Path
import json
import csv
import pickle
import numpy as np

from objects.Objects import Sensor


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, data: dict | list) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fd:
        json.dump(data, fd, indent=2)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(output_path, "w", encoding="utf-8", newline="") as fd:
            fd.write("")
        return

    with open(output_path, "w", encoding="utf-8", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_configuration(config_path: str) -> tuple[list[Sensor], np.ndarray]:
    config_file = Path(config_path)
    if not config_file.exists():
        example_file = config_file.with_name(f"{config_file.name}.example")
        if example_file.exists():
            config_file = example_file
        else:
            raise FileNotFoundError(
                f"Could not find '{config_path}' or fallback '{example_file.name}'."
            )

    with open(config_file, "rb") as fd:
        configuration = pickle.load(fd)

    sensors: list[Sensor] = configuration["sensors"]
    robot_size = np.asarray(configuration["robot_size"], dtype=np.float32)

    for sensor in sensors:
        sensor.clear_cache()

    return sensors, robot_size
