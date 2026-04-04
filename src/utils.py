import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Any

import cv2
import numpy as np
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image_rgb(path: str, img_rgb: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def border_mse(img_rgb: np.ndarray, k: int = 8) -> float:
    left = img_rgb[:, :k].astype(np.float32)
    right = img_rgb[:, -k:].astype(np.float32)
    top = img_rgb[:k, :].astype(np.float32)
    bottom = img_rgb[-k:, :].astype(np.float32)
    return float(((left - right) ** 2).mean() + ((top - bottom) ** 2).mean())


@dataclass
class SampleRecord:
    image_id: str
    file_name: str
    split: str
    caption: str = ""


def save_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
