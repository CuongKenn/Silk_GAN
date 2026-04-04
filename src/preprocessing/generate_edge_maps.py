import argparse
import glob
import os

import cv2
from tqdm import tqdm

from src.utils import ensure_dir, load_yaml


def auto_canny(gray, sigma: float = 0.33):
    v = float(gray.mean())
    low = int(max(0, (1.0 - sigma) * v))
    high = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, low, high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data/dataset.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rgb_dir = cfg["paths"]["tileable_dir"]
    edge_dir = cfg["paths"]["edge_dir"]
    size = int(cfg["image"]["size"])

    ensure_dir(edge_dir)
    paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    if not paths:
        raise RuntimeError(f"No files found in {rgb_dir}")

    for p in tqdm(paths, desc="Edge maps"):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edge = auto_canny(blur)
        edge = cv2.dilate(edge, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
        out = os.path.join(edge_dir, os.path.basename(p))
        cv2.imwrite(out, edge)

    print(f"Generated edge maps: {edge_dir}")


if __name__ == "__main__":
    main()
