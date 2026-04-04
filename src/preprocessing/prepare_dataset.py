import argparse
import glob
import os
from typing import List, Tuple

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils import ensure_dir, load_yaml, read_image_rgb, write_image_rgb, border_mse


def laplacian_var(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def gray_world_white_balance(img_rgb: np.ndarray) -> np.ndarray:
    img = img_rgb.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    mean_gray = float(means.mean())
    scale = mean_gray / (means + 1e-6)
    img = np.clip(img * scale, 0, 255).astype(np.uint8)
    return img


def clahe_l_channel(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def make_tileable_offset_blend(img_rgb: np.ndarray, strength: float = 0.5) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    shifted = np.roll(np.roll(img_rgb, h // 2, axis=0), w // 2, axis=1)
    yy = np.linspace(-1, 1, h)[:, None]
    xx = np.linspace(-1, 1, w)[None, :]
    alpha = np.exp(-4 * (xx * xx + yy * yy)).astype(np.float32)
    alpha = (alpha * strength + (1.0 - strength)).clip(0, 1)
    alpha = alpha[..., None]
    out = img_rgb.astype(np.float32) * alpha + shifted.astype(np.float32) * (1.0 - alpha)
    return out.clip(0, 255).astype(np.uint8)


def rotate_wrap(img_rgb: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return img_rgb
    h, w = img_rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img_rgb, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def center_crop_resize(img_rgb: np.ndarray, out_size: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = img_rgb[y0 : y0 + s, x0 : x0 + s]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def collect_images(path: str) -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.tif", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(path, ext)))
    files.sort()
    return files


def dedup_by_phash(paths: List[str]) -> List[str]:
    seen = set()
    keep = []
    for p in tqdm(paths, desc="Dedup"):
        h = str(imagehash.phash(Image.open(p).convert("RGB"), hash_size=16))
        if h not in seen:
            seen.add(h)
            keep.append(p)
    return keep


def preprocess_one(path: str, out_size: int, min_sharp: float, max_border_mse: float) -> Tuple[bool, np.ndarray]:
    img = read_image_rgb(path)
    if laplacian_var(img) < min_sharp:
        return False, img
    img = gray_world_white_balance(img)
    img = clahe_l_channel(img)
    img = rotate_wrap(img, int(np.random.choice([0, 90, 180, 270])))
    img = center_crop_resize(img, out_size)
    img = make_tileable_offset_blend(img, strength=0.55)
    if border_mse(img, k=8) > max_border_mse:
        return False, img
    return True, img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data/dataset.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    raw_rgb_dir = cfg["paths"]["raw_rgb_dir"]
    cleaned_dir = cfg["paths"]["cleaned_dir"]
    normalized_dir = cfg["paths"]["normalized_dir"]
    tileable_dir = cfg["paths"]["tileable_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    metadata_dir = cfg["paths"]["metadata_dir"]

    ensure_dir(cleaned_dir)
    ensure_dir(normalized_dir)
    ensure_dir(tileable_dir)
    ensure_dir(processed_dir)
    ensure_dir(metadata_dir)

    img_paths = collect_images(raw_rgb_dir)
    if not img_paths:
        raise RuntimeError(f"No images found in {raw_rgb_dir}")

    img_paths = dedup_by_phash(img_paths)
    records = []

    for p in tqdm(img_paths, desc="Preprocess"):
        ok, out = preprocess_one(
            p,
            out_size=int(cfg["image"]["size"]),
            min_sharp=float(cfg["filters"]["min_sharpness"]),
            max_border_mse=float(cfg["filters"]["max_border_mse"]),
        )
        if not ok:
            continue

        base = os.path.splitext(os.path.basename(p))[0]
        cleaned_path = os.path.join(cleaned_dir, f"{base}.png")
        normalized_path = os.path.join(normalized_dir, f"{base}.png")
        tileable_path = os.path.join(tileable_dir, f"{base}.png")

        write_image_rgb(cleaned_path, out)
        write_image_rgb(normalized_path, out)
        write_image_rgb(tileable_path, out)
        records.append({"image_id": base, "file_name": f"{base}.png"})

    if len(records) < 10:
        raise RuntimeError("Too few valid images after preprocessing. Check quality thresholds.")

    df = pd.DataFrame(records)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - float(cfg["split"]["train"])),
        random_state=int(cfg["seed"]),
        shuffle=True,
    )
    rel_val = float(cfg["split"]["val"]) / (float(cfg["split"]["val"]) + float(cfg["split"]["test"]))
    val_df, test_df = train_test_split(temp_df, test_size=(1.0 - rel_val), random_state=int(cfg["seed"]))

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_dir = os.path.join(processed_dir, split_name)
        ensure_dir(split_dir)
        for _, row in split_df.iterrows():
            src = os.path.join(tileable_dir, row["file_name"])
            dst = os.path.join(split_dir, row["file_name"])
            img = read_image_rgb(src)
            write_image_rgb(dst, img)

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df["caption"] = "macro photo of lace textile, intricate threads, perforated pattern, seamless texture"
    all_df.to_csv(os.path.join(metadata_dir, "dataset_index.csv"), index=False)

    print("Preprocessing complete")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


if __name__ == "__main__":
    main()
