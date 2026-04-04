import argparse
import glob
import os
from typing import List

import cv2
import lpips
import numpy as np
import torch
from cleanfid import fid
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def list_pngs(path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(path, "*.png")))


def pair_by_name(real_paths: List[str], fake_paths: List[str]) -> List[tuple[str, str]]:
    real_map = {os.path.basename(p): p for p in real_paths}
    fake_map = {os.path.basename(p): p for p in fake_paths}
    keys = sorted(set(real_map) & set(fake_map))
    return [(real_map[k], fake_map[k]) for k in keys]


def seam_score(img_rgb: np.ndarray, k: int = 8) -> float:
    left = img_rgb[:, :k].astype(np.float32)
    right = img_rgb[:, -k:].astype(np.float32)
    top = img_rgb[:k, :].astype(np.float32)
    bottom = img_rgb[-k:, :].astype(np.float32)
    return float(((left - right) ** 2).mean() + ((top - bottom) ** 2).mean())


def compute_ssim(pairs: List[tuple[str, str]]) -> float:
    if not pairs:
        return float("nan")
    vals = []
    for rp, fp in pairs:
        r = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2GRAY)
        f = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2GRAY)
        f = cv2.resize(f, (r.shape[1], r.shape[0]))
        vals.append(ssim(r, f, data_range=255))
    return float(np.mean(vals))


def compute_lpips(pairs: List[tuple[str, str]], device: str) -> float:
    if not pairs:
        return float("nan")
    model = lpips.LPIPS(net="alex").to(device)
    vals = []
    for rp, fp in tqdm(pairs, desc="LPIPS"):
        r = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
        f = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (r.shape[1], r.shape[0]))

        rt = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0
        ft = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0
        with torch.no_grad():
            vals.append(float(model(rt, ft).item()))
    return float(np.mean(vals))


def compute_seam(paths: List[str]) -> float:
    if not paths:
        return float("nan")
    vals = []
    for p in paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        vals.append(seam_score(img, k=8))
    return float(np.mean(vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, default="data/processed/test")
    parser.add_argument("--fake_dir", type=str, default="outputs/tiles")
    parser.add_argument("--output", type=str, default="outputs/eval_report.txt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    real = list_pngs(args.real_dir)
    fake = list_pngs(args.fake_dir)
    if not real or not fake:
        raise RuntimeError("Missing real or fake images for evaluation")
    pairs = pair_by_name(real, fake)
    if not pairs:
        raise RuntimeError("No matched filenames between real_dir and fake_dir")

    fid_score = fid.compute_fid(args.real_dir, args.fake_dir)
    ssim_score = compute_ssim(pairs)
    lpips_score = compute_lpips(pairs, device=device)
    seam = compute_seam(fake)

    report = [
        f"FID: {fid_score:.4f}",
        f"SSIM: {ssim_score:.4f}",
        f"LPIPS: {lpips_score:.4f}",
        f"SeamMSE(lower is better): {seam:.4f}",
        f"MatchedPairs: {len(pairs)}",
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
