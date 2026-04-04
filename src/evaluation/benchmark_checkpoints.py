import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd
import torch

from cleanfid import fid

from src.evaluation.evaluate_textures import (
    compute_lpips,
    compute_seam,
    compute_ssim,
    list_pngs,
    pair_by_name,
)
from src.utils import ensure_dir


def gather_candidate_dirs(root_dirs: List[str]) -> List[str]:
    candidates = []
    for root in root_dirs:
        if not os.path.exists(root):
            continue
        direct_png = glob.glob(os.path.join(root, "*.png"))
        if direct_png:
            candidates.append(root)
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(d) and glob.glob(os.path.join(d, "*.png")):
                candidates.append(d)
    # preserve order + uniqueness
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def zscore_safe(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    std = x.std()
    if std < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def update_project_plan(plan_path: str, best_label: str) -> None:
    if not os.path.exists(plan_path):
        return
    with open(plan_path, "r", encoding="utf-8") as f:
        text = f.read()

    marker = "- [ ] Chon checkpoint tot nhat"
    if marker in text:
        text = text.replace(marker, f"- [x] Chon checkpoint tot nhat - auto by benchmark: {best_label}")

    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, default="data/processed/test")
    parser.add_argument(
        "--candidate_roots",
        type=str,
        nargs="+",
        default=["outputs/tiles", "outputs/samples/stage2_refined", "outputs/samples/stage2_controlnet"],
    )
    parser.add_argument("--output_csv", type=str, default="outputs/benchmark/leaderboard.csv")
    parser.add_argument("--output_md", type=str, default="outputs/benchmark/leaderboard.md")
    parser.add_argument("--auto_tick_plan", action="store_true")
    parser.add_argument("--plan_path", type=str, default="docs/PROJECT_PLAN.md")
    args = parser.parse_args()

    ensure_dir(os.path.dirname(args.output_csv))
    ensure_dir(os.path.dirname(args.output_md))

    if not os.path.exists(args.real_dir):
        raise RuntimeError(f"real_dir does not exist: {args.real_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_paths = list_pngs(args.real_dir)
    if not real_paths:
        raise RuntimeError(f"No PNG files in {args.real_dir}")

    candidates = gather_candidate_dirs(args.candidate_roots)
    if not candidates:
        raise RuntimeError("No candidate output folders found")

    rows = []
    for c in candidates:
        fake_paths = list_pngs(c)
        if not fake_paths:
            continue
        pairs = pair_by_name(real_paths, fake_paths)
        if not pairs:
            continue

        fid_score = float(fid.compute_fid(args.real_dir, c))
        ssim_score = float(compute_ssim(pairs))
        lpips_score = float(compute_lpips(pairs, device=device))
        seam_score = float(compute_seam(fake_paths))

        rows.append(
            {
                "candidate": c,
                "fid": fid_score,
                "lpips": lpips_score,
                "ssim": ssim_score,
                "seam_mse": seam_score,
                "matched_pairs": len(pairs),
            }
        )

    if not rows:
        raise RuntimeError("No valid candidates to benchmark")

    df = pd.DataFrame(rows)
    # lower is better for fid/lpips/seam; higher is better for ssim
    score = zscore_safe(df["fid"].values) + zscore_safe(df["lpips"].values) + zscore_safe(df["seam_mse"].values) - zscore_safe(df["ssim"].values)
    df["rank_score"] = score
    df = df.sort_values("rank_score", ascending=True).reset_index(drop=True)

    df.to_csv(args.output_csv, index=False)

    md = [
        "# Benchmark Leaderboard",
        "",
        df.to_markdown(index=True),
        "",
        f"Best candidate: {df.iloc[0]['candidate']}",
    ]
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    best_candidate = str(df.iloc[0]["candidate"])
    print(f"Best candidate: {best_candidate}")
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved MD: {args.output_md}")

    if args.auto_tick_plan:
        update_project_plan(args.plan_path, best_candidate)
        print(f"Updated plan: {args.plan_path}")


if __name__ == "__main__":
    main()
