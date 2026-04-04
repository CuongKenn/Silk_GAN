import argparse
import os
import subprocess
import sys

from src.utils import ensure_dir, load_yaml, seed_everything


def run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def maybe_clone_repo(repo_dir: str, auto_clone: bool, repo_url: str) -> None:
    if os.path.exists(os.path.join(repo_dir, "train.py")):
        return
    if not auto_clone:
        raise FileNotFoundError(
            "StyleGAN2-ADA repo not found. Set stylegan2_ada.auto_clone=true in config or clone manually."
        )
    ensure_dir(os.path.dirname(repo_dir) or ".")
    run(["git", "clone", repo_url, repo_dir])


def ensure_dataset_zip(py: str, repo_dir: str, source_dir: str, dest_zip: str, resolution: int) -> None:
    if os.path.exists(dest_zip):
        return
    ensure_dir(os.path.dirname(dest_zip) or ".")
    dataset_tool = os.path.join(repo_dir, "dataset_tool.py")
    if not os.path.exists(dataset_tool):
        raise FileNotFoundError(f"Missing dataset_tool.py in {repo_dir}")
    run(
        [
            py,
            dataset_tool,
            "--source",
            source_dir,
            "--dest",
            dest_zip,
            "--resolution",
            f"{resolution}x{resolution}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage1/stylegan.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    py = sys.executable

    repo_cfg = cfg["stylegan2_ada"]
    repo_dir = repo_cfg["repo_dir"]
    maybe_clone_repo(
        repo_dir=repo_dir,
        auto_clone=bool(repo_cfg.get("auto_clone", True)),
        repo_url=repo_cfg.get("repo_url", "https://github.com/NVlabs/stylegan2-ada-pytorch.git"),
    )

    data_cfg = cfg["data"]
    source_dir = data_cfg["train_dir"]
    resolution = int(data_cfg["image_size"])
    dataset_zip = data_cfg["dataset_zip"]
    ensure_dataset_zip(
        py=py,
        repo_dir=repo_dir,
        source_dir=source_dir,
        dest_zip=dataset_zip,
        resolution=resolution,
    )

    out_dir = cfg["paths"]["checkpoint_dir"]
    ensure_dir(out_dir)

    train_cfg = cfg["train"]
    train_py = os.path.join(repo_dir, "train.py")
    if not os.path.exists(train_py):
        raise FileNotFoundError(f"Missing train.py in {repo_dir}")

    cmd = [
        py,
        train_py,
        "--outdir",
        out_dir,
        "--cfg",
        train_cfg.get("cfg", "auto"),
        "--data",
        dataset_zip,
        "--gpus",
        str(train_cfg.get("gpus", 1)),
        "--batch",
        str(train_cfg.get("batch_size", 8)),
        "--kimg",
        str(train_cfg.get("kimg", 3000)),
        "--gamma",
        str(train_cfg.get("gamma", 10.0)),
        "--snap",
        str(train_cfg.get("snap", 10)),
        "--metrics",
        train_cfg.get("metrics", "fid50k_full"),
        "--mirror",
        "1" if bool(train_cfg.get("mirror", True)) else "0",
        "--aug",
        train_cfg.get("aug", "ada"),
    ]

    if int(train_cfg.get("workers", 0)) > 0:
        cmd.extend(["--workers", str(train_cfg["workers"])])

    resume_path = str(train_cfg.get("resume", "")).strip()
    if resume_path:
        cmd.extend(["--resume", resume_path])

    run(cmd)
    print("StyleGAN2-ADA training launched successfully")


if __name__ == "__main__":
    main()
