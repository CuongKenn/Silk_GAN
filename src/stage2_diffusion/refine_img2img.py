import argparse
import glob
import os

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm

from src.utils import ensure_dir, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage2/infer.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        cfg["model"]["base_model"],
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if cfg["model"].get("lora_dir"):
        pipe.load_lora_weights(cfg["model"]["lora_dir"])
    pipe = pipe.to(device)

    in_dir = cfg["paths"]["input_dir"]
    out_dir = cfg["paths"]["output_dir"]
    ensure_dir(out_dir)

    files = sorted(glob.glob(os.path.join(in_dir, "*.png")))
    if not files:
        raise RuntimeError(f"No input files found in {in_dir}")

    for p in tqdm(files, desc="Refine"):
        image = Image.open(p).convert("RGB").resize((int(cfg["infer"]["resolution"]), int(cfg["infer"]["resolution"])))
        out = pipe(
            prompt=cfg["infer"]["prompt"],
            negative_prompt=cfg["infer"]["negative_prompt"],
            image=image,
            strength=float(cfg["infer"]["strength"]),
            guidance_scale=float(cfg["infer"]["guidance_scale"]),
            num_inference_steps=int(cfg["infer"]["num_steps"]),
        ).images[0]
        out.save(os.path.join(out_dir, os.path.basename(p)))

    print(f"Saved refined images to {out_dir}")


if __name__ == "__main__":
    main()
