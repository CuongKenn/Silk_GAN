import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
)

from src.utils import ensure_dir, load_yaml


def build_canny_control(image: Image.Image, low: int, high: int) -> Image.Image:
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, low, high)
    edge = np.stack([edge, edge, edge], axis=-1)
    return Image.fromarray(edge)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage2/controlnet.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if not bool(cfg["infer"].get("enabled", False)):
        print("ControlNet refine disabled in config. Skipping.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    controlnet = ControlNetModel.from_pretrained(
        cfg["model"]["controlnet_model"],
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        cfg["model"]["base_model"],
        controlnet=controlnet,
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

    resolution = int(cfg["infer"]["resolution"])
    low = int(cfg["infer"]["canny_low"])
    high = int(cfg["infer"]["canny_high"])

    for p in tqdm(files, desc="ControlNet refine"):
        init_image = Image.open(p).convert("RGB").resize((resolution, resolution))
        control_image = build_canny_control(init_image, low=low, high=high)
        out = pipe(
            prompt=cfg["infer"]["prompt"],
            negative_prompt=cfg["infer"]["negative_prompt"],
            image=init_image,
            control_image=control_image,
            strength=float(cfg["infer"]["strength"]),
            controlnet_conditioning_scale=float(cfg["infer"]["control_scale"]),
            guidance_scale=float(cfg["infer"]["guidance_scale"]),
            num_inference_steps=int(cfg["infer"]["num_steps"]),
        ).images[0]
        out.save(os.path.join(out_dir, os.path.basename(p)))

    print(f"Saved ControlNet refined images to {out_dir}")


if __name__ == "__main__":
    main()
