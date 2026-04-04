import argparse
import glob
import os
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.utils import ensure_dir


def unsharp_mask(img: np.ndarray, sigma: float = 1.0, amount: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def height_from_luma(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    height = cv2.bilateralFilter(gray, d=7, sigmaColor=20, sigmaSpace=20)
    return height


def normal_from_height(height: np.ndarray, strength: float = 2.0) -> np.ndarray:
    h = height.astype(np.float32) / 255.0
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(nx)
    n = np.stack([nx, ny, nz], axis=-1)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    n = ((n + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return n


def roughness_from_local_var(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mean = cv2.GaussianBlur(gray, (0, 0), 2.0)
    mean2 = cv2.GaussianBlur(gray * gray, (0, 0), 2.0)
    var = np.clip(mean2 - mean * mean, 0.0, 1.0)
    rough = (1.0 - np.sqrt(var))
    return (rough * 255.0).clip(0, 255).astype(np.uint8)


def try_build_realesrgan(scale: int, model_path: Optional[str]):
    try:
        from realesrgan import RealESRGAN
        import torch
        from PIL import Image
    except Exception:
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealESRGAN(device, scale=scale)
    if model_path and os.path.exists(model_path):
        model.load_weights(model_path)
    else:
        model.load_weights(f"weights/RealESRGAN_x{scale}.pth", download=True)
    return model, Image


def upscale_with_realesrgan(model, image_cls, img_rgb: np.ndarray) -> np.ndarray:
    pil = image_cls.fromarray(img_rgb)
    sr = model.predict(pil)
    return np.array(sr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs/samples/stage2_refined")
    parser.add_argument("--output_dir", type=str, default="outputs/tiles")
    parser.add_argument("--pbr_dir", type=str, default="outputs/pbr")
    parser.add_argument("--use_realesrgan", action="store_true")
    parser.add_argument("--sr_scale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--realesrgan_weights", type=str, default="")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.pbr_dir, "normal"))
    ensure_dir(os.path.join(args.pbr_dir, "roughness"))
    ensure_dir(os.path.join(args.pbr_dir, "height"))

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if not files:
        raise RuntimeError(f"No images found in {args.input_dir}")

    sr_model = None
    sr_image_cls = None
    if args.use_realesrgan:
        sr_model, sr_image_cls = try_build_realesrgan(scale=int(args.sr_scale), model_path=args.realesrgan_weights or None)
        if sr_model is None:
            print("Warning: Real-ESRGAN not available. Falling back to bicubic upscale.")

    for p in tqdm(files, desc="Postprocess"):
        name = os.path.basename(p)
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        sharp = unsharp_mask(rgb, sigma=1.0, amount=1.0)
        if args.use_realesrgan:
            if sr_model is not None:
                sharp = upscale_with_realesrgan(sr_model, sr_image_cls, sharp)
            else:
                sharp = cv2.resize(
                    sharp,
                    (sharp.shape[1] * int(args.sr_scale), sharp.shape[0] * int(args.sr_scale)),
                    interpolation=cv2.INTER_CUBIC,
                )
        cv2.imwrite(os.path.join(args.output_dir, name), cv2.cvtColor(sharp, cv2.COLOR_RGB2BGR))

        h = height_from_luma(sharp)
        n = normal_from_height(h, strength=2.0)
        r = roughness_from_local_var(sharp)

        cv2.imwrite(os.path.join(args.pbr_dir, "height", name), h)
        cv2.imwrite(os.path.join(args.pbr_dir, "normal", name), cv2.cvtColor(n, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.pbr_dir, "roughness", name), r)

    print("Post-processing complete")


if __name__ == "__main__":
    main()
