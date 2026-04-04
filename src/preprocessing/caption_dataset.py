import argparse
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.utils import ensure_dir, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data/dataset.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    metadata_dir = cfg["paths"]["metadata_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    index_path = os.path.join(metadata_dir, "dataset_index.csv")
    out_path = os.path.join(metadata_dir, "captions.csv")
    ensure_dir(metadata_dir)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing {index_path}. Run prepare_dataset.py first.")

    df = pd.read_csv(index_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()

    captions = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caption"):
        file_name = row["file_name"]
        split = row["split"]
        img_path = os.path.join(processed_dir, split, file_name)
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            out_ids = model.generate(**inputs, max_new_tokens=32)
            blip_caption = processor.decode(out_ids[0], skip_special_tokens=True)
            final_caption = (
                f"macro photo of lace textile, seamless texture, high detail, {blip_caption}"
            )
        except Exception:
            final_caption = "macro photo of lace textile, intricate threads, perforated pattern, seamless texture"

        captions.append(
            {
                "image_id": row["image_id"],
                "file_name": file_name,
                "split": split,
                "caption": final_caption,
            }
        )

    pd.DataFrame(captions).to_csv(out_path, index=False)
    print(f"Saved captions: {out_path}")


if __name__ == "__main__":
    main()
