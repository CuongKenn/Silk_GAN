import argparse
import json
import os

import pandas as pd

from src.utils import ensure_dir, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data/dataset.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    metadata_dir = cfg["paths"]["metadata_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    train_dir = os.path.join(processed_dir, "train")

    captions_csv = os.path.join(metadata_dir, "captions.csv")
    out_jsonl = os.path.join(train_dir, "metadata.jsonl")
    ensure_dir(train_dir)

    if not os.path.exists(captions_csv):
        raise FileNotFoundError(f"Missing {captions_csv}. Run caption_dataset.py first.")

    df = pd.read_csv(captions_csv)
    if "split" in df.columns:
        df = df[df["split"] == "train"].copy()

    missing = {"file_name", "caption"} - set(df.columns)
    if missing:
        raise RuntimeError(f"captions.csv missing required columns: {sorted(missing)}")

    count = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            file_name = str(row["file_name"])
            if not os.path.exists(os.path.join(train_dir, file_name)):
                continue
            record = {"file_name": file_name, "text": str(row["caption"]).strip()}
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1

    if count == 0:
        raise RuntimeError("No valid rows written to metadata.jsonl. Verify train split images and captions.csv.")

    print(f"Saved {count} records: {out_jsonl}")


if __name__ == "__main__":
    main()
