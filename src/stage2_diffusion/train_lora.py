import argparse
import os
import subprocess

from src.utils import ensure_dir, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage2/lora.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ensure_dir(cfg["paths"]["output_dir"])

    train_data_dir = cfg["data"]["image_dir"]
    caption_col = cfg["data"]["caption_column"]
    image_col = cfg["data"]["image_column"]
    metadata_jsonl = os.path.join(train_data_dir, "metadata.jsonl")
    if not os.path.exists(train_data_dir):
        raise FileNotFoundError(f"Missing train_data_dir: {train_data_dir}")
    if not os.path.exists(metadata_jsonl):
        raise FileNotFoundError(
            f"Missing {metadata_jsonl}. Run src.preprocessing.build_diffusion_dataset first."
        )

    cmd = [
        "accelerate",
        "launch",
        "-m",
        "diffusers.examples.text_to_image.train_text_to_image_lora",
        "--pretrained_model_name_or_path",
        cfg["model"]["base_model"],
        "--train_data_dir",
        train_data_dir,
        "--image_column",
        image_col,
        "--caption_column",
        caption_col,
        "--resolution",
        str(cfg["train"]["resolution"]),
        "--train_batch_size",
        str(cfg["train"]["batch_size"]),
        "--learning_rate",
        str(cfg["train"]["learning_rate"]),
        "--max_train_steps",
        str(cfg["train"]["max_steps"]),
        "--checkpointing_steps",
        str(cfg["train"]["checkpoint_every"]),
        "--validation_prompt",
        cfg["train"]["validation_prompt"],
        "--output_dir",
        cfg["paths"]["output_dir"],
        "--rank",
        str(cfg["train"]["lora_rank"]),
        "--mixed_precision",
        cfg["train"]["mixed_precision"],
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("LoRA training complete")


if __name__ == "__main__":
    main()
