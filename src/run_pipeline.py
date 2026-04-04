import argparse
import subprocess
import sys


def run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="all", choices=["all", "data", "stage1", "stage2", "post", "eval", "benchmark"])
    parser.add_argument("--gan_mode", type=str, default="stylegan", choices=["stylegan", "pix2pix"])
    args = parser.parse_args()

    py = sys.executable

    if args.stage in ["all", "data"]:
        run([py, "-m", "src.preprocessing.prepare_dataset", "--config", "configs/data/dataset.yaml"])
        run([py, "-m", "src.preprocessing.generate_edge_maps", "--config", "configs/data/dataset.yaml"])
        run([py, "-m", "src.preprocessing.caption_dataset", "--config", "configs/data/dataset.yaml"])
        run([py, "-m", "src.preprocessing.build_diffusion_dataset", "--config", "configs/data/dataset.yaml"])

    if args.stage in ["all", "stage1"]:
        if args.gan_mode == "stylegan":
            run([py, "-m", "src.stage1_gan.train_stylegan", "--config", "configs/stage1/stylegan.yaml"])
        else:
            run([py, "-m", "src.stage1_gan.train_pix2pix", "--config", "configs/stage1/pix2pix.yaml"])

    if args.stage in ["all", "stage2"]:
        run([py, "-m", "src.stage2_diffusion.train_lora", "--config", "configs/stage2/lora.yaml"])
        run([py, "-m", "src.stage2_diffusion.refine_img2img", "--config", "configs/stage2/infer.yaml"])
        run([py, "-m", "src.stage2_diffusion.refine_controlnet", "--config", "configs/stage2/controlnet.yaml"])

    if args.stage in ["all", "post"]:
        run([
            py,
            "-m",
            "src.postprocessing.enhance_texture",
            "--input_dir",
            "outputs/samples/stage2_controlnet",
            "--output_dir",
            "outputs/tiles",
            "--pbr_dir",
            "outputs/pbr",
            "--use_realesrgan",
            "--sr_scale",
            "2",
        ])

    if args.stage in ["all", "eval"]:
        run([py, "-m", "src.evaluation.evaluate_textures", "--real_dir", "data/processed/test", "--fake_dir", "outputs/tiles", "--output", "outputs/eval_report.txt"])

    if args.stage in ["all", "benchmark"]:
        run([
            py,
            "-m",
            "src.evaluation.benchmark_checkpoints",
            "--real_dir",
            "data/processed/test",
            "--candidate_roots",
            "outputs/tiles",
            "outputs/samples/stage2_refined",
            "outputs/samples/stage2_controlnet",
            "--output_csv",
            "outputs/benchmark/leaderboard.csv",
            "--output_md",
            "outputs/benchmark/leaderboard.md",
            "--auto_tick_plan",
            "--plan_path",
            "docs/PROJECT_PLAN.md",
        ])


if __name__ == "__main__":
    main()
