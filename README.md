# ModelSilk - Lace Texture Generation Project

Du an nay dung kien truc 2 giai doan:
- Stage 1: GAN sinh cau truc hoa van ren (StyleGAN hoac Pix2Pix)
- Stage 2: Diffusion tinh chinh chi tiet vi mo, tang realism

Tai lieu chinh:
- docs/PROJECT_PLAN.md: Ke hoach theo buoc + checklist
- docs/DATA_COLLECTION_GUIDE.md: Huong dan thu thap va chuan hoa du lieu
- docs/PROJECT_STRUCTURE.md: So do cau truc thu muc du an

## Quick start

1. Cai thu vien:
	pip install -r requirements.txt

2. Bo data RGB goc vao:
	data/raw/rgb

3. Chay data pipeline:
	python -m src.run_pipeline --stage data

4. Chay Stage 1 GAN:
	python -m src.run_pipeline --stage stage1 --gan_mode stylegan

5. Chay Stage 2 diffusion:
	python -m src.run_pipeline --stage stage2

6. Chay post-process + evaluate:
	python -m src.run_pipeline --stage post
	python -m src.run_pipeline --stage eval

7. Chay benchmark va auto tick plan:
	python -m src.run_pipeline --stage benchmark

## Ghi chu

- Neu muon train Pix2Pix, data edge se duoc tao o data/raw/edge khi chay stage data.
- LoRA diffusion can file metadata.jsonl trong data/processed/train; script stage data se tao san.
- Stage 2 co them ControlNet Canny qua configs/stage2/controlnet.yaml.
- Post-process ho tro Real-ESRGAN (co fallback khi chua cai package/weights).
- Stage 1 StyleGAN da dung NVLabs StyleGAN2-ADA launcher trong src/stage1_gan/train_stylegan.py.
- Lan chay Stage 1 StyleGAN dau tien se auto clone repo third_party/stylegan2-ada-pytorch (can git).

## Run nhanh

1. Cai dependencies:
	- `pip install -r requirements.txt`

2. Chay preprocessing + caption:
	- `python -m src.preprocessing.prepare_dataset --config configs/data/dataset.yaml`
	- `python -m src.preprocessing.caption_dataset --config configs/data/dataset.yaml`

3. Train Stage 1:
	- StyleGAN2-ADA: `python -m src.stage1_gan.train_stylegan --config configs/stage1/stylegan.yaml`
	- Pix2Pix: `python -m src.stage1_gan.train_pix2pix --config configs/stage1/pix2pix.yaml`

4. Stage 2 diffusion:
	- `python -m src.stage2_diffusion.train_lora --config configs/stage2/lora.yaml`
	- `python -m src.stage2_diffusion.refine_img2img --config configs/stage2/infer.yaml`

5. Post-process + evaluate:
	- `python -m src.postprocessing.enhance_texture --input_dir outputs/samples/stage2_refined --output_dir outputs/tiles --pbr_dir outputs/pbr`
	- `python -m src.evaluation.evaluate_textures --real_dir data/processed/test --fake_dir outputs/tiles --output outputs/eval_report.txt`

6. Orchestration 1 lenh:
	- `python -m src.run_pipeline --stage all --gan_mode stylegan`
