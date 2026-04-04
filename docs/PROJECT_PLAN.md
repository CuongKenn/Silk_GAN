# PROJECT PLAN - Lace Texture Pipeline (GAN + Diffusion)

## Muc tieu
Xay dung pipeline end-to-end de sinh texture ren chat luong cao, tileable, chi tiet soi vai ro net.

## Checklist Tong

### Phase 0 - Setup du an
- [x] Tao cau truc thu muc du an
- [x] Tao docs ke hoach va huong dan thu thap data
- [x] Tao template metadata de thu thap data dong nhat

### Phase 1 - Data pipeline
- [ ] Thu thap anh RGB ren theo dung schema
- [ ] Thu thap du lieu bo sung (mask/edge/PBR) neu co
- [x] Chay data cleaning (blur, duplicate, artifact) - code san
- [x] Chuan hoa kich thuoc va anh sang - code san
- [x] Tao train/val/test split - code san
- [x] Tao caption tu dong cho diffusion - code san

### Phase 2 - Stage 1 (GAN)
- [ ] Chon huong A (StyleGAN) hoac B (Pix2Pix)
- [x] Train baseline - code san
- [x] Danh gia FID + tile consistency - script eval san
- [ ] Chon checkpoint tot nhat

### Phase 3 - Stage 2 (Diffusion)
- [x] Fine-tune LoRA/DreamBooth tren texture ren - LoRA pipeline san
- [x] Tich hop img2img tu output GAN - code san
- [x] Them conditioning bang edge/ControlNet neu can - code san
- [x] Tune sampling de toi uu realism - config infer san

### Phase 4 - Post-process
- [x] Sharpen nhe
- [x] Super-resolution (Real-ESRGAN) - code san
- [x] Sinh PBR maps (normal/roughness/height)

### Phase 5 - Evaluation + release
- [x] Danh gia FID/LPIPS/SSIM - script eval san
- [x] Kiem tra tile seam (3x3 repeat test) - seam score san
- [ ] Chot bo model + infer pipeline

---

## Cac buoc da thuc hien trong yeu cau nay

### Step 1 - Dung bo docs + structure
Trang thai: [x] Done

Da lam:
- Tao docs ke hoach co checklist va tick
- Tao docs huong dan data collection
- Tao docs mo ta cau truc thu muc
- Tao san cay thu muc data de ban bo du lieu vao
- Tao template metadata mau

### Step 2 - Chua code chi tiet
Trang thai: [x] Done theo yeu cau hien tai

Ghi chu:
- Theo yeu cau cua ban, hien tai chi dung skeleton va docs.
- Chua viet train code va preprocess code chi tiet.

### Step 3 - Tao code pipeline core (skeleton co the chay)
Trang thai: [x] Done

Da lam:
- Tao preprocessing script: clean + normalize + tileable + split
- Tao captioning script tu dong cho diffusion
- Tao train Stage 1 cho 2 huong: StyleGAN baseline va Pix2Pix baseline
- Tao train Stage 2 LoRA launcher + refine img2img
- Tao postprocess script: sharpen + tao normal/roughness/height
- Tao evaluation script: FID/LPIPS/SSIM + seam score
- Tao file config cho data, stage1, stage2
- Tao orchestration script chay tung stage hoac all stage

Ghi chu:
- Chua train that vi ban se thu thap data sau.

### Step 4 - Hoan thien pipeline chay train khi co data
Trang thai: [x] Done

Da lam:
- Them script tao edge map tu data tileable cho Pix2Pix/conditioning
- Them script tao metadata.jsonl cho diffusion LoRA
- Cap nhat run_pipeline de chay du cac buoc data
- Tang do ben cho evaluate: ghep cap anh theo filename
- Cap nhat README voi quick start command

### Step 5 - Nang cap chat luong production
Trang thai: [x] Done

Da lam:
- Them super-resolution Real-ESRGAN vao post-processing (co fallback)
- Them Stage 2 ControlNet Canny refinement de giu cau truc ren
- Them benchmark tu dong nhieu output/checkpoint va tu tick plan khi chon best

### Step 6 - Tich hop StyleGAN2-ADA chuan cho Stage 1
Trang thai: [x] Done

Da lam:
- Thay Stage 1 StyleGAN baseline bang launcher NVLabs StyleGAN2-ADA
- Tu dong tao dataset zip bang dataset_tool.py tu data/processed/train
- Ho tro auto clone repo StyleGAN2-ADA vao third_party/
- Cap nhat config train StyleGAN sang tham so chuan (kimg, gamma, ada, metrics)
