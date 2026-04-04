# DATA COLLECTION GUIDE

## 1) Ban can thu thap data gi?

## A. Bat buoc
1. RGB lace texture images
- Dinh dang: PNG (uu tien) hoac JPG chat luong cao
- Do phan giai khuyen nghi: >= 2048x2048 (cang cao cang tot)
- Goc chup: camera vuong goc mat vai, tranh perspective nghieng

## B. Nen co them (neu lam duoc)
1. Binary mask (vung ren vs nen)
- Dinh dang: PNG 1 kenh (0/255)

2. Edge map
- Dinh dang: PNG 1 kenh
- Co the tao sau bang Canny/HED

3. PBR maps (neu co)
- normal: PNG 3 kenh
- roughness: PNG 1 kenh
- height/displacement: PNG 1 kenh

## C. Metadata bat buoc
Moi anh can metadata trong file CSV:
- image_id
- file_name
- split (train/val/test)
- lace_style (floral/geometric/guipure/...)
- color_family
- source
- license
- lighting
- notes

---

## 2) Tieu chuan chat luong khi thu thap
1. Khong mo nhoe, khong out-of-focus
2. Khong watermark/logo/text lon tren be mat vai
3. It bong do manh, it diem chay sang
4. Nhieu mau hoa van ren khac nhau de tranh overfit
5. Co ca vung texture day va vung lo ren nho

---

## 3) Quy tac dat ten file
Mau:
- lace_000001.png
- lace_000001_mask.png
- lace_000001_normal.png

Quy tac:
1. ID giu dong nhat giua RGB va map phu
2. Khong dung dau cach, uu tien chu thuong + underscore

---

## 4) Cau truc data trong du an
Xem chi tiet tai docs/PROJECT_STRUCTURE.md.

Tom tat:
1. data/raw/rgb: anh RGB goc
2. data/raw/masks: mask neu co
3. data/raw/edge: edge neu co
4. data/raw/pbr: normal/roughness/height neu co
5. data/metadata: file CSV metadata

---

## 5) So luong data khuyen nghi
1. Muc toi thieu chay thu: 2,000-3,000 anh
2. Muc tot de train on dinh: 10,000+ patches
3. Muc rat tot cho chat luong cao: 30,000+ patches da clean

---

## 6) Prompt/caption cho diffusion
Ban can co caption text cho moi image (sinh tu dong sau cung duoc), vi du:
"macro photo of floral lace textile, intricate threads, perforated pattern, seamless texture"
