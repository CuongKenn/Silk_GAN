# PROJECT STRUCTURE

```text
ModelSilk/
  README.md
  docs/
    PROJECT_PLAN.md
    DATA_COLLECTION_GUIDE.md
    PROJECT_STRUCTURE.md
  data/
    README.md
    raw/
      rgb/
      masks/
      edge/
      pbr/
        normal/
        roughness/
        height/
    interim/
      cleaned/
      normalized/
      tileable/
    processed/
      train/
      val/
      test/
      captions/
    metadata/
      schema.csv
      collection_template.csv
      caption_template.csv
  src/
    stage1_gan/
    stage2_diffusion/
    preprocessing/
    postprocessing/
    evaluation/
  configs/
    stage1/
    stage2/
    data/
  notebooks/
  checkpoints/
    stage1/
    stage2/
  outputs/
    samples/
    tiles/
    pbr/
```

Ghi chu:
- `raw/` la data goc ban thu thap.
- `interim/` la data sau cleaning/normalization.
- `processed/` la data da split va san sang train.
