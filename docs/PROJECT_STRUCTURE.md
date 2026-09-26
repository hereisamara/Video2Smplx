# Project Structure

The repository root is reserved for project entry documentation, the three
external inference repositories, the core Python package, and major project
areas. Generated outputs and specialist commands are kept out of the root.

```text
Video2Smplx/
  README*.md                  Delivery and project entry documentation
  pipeline.py                 Legacy file-based pipeline entry point
  smplestx_wilor_emoca_fuse.py
  zero_filter_render.py       Legacy core pipeline support
  video2smplx/                Integrated runtime package
  tools/
    postprocessing/           Adopted learned-correction CLIs
    preprocessing/            YOLO pose extraction
    evaluation/               Geometry and benchmark metrics
    diagnostics/              Error-source analysis
    visualization/            GIF, mesh, and report rendering
  delivery/                   Customer-facing code, docs, scripts, and samples
  research/                   Training and non-adopted experiments
  requirements/               Optional dependency groups
  docs/                       Project documentation and media
  tests/                      Automated tests
  artifacts/                  Generated packages, results, and handover copies
  tmp/                        Local samples and backups
  SMPLest-X-Inference/        Body estimator dependency
  WiLoR-Inference/            Hand estimator dependency
  EMOCA-Inference/            Face estimator dependency
```

## Delivery Boundary

The delivery runtime uses `delivery/`, `video2smplx/`, and the operational
subdirectories under `tools/`. Training and GT-oracle code remains under
`research/signlanguage_training/` for reproducibility and is excluded from
document-set and flash-drive packaging.

## Generated Material

`artifacts/`, `tmp/`, Python caches, and macOS metadata are ignored by Git.
The delivery document-set builder writes to `artifacts/delivery_packages/` by
default.
