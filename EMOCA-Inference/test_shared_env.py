from __future__ import annotations

import importlib
import sys
from pathlib import Path


REQUIRED_MODULES = [
    "numpy",
    "torch",
    "torchvision",
    "torchaudio",
    "timm",
    "pytorch_lightning",
    "torchmetrics",
    "cv2",
    "mediapipe",
    "trimesh",
    "smplx",
    "ultralytics",
]


def check_import(name: str) -> None:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"[OK] {name}: {version}")


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 10):
        print("[WARN] Shared environment was designed for Python 3.10.")

    for module_name in REQUIRED_MODULES:
        check_import(module_name)

    from gdl.models.Swin import EXTERNAL_SWIN_AVAILABLE, create_swin_backbone, swin_cfg_from_name

    cfg = swin_cfg_from_name("swin_tiny_patch4_window7_224")
    model = create_swin_backbone(
        cfg,
        num_classes=10,
        img_size=224,
        load_pretrained_swin=False,
        pretrained_model=None,
    )
    print(f"[OK] EMOCA Swin fallback works. External repo available: {EXTERNAL_SWIN_AVAILABLE}")
    print(f"[OK] Swin type: {type(model)}")

    sibling_root = Path(__file__).resolve().parents[1] / "SMPLest-X-Inference"
    if sibling_root.is_dir():
        sys.path.insert(0, str(sibling_root))
        from human_models.human_models import SMPLX  # noqa: F401
        from ultralytics import YOLO  # noqa: F401

        print("[OK] SMPLest-X core imports succeeded.")
    else:
        print(f"[WARN] SMPLest-X-Inference not found at {sibling_root}")


if __name__ == "__main__":
    main()

