import sys
import torch
import numpy as np

def test_numpy_compatibility():
    print("Testing NumPy compatibility...")
    try:
        # These are the ones that usually break in 1.24+
        # If this script runs in 3.10 with modern numpy, it should handle standard types
        _ = np.array([1, 2, 3], dtype=int)
        _ = np.array([True, False], dtype=bool)
        print(" [OK] NumPy types are valid.")
    except Exception as e:
        print(f" [FAIL] NumPy issue: {e}")

def test_gdl_imports():
    print("Testing GDL imports...")
    try:
        import gdl
        from gdl.models.Swin import EXTERNAL_SWIN_AVAILABLE
        print(f" [OK] GDL imported. External Swin Available: {EXTERNAL_SWIN_AVAILABLE}")
    except ImportError as e:
        print(f" [FAIL] GDL is not installed or dependencies missing: {e}")
    except Exception as e:
        print(f" [FAIL] Unexpected error: {e}")

def test_swin_logic():
    print("Testing Swin creation logic...")
    try:
        from gdl.models.Swin import create_swin_backbone
        from omegaconf import OmegaConf
        
        # Dummy config
        cfg = OmegaConf.create({"MODEL": {"SWIN": {"TYPE": "swin_tiny_patch4_window7_224"}}})
        # Try to create a backbone (CPU for testing)
        # Note: If timm is installed, this should work even if external repo is missing
        model = create_swin_backbone(cfg, num_classes=50, img_size=224, load_pretrained_swin=False)
        print(f" [OK] Swin backbone created successfully with type: {type(model)}")
    except Exception as e:
        print(f" [FAIL] Swin creation failed: {e}")

if __name__ == "__main__":
    print(f"Python Version: {sys.version}")
    test_numpy_compatibility()
    test_gdl_imports()
    test_swin_logic()
