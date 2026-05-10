"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""


import os, sys
from pathlib import Path
from gdl.utils.other import get_path_to_externals
try:
    from SwinTransformer.models.build import build_model
    EXTERNAL_SWIN_AVAILABLE = True
except ImportError:
    import timm
    EXTERNAL_SWIN_AVAILABLE = False
from omegaconf import open_dict, OmegaConf
import torch


def swin_cfg_from_name(name):
    swin_path = (get_path_to_externals() / "SwinTransformer").absolute()
    if not swin_path.is_dir():
        # Fallback to a minimal config if external is missing
        return OmegaConf.create({"MODEL": {"SWIN": {"TYPE": name}}})
    swin_cfg = OmegaConf.load(
        Path(swin_path) / "configs" / (name + ".yaml"))
    OmegaConf.set_struct(swin_cfg, True)
    return swin_cfg


def create_swin_backbone(swin_cfg, num_classes, img_size, load_pretrained_swin=False, pretrained_model=None):
    """
    Returns a SWIN backbone with a head of size num_classes.
    """
    if not EXTERNAL_SWIN_AVAILABLE:
        print(f"SwinTransformer external repo not found. Using timm for model: {pretrained_model}")
        # mapping model names to timm names
        timm_name = pretrained_model or "swin_tiny_patch4_window7_224"
        model = timm.create_model(timm_name, pretrained=load_pretrained_swin, num_classes=num_classes)
        return model

    with open_dict(swin_cfg):
        swin_cfg.MODEL.NUM_CLASSES = num_classes
        # ... (rest of configuration)
        swin_cfg.MODEL.SWIN.PATCH_SIZE = 4
        swin_cfg.MODEL.SWIN.IN_CHANS = 3
        swin_cfg.MODEL.SWIN.MLP_RATIO = 4.
        swin_cfg.MODEL.SWIN.QKV_BIAS = True
        swin_cfg.MODEL.SWIN.QK_SCALE = None
        swin_cfg.MODEL.SWIN.APE = False
        swin_cfg.MODEL.SWIN.PATCH_NORM = True

        if 'DROP_RATE' not in swin_cfg.MODEL.keys():
            swin_cfg.MODEL.DROP_RATE = 0.0
        if 'DROP_PATH_RATE' not in swin_cfg.MODEL.keys():
            swin_cfg.MODEL.DROP_PATH_RATE = 0.1
        if 'LABEL_SMOOTHING' not in swin_cfg.MODEL.keys():
            swin_cfg.MODEL.LABEL_SMOOTHING = 0.1

        swin_cfg.DATA = {}
        swin_cfg.DATA.IMG_SIZE = img_size
        swin_cfg.TRAIN = {}
        swin_cfg.TRAIN.USE_CHECKPOINT = False

    swin = build_model(swin_cfg)

    if load_pretrained_swin:
        swin_path = (get_path_to_externals() / "SwinTransformer").absolute()
        path_to_model = swin_path / "pretrained_models" / (
                    pretrained_model + ".pth")
        if path_to_model.exists():
            state_dict = torch.load(path_to_model)
            del state_dict['model']['head.weight']
            del state_dict['model']['head.bias']
            swin.load_state_dict(state_dict['model'], strict=False)
            print(f"Loading pretrained model from '{path_to_model}'")
        else:
            print(f"Pretrained model file not found: {path_to_model}")

    return swin