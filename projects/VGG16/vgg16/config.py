# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_vgg_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.VGG = CN()

    _C.MODEL.VGG.DEPTH = 16
    _C.MODEL.VGG.OUT_FEATURES = ["vgg_block4"]

    # Options: FrozenBN, GN, "SyncBN", "BN"
    _C.MODEL.VGG.NORM = "FrozenBN"

    # Output channels of conv5 block
    _C.MODEL.VGG.CONV5_OUT_CHANNELS = 512
