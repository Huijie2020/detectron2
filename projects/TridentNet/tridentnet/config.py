# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_tridentnet_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.TRIDENT = CN()

    # Number of branches for TridentNet.
    _C.MODEL.TRIDENT.NUM_BRANCH = 3
    # Specify the dilations for each branch.
    _C.MODEL.TRIDENT.BRANCH_DILATIONS = [1, 2, 3]
    # Specify the stage for applying trident blocks. Default stage is Res4 according to the
    # TridentNet paper.
    _C.MODEL.TRIDENT.TRIDENT_STAGE = "res4"
    # Specify the test branch index TridentNet Fast inference:
    #   - use -1 to aggregate results of all branches during inference.
    #   - otherwise, only using specified branch for fast inference. Recommended setting is
    #     to use the middle branch.
    _C.MODEL.TRIDENT.TEST_BRANCH_IDX = 1

def add_atss_config(cfg):
    """
    Add config for atss.
    """
    _C = cfg

    _C.MODEL.ATSS_ON = False

    _C.MODEL.ATSS = CN()
    _C.MODEL.ATSS.NUM_CLASSES = 4  # the number of classes including background

    # Anchor parameter
    # _C.MODEL.ATSS.ANCHOR_SIZES = (64, 128, 256, 512, 1024)
    # _C.MODEL.ATSS.ASPECT_RATIOS = (1.0,)
    # _C.MODEL.ATSS.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
    # _C.MODEL.ATSS.STRADDLE_THRESH = 0
    # _C.MODEL.ATSS.OCTAVE = 2.0
    # _C.MODEL.ATSS.SCALES_PER_OCTAVE = 1

    # Head parameter
    # _C.MODEL.ATSS.NUM_CONVS = 4
    # _C.MODEL.ATSS.USE_DCN_IN_TOWER = False

    # Focal loss parameter
    # _C.MODEL.ATSS.LOSS_ALPHA = 0.25
    # _C.MODEL.ATSS.LOSS_GAMMA = 2.0

    # how to select positves: ATSS (Ours) , SSC (FCOS), IoU (RetinaNet), TOPK
    _C.MODEL.ATSS.POSITIVE_TYPE = 'ATSS'
    _C.MODEL.ATSS.TOPK = 9

    # IoU parameter to select positves
    # _C.MODEL.ATSS.FG_IOU_THRESHOLD = 0.5
    # _C.MODEL.ATSS.BG_IOU_THRESHOLD = 0.4