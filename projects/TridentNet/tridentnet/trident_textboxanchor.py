# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from detectron2.modeling import ANCHOR_GENERATOR_REGISTRY
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
import math


@ANCHOR_GENERATOR_REGISTRY.register()
class TextBoxesppAnchorGenerator(DefaultAnchorGenerator):
    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0 - h / 2, x1, y1 - h / 2])
                anchors.append([x0, y0 + h / 2, x1, y1 + h / 2])
        return torch.tensor(anchors)

