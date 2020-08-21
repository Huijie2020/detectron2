# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.layers import batched_nms, ShapeSpec
## from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads, ROIHeads, build_mask_head, make_stage
from detectron2.modeling.backbone.resnet import BottleneckBlock
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals
import torch
from torch import nn
from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads
from detectron2.structures import Instances


def merge_branch_instances(instances, num_branch, nms_thresh, topk_per_image):
    """
    Merge detection results from different branches of TridentNet.
    Return detection results by applying non-maximum suppression (NMS) on bounding boxes
    and keep the unsuppressed boxes and other instances (e.g mask) if any.

    Args:
        instances (list[Instances]): A list of N * num_branch instances that store detection
            results. Contain N images and each image has num_branch instances.
        num_branch (int): Number of branches used for merging detection results for each image.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        results: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections after merging results from multiple
            branches.
    """
    if num_branch == 1:
        return instances

    batch_size = len(instances) // num_branch
    results = []
    for i in range(batch_size):
        instance = Instances.cat([instances[i + batch_size * j] for j in range(num_branch)])

        # Apply per-class NMS
        keep = batched_nms(
            instance.pred_boxes.tensor, instance.scores, instance.pred_classes, nms_thresh
        )
        keep = keep[:topk_per_image]
        result = instance[keep]

        results.append(result)

    return results

@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsHalf(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg)

        # fmt: off
        self.in_features  = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(self.in_features) == 1

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

    def _build_res5_block(self, cfg):
        # fmt: off
        ## stage_channel_factor = 2 ** 3  # res5 is 8x res2
        # stage_channel_factor = 2  # res5 is 8x res2
        stage_channel_factor = 2 ** 2 # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances


@ROI_HEADS_REGISTRY.register()
## class TridentRes5ROIHeads(Res5ROIHeads):
class TridentRes5ROIHeads(Res5ROIHeadsHalf):
    """
    The TridentNet ROIHeads in a typical "C4" R-CNN model.
    See :class:`Res5ROIHeads`.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.num_branch = cfg.MODEL.TRIDENT.NUM_BRANCH
        self.trident_fast = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX != -1

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`Res5ROIHeads.forward`.
        """
        num_branch = self.num_branch if self.training or not self.trident_fast else 1
        all_targets = targets * num_branch if targets is not None else None
        pred_instances, losses = super().forward(images, features, proposals, all_targets)
        del images, all_targets, targets

        if self.training:
            return pred_instances, losses
        else:
            pred_instances = merge_branch_instances(
                pred_instances,
                num_branch,
                self.box_predictor.test_nms_thresh,
                self.box_predictor.test_topk_per_image,
            )

            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class TridentStandardROIHeads(StandardROIHeads):
    """
    The `StandardROIHeads` for TridentNet.
    See :class:`StandardROIHeads`.
    """

    def __init__(self, cfg, input_shape):
        super(TridentStandardROIHeads, self).__init__(cfg, input_shape)

        self.num_branch = cfg.MODEL.TRIDENT.NUM_BRANCH
        self.trident_fast = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX != -1

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`Res5ROIHeads.forward`.
        """
        # Use 1 branch if using trident_fast during inference.
        num_branch = self.num_branch if self.training or not self.trident_fast else 1
        # Duplicate targets for all branches in TridentNet.
        all_targets = targets * num_branch if targets is not None else None
        pred_instances, losses = super().forward(images, features, proposals, all_targets)
        del images, all_targets, targets

        if self.training:
            return pred_instances, losses
        else:
            pred_instances = merge_branch_instances(
                pred_instances,
                num_branch,
                self.box_predictor.test_nms_thresh,
                self.box_predictor.test_topk_per_image,
            )

            return pred_instances, {}
