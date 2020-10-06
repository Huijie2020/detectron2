import torch
from torch import nn
import os
import numpy
# from ..utils import concat_box_prediction_layers
# from atss_core.layers import SigmoidFocalLoss
# from atss_core.modeling.matcher import Matcher
# from atss_core.structures.boxlist_ops import boxlist_iou
# from atss_core.structures.boxlist_ops import cat_boxlist

from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY
# from detectron2.modeling.proposal_generator.rpn import RPN
# from detectron2.structures import ImageList
from .trident_rpn import TridentRPN
from .atss_subsample import atss_subsample
from detectron2.structures import Boxes, pairwise_iou

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from typing import Dict, List, Optional, Tuple, Union

INF = 100000000


# def get_num_gpus():
#     return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
#
#
# def reduce_sum(tensor):
#     if get_num_gpus() <= 1:
#         return tensor
#     import torch.distributed as dist
#     tensor = tensor.clone()
#     dist.all_reduce(tensor, op=dist.reduce_op.SUM)
#     return tensor


@PROPOSAL_GENERATOR_REGISTRY.register()
class ATSSRPN(TridentRPN):
    """
      ATSS RPN subnetwork.
     """
    def __init__(self, cfg, input_shape):
        super(ATSSRPN, self).__init__(cfg, input_shape)

        self.atss_positive_type = cfg.MODEL.ATSS.POSITIVE_TYPE
        self.atss_topk = cfg.MODEL.ATSS.TOPK

        self.anchor_generator_size = cfg.MODEL.ANCHOR_GENERATOR.SIZES
        self.aspect_ratios = cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS

        #self.atss_aspect_ratios = cfg.MODEL.ATSS.ASPECT_RATIOS
        #self.atss_scale_per_octave = cfg.MODEL.ATSS.SCALES_PER_OCTAVE
        #self.atss_reg_loss_weight = cfg.MODEL.ATSS.REG_LOSS_WEIGHT

        #self.cfg = cfg
        #self.cls_loss_func = SigmoidFocalLoss(cfg.MODEL.ATSS.LOSS_GAMMA, cfg.MODEL.ATSS.LOSS_ALPHA)
        #self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        #self.matcher = Matcher(cfg.MODEL.ATSS.FG_IOU_THRESHOLD, cfg.MODEL.ATSS.BG_IOU_THRESHOLD, True)
        #self.box_coder = box_coder

    def _atss_subsample(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
         """
        pos_idx, neg_idx = atss_subsample(
            label, self.batch_size_per_image, self.positive_fraction, -1
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.no_grad()
    def label_and_sample_anchors(self, anchors, targets):
    # def label_and_sample_anchors(
    #         self, anchors: List[Boxes], targets: List[Instances]
    # ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        anchors_per_im = Boxes.cat(anchors)
        cls_labels = []
        matched_gts = []
        #reg_targets = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            #assert targets_per_im.mode == "xyxy"
            #bboxes_per_im = targets_per_im.bbox
            bboxes_per_im = targets_per_im.gt_boxes
            #labels_per_im = targets_per_im.get_field("labels")
            # labels_per_im = torch.tensor([1, 1, 1])
            labels_per_im = torch.ones(len(bboxes_per_im))
            #anchors_per_im = cat_boxlist(anchors[im_i])

            # num_gt = bboxes_per_im.shape[0]
            num_gt = len(bboxes_per_im)

            if self.atss_positive_type == 'ATSS':
                # num_anchors_per_loc = len(self.cfg.MODEL.ATSS.ASPECT_RATIOS) * self.cfg.MODEL.ATSS.SCALES_PER_OCTAVE
                num_anchors_per_loc = len(self.anchor_generator_size[0]) * len(self.aspect_ratios[0])
                # num_anchors_per_level = [len(anchors_per_level.bbox) for anchors_per_level in anchors[im_i]]
                # ious = boxlist_iou(anchors_per_im, targets_per_im)
                # num_anchors_per_level = [len(anchors_per_level.gt_boxes) for anchors_per_level in anchors[im_i]]
                num_anchors_per_level = [len(anchors_per_level) for anchors_per_level in anchors]
                ious = pairwise_iou(anchors_per_im, bboxes_per_im)
                #print(bboxes_per_im)
                # gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
                # gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
                gt_cx = (bboxes_per_im.tensor[:, 2] + bboxes_per_im.tensor[:, 0]) / 2.0
                gt_cy = (bboxes_per_im.tensor[:, 3] + bboxes_per_im.tensor[:, 1]) / 2.0
                gt_points = torch.stack((gt_cx, gt_cy), dim=1)

                # anchors_cx_per_im = (anchors_per_im.bbox[:, 2] + anchors_per_im.bbox[:, 0]) / 2.0
                # anchors_cy_per_im = (anchors_per_im.bbox[:, 3] + anchors_per_im.bbox[:, 1]) / 2.0
                anchors_cx_per_im = (anchors_per_im.tensor[:, 2] + anchors_per_im.tensor[:, 0]) / 2.0
                anchors_cy_per_im = (anchors_per_im.tensor[:, 3] + anchors_per_im.tensor[:, 1]) / 2.0
                anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

                distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

                # Selecting candidates based on the center distance between anchor box and object
                candidate_idxs = []
                star_idx = 0
                for level, anchors_per_level in enumerate(anchors):
                    end_idx = star_idx + num_anchors_per_level[level]
                    distances_per_level = distances[star_idx:end_idx, :]
                    #topk = min(self.atss_topk * num_anchors_per_loc, num_anchors_per_level[level])
                    topk = min(self.atss_topk, num_anchors_per_level[level])
                    _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                    candidate_idxs.append(topk_idxs_per_level + star_idx)
                    star_idx = end_idx
                candidate_idxs = torch.cat(candidate_idxs, dim=0)
                #print("candidate anchor")
                #print(anchors_per_im[candidate_idxs[:,0]])
                #print("size")
                #print(Boxes.area(anchors_per_im[candidate_idxs[:,0]]).sqrt())
                # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
                candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
                iou_mean_per_gt = candidate_ious.mean(0)
                iou_std_per_gt = candidate_ious.std(0)
                iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
                is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

                # Limiting the final positive samples’ center to object
                anchor_num = anchors_cx_per_im.shape[0]
                for ng in range(num_gt):
                    candidate_idxs[:, ng] += ng * anchor_num
                e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                candidate_idxs = candidate_idxs.view(-1)
                # l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 0]
                # t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
                # r = bboxes_per_im[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
                # b = bboxes_per_im[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
                l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im.tensor[:, 0]
                t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im.tensor[:, 1]
                r = bboxes_per_im.tensor[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
                b = bboxes_per_im.tensor[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
                is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                is_pos = is_pos & is_in_gts

                # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
                ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
                index = candidate_idxs.view(-1)[is_pos.view(-1)]
                ious_inf[index] = ious.t().contiguous().view(-1)[index]
                ious_inf = ious_inf.view(num_gt, -1).t()

                anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
                cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
                # cls_labels_per_im[anchors_to_gt_values == -INF] = 0
                cls_labels_per_im[anchors_to_gt_values == -INF] = -1
                cls_labels_per_im = self._atss_subsample(cls_labels_per_im)
                matched_gts_per_im = bboxes_per_im[anchors_to_gt_indexs]
                matched_gts_per_im = matched_gts_per_im.tensor
                cls_labels_per_im = cls_labels_per_im.to(device=matched_gts_per_im.device)
                # print("matched_gts_per_im")
                # print(matched_gts_per_im)
            else:
                raise NotImplementedError

            #reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            #reg_targets.append(reg_targets_per_im)
            matched_gts.append(matched_gts_per_im)
            #matched_gts.append(Boxes(matched_gts_per_im))
        # print("gt_labels")
        # print(cls_labels)
        # print("matched_gt_boxes")
        # print(matched_gts)
        #return cls_labels, reg_targets
        return cls_labels, matched_gts

    # def forward(self, images, features, gt_instances=None):
    #     """
    #     See :class:`RPN.forward`.
    #     """
    #
    #     #anchors = self.anchor_generator(features)
    #
    #     #pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
    #
    #     # if self.training:
    #     #     assert gt_instances is not None, "RPN requires gt_instances in training!"
    #     #     # To delete
    #     #     gt_labels, gt_boxes = self.prepare_targets(anchors, gt_instances)
    #     #     losses = self.losses(
    #     #         anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
    #     #     )
    #     # else:
    #     #     losses = {}
    #     print(len(gt_instances))
    #     return TridentRPN(images, features, gt_instances)
