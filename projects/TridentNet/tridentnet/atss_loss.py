import torch
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from atss_core.layers import SigmoidFocalLoss
from atss_core.modeling.matcher import Matcher
from atss_core.structures.boxlist_ops import boxlist_iou
from atss_core.structures.boxlist_ops import cat_boxlist


INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class ATSSLossComputation(object):

    def __init__(self, cfg, box_coder):
        self.cfg = cfg
        #self.cls_loss_func = SigmoidFocalLoss(cfg.MODEL.ATSS.LOSS_GAMMA, cfg.MODEL.ATSS.LOSS_ALPHA)
        #self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.matcher = Matcher(cfg.MODEL.ATSS.FG_IOU_THRESHOLD, cfg.MODEL.ATSS.BG_IOU_THRESHOLD, True)
        self.box_coder = box_coder

    def prepare_targets(self, targets, anchors):
        cls_labels = []
        reg_targets = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes_per_im = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            anchors_per_im = cat_boxlist(anchors[im_i])
            num_gt = bboxes_per_im.shape[0]

            if self.cfg.MODEL.ATSS.POSITIVE_TYPE == 'ATSS':
                num_anchors_per_loc = len(self.cfg.MODEL.ATSS.ASPECT_RATIOS) * self.cfg.MODEL.ATSS.SCALES_PER_OCTAVE

                num_anchors_per_level = [len(anchors_per_level.bbox) for anchors_per_level in anchors[im_i]]
                ious = boxlist_iou(anchors_per_im, targets_per_im)

                gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
                gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
                gt_points = torch.stack((gt_cx, gt_cy), dim=1)

                anchors_cx_per_im = (anchors_per_im.bbox[:, 2] + anchors_per_im.bbox[:, 0]) / 2.0
                anchors_cy_per_im = (anchors_per_im.bbox[:, 3] + anchors_per_im.bbox[:, 1]) / 2.0
                anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

                distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

                # Selecting candidates based on the center distance between anchor box and object
                candidate_idxs = []
                star_idx = 0
                for level, anchors_per_level in enumerate(anchors[im_i]):
                    end_idx = star_idx + num_anchors_per_level[level]
                    distances_per_level = distances[star_idx:end_idx, :]
                    topk = min(self.cfg.MODEL.ATSS.TOPK * num_anchors_per_loc, num_anchors_per_level[level])
                    _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                    candidate_idxs.append(topk_idxs_per_level + star_idx)
                    star_idx = end_idx
                candidate_idxs = torch.cat(candidate_idxs, dim=0)

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
                l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 0]
                t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
                r = bboxes_per_im[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
                b = bboxes_per_im[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
                is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                is_pos = is_pos & is_in_gts

                # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
                ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
                index = candidate_idxs.view(-1)[is_pos.view(-1)]
                ious_inf[index] = ious.t().contiguous().view(-1)[index]
                ious_inf = ious_inf.view(num_gt, -1).t()

                anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
                cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
                cls_labels_per_im[anchors_to_gt_values == -INF] = 0
                matched_gts = bboxes_per_im[anchors_to_gt_indexs]
            else:
                raise NotImplementedError

            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return cls_labels, reg_targets

    def __call__(self, box_cls, box_regression, centerness, targets, anchors):
        labels, reg_targets = self.prepare_targets(targets, anchors)

        N = len(labels)
        box_cls_flatten, box_regression_flatten = concat_box_prediction_layers(box_cls, box_regression)
        centerness_flatten = [ct.permute(0, 2, 3, 1).reshape(N, -1, 1) for ct in centerness]
        centerness_flatten = torch.cat(centerness_flatten, dim=1).reshape(-1)

        labels_flatten = torch.cat(labels, dim=0)
        reg_targets_flatten = torch.cat(reg_targets, dim=0)
        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).bbox for anchors_per_image in anchors], dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten.int()) / num_pos_avg_per_gpu

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        anchors_flatten = anchors_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        centerness_targets = self.compute_centerness_targets(reg_targets_flatten, anchors_flatten)
        sum_centerness_targets_avg_per_gpu = reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

        if pos_inds.numel() > 0:
            reg_loss = self.GIoULoss(box_regression_flatten, reg_targets_flatten, anchors_flatten,
                                     weight=centerness_targets) / sum_centerness_targets_avg_per_gpu
            centerness_loss = self.centerness_loss_func(centerness_flatten, centerness_targets) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss * self.cfg.MODEL.ATSS.REG_LOSS_WEIGHT, centerness_loss


def make_atss_loss_evaluator(cfg, box_coder):
    loss_evaluator = ATSSLossComputation(cfg, box_coder)
    return loss_evaluator
