# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Construct minibatches for Fast R-CNN training. Handles the minibatch blobs
that are specific to Fast R-CNN. Other blobs that are generic to RPN, etc.
are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import numpy.random as npr

from detectron.core.config import cfg
import detectron.modeling.FPN as fpn
import detectron.roi_data.keypoint_rcnn as keypoint_rcnn_roi_data
import detectron.roi_data.mask_rcnn as mask_rcnn_roi_data
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
from functools import reduce

logger = logging.getLogger(__name__)
super2fine_map={1:[1,1],2:[2,9],3:[10,14],4:[15,24],5:[25,29],6:[30,39],
                7:[40,46],8:[47,56],9:[57,62],10:[63,68],11:[69,73],12:[74,80],13:[1,80]}


def get_roi_81_cls_blob_names(is_training=True):
    """super cls blob names."""
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)
    blob_names = ['roi_81_cls']
    if is_training:
        # labels_int32 blob: R categorical labels in [0, ..., K] for K
        # foreground classes plus background
        blob_names += ['labels_int32_roi_81_cls']

    if is_training and cfg.MODEL.MASK_ON and cfg.MRCNN.AT_STAGE == 1:
        # 'mask_rois': RoIs sampled for training the mask prediction branch.
        # Shape is (#masks, 5) in format (batch_idx, x1, y1, x2, y2).
        blob_names += ['mask_rois']
        # 'roi_has_mask': binary labels for the RoIs specified in 'rois'
        # indicating if each RoI has a mask or not. Note that in some cases
        # a *bg* RoI will have an all -1 (ignore) mask associated with it in
        # the case that no fg RoIs can be sampled. Shape is (batchsize).
        blob_names += ['roi_has_mask_int32']
        # 'masks_int32' holds binary masks for the RoIs specified in
        # 'mask_rois'. Shape is (#fg, M * M) where M is the ground truth
        # mask size.
        blob_names += ['masks_int32']
    if is_training and cfg.MODEL.KEYPOINTS_ON and cfg.KRCNN.AT_STAGE == 1:
        # 'keypoint_rois': RoIs sampled for training the keypoint prediction
        # branch. Shape is (#instances, 5) in format (batch_idx, x1, y1, x2,
        # y2).
        blob_names += ['keypoint_rois']
        # 'keypoint_locations_int32': index of keypoint in
        # KRCNN.HEATMAP_SIZE**2 sized array. Shape is (#instances). Used in
        # SoftmaxWithLoss.
        blob_names += ['keypoint_locations_int32']
        # 'keypoint_weights': weight assigned to each target in
        # 'keypoint_locations_int32'. Shape is (#instances). Used in
        # SoftmaxWithLoss.
        blob_names += ['keypoint_weights']
        # 'keypoint_loss_normalizer': optional normalization factor to use if
        # cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS is False.
        blob_names += ['keypoint_loss_normalizer']
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ['roi_81_cls_fpn' + str(lvl)]
        blob_names += ['roi_81_cls_idx_restore_int32']
        if is_training:
            if cfg.MODEL.MASK_ON and cfg.MRCNN.AT_STAGE == 1:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['mask_rois_fpn' + str(lvl)]
                blob_names += ['mask_rois_idx_restore_int32']
            if cfg.MODEL.KEYPOINTS_ON and cfg.KRCNN.AT_STAGE == 1:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['keypoint_rois_fpn' + str(lvl)]
                blob_names += ['keypoint_rois_idx_restore_int32']
    return blob_names


def add_roi_81_cls_blobs(blobs, rois,pred_cls_score, label=None):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    im_ids=np.unique(rois[:,0])
    for i in range(im_ids.size):
        im_i=im_ids[i]
        rois_ind=np.where((rois[:, 0]==im_i))[0]

        frcn_blobs = _sample_rois(rois[rois_ind,:],label[rois_ind],pred_cls_score[rois_ind,:])

        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois(blobs)

    # Perform any final work and validity checks after the collating blobs for
    # all minibatch images
    valid = True
    if cfg.MODEL.KEYPOINTS_ON and cfg.KRCNN.AT_STAGE == 1:
        valid = keypoint_rcnn_roi_data.finalize_keypoint_minibatch(blobs, valid)

    return valid


def _sample_rois(rois, label,pred_cls_score):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    sampled_rois=rois
    sampled_labels=label
    sampled_labels_binary=label.copy()
    sampled_labels_binary[sampled_labels>0]=1
    # remove non-vehicle object with high probability
    max_cls_label = np.argmax(pred_cls_score, axis=1)
    max_cls_score = np.max(pred_cls_score, axis=1)



    # hard negatives
    gt_neg_ind=np.where(sampled_labels_binary==0)[0]

    gt_hard_neg_ind_tmp=np.where(max_cls_label[gt_neg_ind]>=1)[0]

    gt_hard_neg_ind=gt_neg_ind[gt_hard_neg_ind_tmp]

    gt_hard_neg_ind_tmp=np.where(max_cls_score[gt_hard_neg_ind]>0.3)[0]
    gt_hard_neg_ind=gt_neg_ind[gt_hard_neg_ind_tmp]

    # normal negatives
    gt_neg_score=max_cls_score[gt_neg_ind]
    sort_ind=np.argsort(gt_neg_score)
    non_foreground_num = int(sort_ind.size * 0.7)
    gt_neg_ind=gt_neg_ind[sort_ind[:non_foreground_num]]
    gt_neg_score = max_cls_score[gt_neg_ind]
    sort_ind = np.argsort(gt_neg_score)


    # positives
    gt_pos_ind=np.where(sampled_labels_binary==1)[0]
    pos_num=gt_pos_ind.size


    neg_num=int(np.minimum(3*pos_num,gt_neg_ind.size))
    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
    if neg_num<rois_per_image-pos_num:
        neg_num=rois_per_image-pos_num

    gt_neg_ind=gt_neg_ind[sort_ind[:neg_num]]

    sel_obj_ind = reduce(np.union1d, (gt_neg_ind,gt_hard_neg_ind, gt_pos_ind))


    sampled_rois=sampled_rois[sel_obj_ind]
    sampled_labels=sampled_labels[sel_obj_ind]


    labels_int32_roi_81_cls='labels_int32_roi_81_cls'
    roi_81_cls_rois='roi_81_cls'

    blob_dict = {labels_int32_roi_81_cls:sampled_labels.astype(np.int32, copy=False),roi_81_cls_rois:sampled_rois}

    return blob_dict


def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        num_bbox_reg_classes = 2  # bg and fg

    clss = bbox_target_data[:, 0]
    bbox_targets = blob_utils.zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = blob_utils.zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind]) if not cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else 1
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


def _add_multilevel_rois(blobs):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL

    def _distribute_rois_over_fpn_levels(rois_blob_name):
        """Distribute rois over the different FPN levels."""
        # Get target level for each roi
        # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
        # the box coordinates from columns 1:5
        # print('total rois number',blobs[rois_blob_name][:, 1:5].size)
        target_lvls = fpn.map_rois_to_fpn_levels(
            blobs[rois_blob_name][:, 1:5], lvl_min, lvl_max
        )
        # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
        fpn.add_multilevel_roi_blobs(
            blobs, rois_blob_name, blobs[rois_blob_name], target_lvls, lvl_min,
            lvl_max
        )

    _distribute_rois_over_fpn_levels('roi_81_cls')
    if cfg.MODEL.MASK_ON and cfg.MRCNN.AT_STAGE == 1:
        _distribute_rois_over_fpn_levels('mask_rois')
    if cfg.MODEL.KEYPOINTS_ON and cfg.KRCNN.AT_STAGE == 1:
        _distribute_rois_over_fpn_levels('keypoint_rois')
