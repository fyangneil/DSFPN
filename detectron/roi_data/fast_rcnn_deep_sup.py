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
import detectron.roi_data.mask_rcnn_deep_sup as mask_rcnn_roi_deep_sup_data
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
from functools import reduce

logger = logging.getLogger(__name__)


def get_roi_deep_sup_blob_names(is_training=True):
    """super cls blob names."""
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)
    blob_names = ['roi_deep_sup']
    if is_training:
        # labels_int32 blob: R categorical labels in [0, ..., K] for K
        # foreground classes plus background
        blob_names += ['labels_int32_roi_deep_sup']

    if is_training and cfg.MODEL.MASK_RCNN_DEEP_SUP_ON and cfg.MRCNN_DEEP_SUP.AT_STAGE == cfg.MRCNN.AT_STAGE:
        # 'mask_rois': RoIs sampled for training the mask prediction branch.
        # Shape is (#masks, 5) in format (batch_idx, x1, y1, x2, y2).
        blob_names += ['mask_rois_deep_sup']
        # 'roi_has_mask': binary labels for the RoIs specified in 'rois'
        # indicating if each RoI has a mask or not. Note that in some cases
        # a *bg* RoI will have an all -1 (ignore) mask associated with it in
        # the case that no fg RoIs can be sampled. Shape is (batchsize).
        blob_names += ['roi_has_mask_deep_sup_int32']
        # 'masks_int32' holds binary masks for the RoIs specified in
        # 'mask_rois'. Shape is (#fg, M * M) where M is the ground truth
        # mask size.
        blob_names += ['masks_deep_sup_int32']

    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ['roi_deep_sup_fpn' + str(lvl)]
        blob_names += ['roi_deep_sup_idx_restore_int32']
        if is_training:
            if cfg.MODEL.MASK_RCNN_DEEP_SUP_ON and cfg.MRCNN_DEEP_SUP.AT_STAGE == cfg.MRCNN.AT_STAGE:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['mask_rois_deep_sup_fpn' + str(lvl)]
                blob_names += ['mask_rois_deep_sup_idx_restore_int32']

    return blob_names


def add_roi_deep_sup_blobs(blobs, rois, label=None,
                           mask_rois=None,roi_has_mask=None,masks=None):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    im_ids=np.unique(rois[:,0])
    for i in range(im_ids.size):
        im_i=im_ids[i]
        rois_ind=np.where((rois[:, 0]==im_i))[0]
        if cfg.MODEL.MASK_RCNN_DEEP_SUP_ON:
            mask_rois_ind = np.where((mask_rois[:, 0] == im_i))[0]
            frcn_blobs = _sample_rois(rois[rois_ind,:],label[rois_ind],
                    mask_rois[mask_rois_ind,:],roi_has_mask[rois_ind],masks[mask_rois_ind,:])
        else:
            frcn_blobs = _sample_rois(rois[rois_ind, :], label[rois_ind])

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


    return valid


def _sample_rois(rois, label,mask_rois=None,roi_has_mask=None,masks=None):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    sampled_rois=rois
    sampled_labels=label
    labels_int32_roi_deep_sup='labels_int32_roi_deep_sup'
    roi_deep_sup='roi_deep_sup'

    blob_dict = {labels_int32_roi_deep_sup:sampled_labels.astype(np.int32, copy=False),roi_deep_sup:sampled_rois}
    # Optionally add Mask R-CNN blobs
    if cfg.MODEL.MASK_RCNN_DEEP_SUP_ON and cfg.MRCNN_DEEP_SUP.AT_STAGE == cfg.MRCNN.AT_STAGE:

        mask_rcnn_roi_deep_sup_data.add_mask_rcnn_deep_sup_blobs(
            blob_dict, mask_rois,roi_has_mask,masks
        )


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

    _distribute_rois_over_fpn_levels('roi_deep_sup')
    if cfg.MODEL.MASK_RCNN_DEEP_SUP_ON and cfg.MRCNN_DEEP_SUP.AT_STAGE == cfg.MRCNN.AT_STAGE:
        _distribute_rois_over_fpn_levels('mask_rois_deep_sup')
    if cfg.MODEL.KEYPOINTS_ON and cfg.KRCNN.AT_STAGE == 1:
        _distribute_rois_over_fpn_levels('keypoint_rois')
