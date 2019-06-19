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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from detectron.core.config import cfg
from detectron.datasets import json_dataset
from detectron.datasets import roidb as roidb_utils
import detectron.modeling.FPN as fpn
import detectron.roi_data.fast_rcnn as fast_rcnn_roi_data
import detectron.roi_data.super_cls as super_cls_roi_data
import detectron.utils.blob as blob_utils

# import pydevd
class AddSuperClsOp(object):
    def __init__(self, train,category):
        self._train = train
        self._category = category
        print(self._category)
    def forward(self, inputs, outputs):
        """See modeling.detector.CollectAndDistributeFpnRpnProposals for
        inputs/outputs documentation.
        """
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        if self._train:

            rois = inputs[0].data
            label=inputs[1].data
            pred_cls=inputs[2].data
            output_blob_names = super_cls_roi_data.get_super_cls_blob_names(self._category)
            blobs = {k: [] for k in output_blob_names}
            super_cls_roi_data.add_super_cls_blobs(blobs, rois,self._category,pred_cls,label)
            for i, k in enumerate(output_blob_names):
                blob_utils.py_op_copy_blob(blobs[k], outputs[i])
        else:
            rois = inputs[0].data
            distribute(rois, outputs)




def distribute(rois, outputs):
    """To understand the output blob order see return value of
    detectron.roi_data.fast_rcnn.get_fast_rcnn_blob_names(is_training=False)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)

    outputs[0].reshape(rois.shape)
    outputs[0].data[...] = rois

    # Create new roi blobs for each FPN level
    # (See: modeling.FPN.add_multilevel_roi_blobs which is similar but annoying
    # to generalize to support this particular case.)
    rois_idx_order = np.empty((0, ))
    for output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)):
        idx_lvl = np.where(lvls == lvl)[0]
        blob_roi_level = rois[idx_lvl, :]
        outputs[output_idx + 1].reshape(blob_roi_level.shape)
        outputs[output_idx + 1].data[...] = blob_roi_level
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
    rois_idx_restore = np.argsort(rois_idx_order)
    blob_utils.py_op_copy_blob(rois_idx_restore.astype(np.int32), outputs[-1])


