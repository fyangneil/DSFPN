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

"""Construct minibatches for Mask R-CNN training. Handles the minibatch blobs
that are specific to Mask R-CNN. Other blobs that are generic to RPN or
Fast/er R-CNN are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

from detectron.core.config import cfg
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
import detectron.utils.segms as segm_utils

logger = logging.getLogger(__name__)


def add_mask_rcnn_deep_sup_blobs(blobs, mask_rois,roi_has_mask,masks):
    """Add Mask R-CNN specific blobs to the input blob dictionary."""

    # Update blobs dict with Mask R-CNN blobs
    blobs['mask_rois_deep_sup'] = mask_rois
    blobs['roi_has_mask_deep_sup_int32'] = roi_has_mask
    blobs['masks_deep_sup_int32'] = masks


