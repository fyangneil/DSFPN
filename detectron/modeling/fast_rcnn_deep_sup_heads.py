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

"""Various network "heads" for classification and bounding box prediction.

The design is as follows:

... -> RoI ----\                               /-> box cls output -> cls loss
                -> RoIFeatureXform -> box head
... -> Feature /                               \-> box reg output -> reg loss
       Map

The Fast R-CNN head produces a feature representation of the RoI for the purpose
of bounding box classification and regression. The box output module converts
the feature representation into classification and regression predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils


def add_roi_deep_sup_inputs(model):
    model.AddRoiDeepSup()
# ---------------------------------------------------------------------------- #
# Fast R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_roi_deep_sup_outputs(model, blob_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    # Box classification layer
    model.FC(
        blob_in,
        'roi_deep_sup_score',
        dim,
        cfg.MODEL.NUM_CLASSES,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('roi_deep_sup_score', 'roi_deep_sup_prob', engine='CUDNN')

    # Box regression layer
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    model.FC(
        blob_in,
        'roi_deep_sup_bbox_pred',
        dim,
        num_bbox_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )

    if cfg.MODEL.CASCADE_ON:
        # add stage parameters to list
        if '1' not in model.stage_params:
                model.stage_params['1'] = []
        for idx in range(-2, 0):
            model.stage_params['1'].append(model.weights[idx])
            model.stage_params['1'].append(model.biases[idx])
def add_roi_deep_sup_decouple_outputs(model, blob_in_cls,blob_in_reg, dim):
    """Add RoI classification and bounding box regression output ops."""
    # Box classification layer
    model.FC(
        blob_in_cls,
        'roi_deep_sup_score',
        dim,
        cfg.MODEL.NUM_CLASSES,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('roi_deep_sup_score', 'roi_deep_sup_prob', engine='CUDNN')


    if cfg.MODEL.CASCADE_ON:
        # add stage parameters to list
        if '1' not in model.stage_params:
            model.stage_params['1'] = []
        for idx in range(-2, 0):
            model.stage_params['1'].append(model.weights[idx])
            model.stage_params['1'].append(model.biases[idx])
    # Box regression layer
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    model.FC(
        blob_in_reg,
        'roi_deep_sup_bbox_pred',
        dim,
        num_bbox_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )

    if cfg.MODEL.CASCADE_ON:
        # add stage parameters to list
        if '1' not in model.stage_params:
                model.stage_params['1'] = []
        for idx in range(-2, 0):
            model.stage_params['1'].append(model.weights[idx])
            model.stage_params['1'].append(model.biases[idx])

def add_roi_deep_sup_losses(model):
    """Add losses for RoI classification and bounding box regression."""
    stage = 1 if cfg.FAST_RCNN_DEEP_SUP.AT_STAGE == 1 else cfg.FAST_RCNN_DEEP_SUP.AT_STAGE
    stage_name = '_{}'.format(stage) if stage > 1 else ''
    loss_scalar = 1
    if cfg.MODEL.CASCADE_ON and cfg.CASCADE_RCNN.SCALE_LOSS:
        loss_scalar = cfg.CASCADE_RCNN.STAGE_WEIGHTS[0]
    roi_deep_sup_prob, loss_roi_deep_sup_cls = model.net.SoftmaxWithLoss(
        ['roi_deep_sup_score', 'labels_int32_roi_deep_sup'],
        ['roi_deep_sup_prob', 'loss_roi_deep_sup_cls'+stage_name],
        scale=model.GetLossScale() * loss_scalar
    )


    loss_roi_deep_sup_bbox = model.net.SmoothL1Loss(
        [
            'roi_deep_sup_bbox_pred', 'bbox_targets'+stage_name, 'bbox_inside_weights'+stage_name,
            'bbox_outside_weights'+stage_name
        ],
        'loss_deep_sup_bbox'+stage_name,
        scale=model.GetLossScale() * loss_scalar
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_roi_deep_sup_cls, loss_roi_deep_sup_bbox])
    model.Accuracy(['roi_deep_sup_prob', 'labels_int32_roi_deep_sup'],
                   'accuracy_roi_deep_sup_cls'+stage_name)
    model.AddLosses(['loss_roi_deep_sup_cls'+stage_name,'loss_deep_sup_bbox'+stage_name])
    model.AddMetrics('accuracy_roi_deep_sup_cls'+stage_name)

    bbox_reg_weights = cfg.MODEL.BBOX_REG_WEIGHTS
    model.AddBBoxAccuracy(
        ['roi_deep_sup_bbox_pred', 'roi_deep_sup', 'labels_int32_roi_deep_sup', 'mapped_gt_boxes'+stage_name],
        ['deep_sup_bbox_iou'+stage_name, 'deep_sup_bbox_iou_pre'+stage_name], bbox_reg_weights)
    model.AddMetrics(['deep_sup_bbox_iou'+stage_name, 'deep_sup_bbox_iou_pre'+stage_name])
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    print('add_deep_sup_roi_2mlp_head')
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    blob_out='roi_deep_sup_feat'
    blob_rois_name='roi_deep_sup'
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        blob_out,
        blob_rois=blob_rois_name,
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    # normalize the gradient by the number of cascade heads
    if cfg.MODEL.CASCADE_ON and cfg.CASCADE_RCNN.SCALE_GRAD:
        grad_scalar = cfg.CASCADE_RCNN.STAGE_WEIGHTS[0]
        model.net.Scale(
            roi_feat, roi_feat, scale=1.0, scale_grad=grad_scalar
        )
    model.FC(roi_feat, 'fc6_roi_deep_sup', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6_roi_deep_sup', 'fc6_roi_deep_sup')
    model.FC('fc6_roi_deep_sup', 'fc7_roi_deep_sup', hidden_dim, hidden_dim)
    model.Relu('fc7_roi_deep_sup', 'fc7_roi_deep_sup')


    if cfg.MODEL.CASCADE_ON:
        # add stage parameters to list
        if '1' not in model.stage_params:
            model.stage_params['1'] = []
        for idx in range(-2, 0):
            model.stage_params['1'].append(model.weights[idx])
            model.stage_params['1'].append(model.biases[idx])
    return 'fc7_roi_deep_sup', hidden_dim

def add_roi_2mlp_decouple_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    print('add_deep_sup_roi_2mlp_head')
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    blob_out='roi_deep_sup_feat'
    blob_rois_name='roi_deep_sup'
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        blob_out,
        blob_rois=blob_rois_name,
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    # normalize the gradient by the number of cascade heads
    if cfg.MODEL.CASCADE_ON and cfg.CASCADE_RCNN.SCALE_GRAD:
        grad_scalar = cfg.CASCADE_RCNN.STAGE_WEIGHTS[0]
        model.net.Scale(
            roi_feat, roi_feat, scale=1.0, scale_grad=grad_scalar
        )
    model.FC(roi_feat, 'fc6_roi_deep_sup_cls', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6_roi_deep_sup_cls', 'fc6_roi_deep_sup_cls')
    model.FC('fc6_roi_deep_sup_cls', 'fc7_roi_deep_sup_cls', hidden_dim, hidden_dim)
    model.Relu('fc7_roi_deep_sup_cls', 'fc7_roi_deep_sup_cls')

    model.FC(roi_feat, 'fc6_roi_deep_sup_reg', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6_roi_deep_sup_reg', 'fc6_roi_deep_sup_reg')
    model.FC('fc6_roi_deep_sup_reg', 'fc7_roi_deep_sup_reg', hidden_dim, hidden_dim)
    model.Relu('fc7_roi_deep_sup_reg', 'fc7_roi_deep_sup_reg')
    if cfg.MODEL.CASCADE_ON:
        # add stage parameters to list
        if '1' not in model.stage_params:
            model.stage_params['1'] = []
        for idx in range(-2, 0):
            model.stage_params['1'].append(model.weights[idx])
            model.stage_params['1'].append(model.biases[idx])
    return 'fc7_roi_deep_sup_cls','fc7_roi_deep_sup_reg', hidden_dim

def add_roi_Xconv1fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, with GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in, 'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    # normalize the gradient by the number of cascade heads
    if cfg.MODEL.CASCADE_ON and cfg.CASCADE_RCNN.SCALE_GRAD:
        grad_scalar = cfg.CASCADE_RCNN.STAGE_WEIGHTS[0]
        model.net.Scale(
            roi_feat, roi_feat, scale=1.0, scale_grad=grad_scalar
        )

    current = roi_feat
    num_convs = cfg.FAST_RCNN.NUM_STACKED_CONVS
    for i in range(num_convs):
        current = model.ConvGN(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            group_gn=get_group_gn(hidden_dim),
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    if cfg.MODEL.CASCADE_ON:
        # add stage parameters to list
        if '1' not in model.stage_params:
            model.stage_params['1'] = []
        num_params = 2 * num_convs + 1
        for idx in range(-num_params, 0):
            model.stage_params['1'].append(model.weights[idx])
        # head convs don't have bias
        num_params = num_convs + 1
        for idx in range(-num_params, 0):
            model.stage_params['1'].append(model.biases[idx])
    return 'fc6', fc_dim
