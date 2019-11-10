import pickle
file='/media/fanyang/C/code/detector/Detectron-Cascade-RCNN/test/mask_rcnn_DSFPN_r50_2x/test/coco_2014_val/generalized_rcnn/detection_results.pkl'
with open(file, "rb") as fp:
    feat_map_list = pickle.load(fp)
    print('load file done')
