import pickle,os
import numpy as np
import matplotlib.pyplot as plt
def vis_feat_map():
    """
    visualize feature map of each image
    """

    file_path = '../test/mask_rcnn_DSFPN_r50_2x/test/coco_2014_val/generalized_rcnn/feat_map'

    output_path = '../test/mask_rcnn_DSFPN_r50_2x/test/coco_2014_val/generalized_rcnn/feat_map_vis'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    file_lsit=os.listdir(file_path)
    for file_name in file_lsit:
        file = os.path.join(file_path, file_name)

        save_path = os.path.join(output_path, file_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(file, "rb") as fp:
            feat_map_list = pickle.load(fp)
            print("load file done")
            for i,entry in enumerate(feat_map_list):
                feat_map=entry
                feat_map[feat_map < 0] = 0
                feat_map_sum = np.sum(feat_map, axis=0)
                save_file=os.path.join(save_path,file_name+'_'+str(i)+'.png')
                plt.imsave(save_file, feat_map_sum)
vis_feat_map()