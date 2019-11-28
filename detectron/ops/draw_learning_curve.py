import pickle,glob,os
import matplotlib.pyplot as plt
def extract_APbbox():
    folder_path='/media/fanyang/C/data/coco/evaluate_results/' \
              'mask_rcnn_dsfpn_1x/10ktrain/'
    folder_list=glob.glob(folder_path+'model*')
    save_path='/media/fanyang/C/data/coco/evaluate_results/mask_rcnn_dsfpn_1x'
    for folder_path_name in folder_list:
        folder_name=os.path.basename(folder_path_name)
        iter_num=folder_name.split('iter')[1]
        new_folder_name='model_iter{:07d}'.format(int(iter_num))
        os.rename(folder_path+folder_name,folder_path+new_folder_name)
    folder_list=glob.glob(folder_path+'model*')
    folder_list.sort()
    APbbox_list=[]
    for i,folder_path_name in enumerate(folder_list):
        print('model {}'.format(i))

        file='test/coco_2014_minival/generalized_rcnn/detection_results.pkl'
        file=os.path.join(folder_path_name,file)

        with open(file, "rb") as fp:
            det_eval_result = pickle.load(fp)
            print('load file done')
            APbbox=det_eval_result.stats[0]
            APbbox_list.append(APbbox)

    save_file=os.path.join(save_path,'10ktrain.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(APbbox_list, f)
def draw_curve():
    file_path='/media/fanyang/C/data/coco/evaluate_results/mask_rcnn_dsfpn_1x'
    file = os.path.join(file_path, '10ktrain.pkl')
    save_file=os.path.join(file_path,'learning_curve.pdf')
    with open(file, 'rb') as f:
        APbbox_list = pickle.load(f)
    iter_num=range(4999,360000,5000)
    plt.plot(iter_num, APbbox_list, 'r')

    file = os.path.join(file_path, 'minival.pkl')
    with open(file, 'rb') as f:
        APbbox_list = pickle.load(f)
    plt.plot(iter_num, APbbox_list, 'r--')

    file_path = '/media/fanyang/C/data/coco/evaluate_results/mask_rcnn_1x'
    file = os.path.join(file_path, '10ktrain_v1.pkl')
    with open(file, 'rb') as f:
        APbbox_list = pickle.load(f)
    plt.plot(iter_num, APbbox_list, 'b')


    file = os.path.join(file_path, 'minival.pkl')
    with open(file, 'rb') as f:
        APbbox_list = pickle.load(f)
    plt.plot(iter_num, APbbox_list, 'b--')


    plt.legend(['train-DSFPN','val-DSFPN','train-FPN','val-FPN'],fontsize=12)
    plt.xlabel('iterations',fontsize=12)
    plt.ylabel('APbbox',fontsize=12)
    plt.ticklabel_format(axis='x', style='sci',scilimits=[0,0],useMathText=True)
    plt.gca().yaxis.grid(True)    # plt.show()
    plt.savefig(save_file,bbox_inches = 'tight',
    pad_inches = 0)

# extract_APbbox()
draw_curve()
