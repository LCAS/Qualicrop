import os,mat73

def example_data_load():
    data_path = os.path.join(os.getcwd(), 'data')
    file_path = os.path.join(data_path, 'qualicrop_example_predictions.mat')

    data = mat73.loadmat(file_path)
    gt_maps = data['gt_img']
    pred_maps = data['pred_img']

    return gt_maps, pred_maps

