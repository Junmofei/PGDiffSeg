import os
import SimpleITK as sitk
import numpy as np
import pandas as pd

from utils import resample_3D_nii_to_Fixed_size
from metrics import Metirc

def convert_probability_to_mask_array(predict, choice):
    predict = sitk.GetArrayFromImage(predict)
    mask = np.zeros_like(predict, dtype='uint8')
    mask[predict > choice] = 1
    return mask



class Evaluation():
    def __init__(self, save_path, choice=0, metrics='all', Global=False):
        if metrics == 'all':
            self.metrics = ['dice', 'Jaccard', 'recall', 'precision', 'FNR', 'FPR']  # 'RVD', 
        else:
            self.metrics = metrics

        self.result={
            'name':[],
            'dice':[],
            'Jaccard':[],
            'recall':[],
            'precision':[],
            # 'RVD':[],
            'FNR':[],
            'FPR':[]
        }
        self.fun = {
            'dice':'metirc.dice_coef()',
            'Jaccard':'metirc.iou_score()',
            'recall':'metirc.recall()',
            'precision':'metirc.precision()',
            'RVD':'metirc.RVD()',
            'FNR':'metirc.FNR()',
            'FPR':'metirc.FPR()'
        }
        self.Global = Global
        self.PRE = []
        self.GT = []
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.save_path = save_path
        self.choice = choice

    
    def cacu(self, name, predict, groundtruth):        
        pre2 = np.zeros_like(predict, dtype='uint8')
        pre2[predict > self.choice] = 1
        gt2 = np.zeros_like(groundtruth, dtype='uint8')
        gt2[groundtruth > 0] = 1
        
        if self.Global:
            self.PRE.append(pre2)
            self.GT.append(gt2)
            
        self.result['name'].append(name)
        metirc = Metirc(pre2, gt2)
        for m in self.metrics:
            self.result[m].append(eval(self.fun[m]))

    def view(self):
        for m in self.metrics:
            print(m, ':', np.mean(self.result[m]))
        if self.Global:
            print('Global:')
            metirc = Metirc(np.array(self.PRE), np.array(self.GT))
            for m in self.metrics:
                print(m, eval(self.fun[m]))
            
    def save(self, csv_name):
        self.result['name'].append('means')
        for m in self.metrics:
            mean = np.mean(self.result[m])
            print(m, ':', mean)
            self.result[m].append(mean)
        if self.Global:
            print(f'Global:({np.array(self.PRE).shape})({np.array(self.GT).shape})')
            self.result['name'].append('Global')
            metirc = Metirc(np.array(self.PRE), np.array(self.GT))
            for m in self.metrics:
                g = eval(self.fun[m])
                print(m, g)
                self.result[m].append(g)
        df = pd.DataFrame(self.result)
        df.to_csv(self.save_path + '/' + csv_name, index=False)
        # print(df)







def evaluation(save_path, pred_dir, gt_dir, choice=0, resize=None, metrics='all'):
    if metrics == 'all':
        metrics = ['dice', 'Jaccard', 'recall', 'precision', 'FNR', 'FPR']  # 'RVD', 
    result={
        'name':[],
        'dice':[],
        'Jaccard':[],
        'recall':[],
        'precision':[],
        # 'RVD':[],
        'FNR':[],
        'FPR':[]
    }
    fun = {
        'dice':'metirc.dice_coef()',
        'Jaccard':'metirc.iou_score()',
        'recall':'metirc.recall()',
        'precision':'metirc.precision()',
        'RVD':'metirc.RVD()',
        'FNR':'metirc.FNR()',
        'FPR':'metirc.FPR()'
    }
    os.makedirs(save_path[:save_path.rfind('/')], exist_ok=True)
    pred_filenames = os.listdir(pred_dir)

    for i in range(len(pred_filenames)):
        name = pred_filenames[i]

        predict = sitk.ReadImage(os.path.join(pred_dir, name))
        predict = convert_probability_to_mask_array(predict, choice)
        #predict = sitk.GetArrayFromImage(predict)

        groundtruth = sitk.ReadImage(os.path.join(gt_dir, name))
        if resize:
            groundtruth = resample_3D_nii_to_Fixed_size(groundtruth, (resize, resize,
                            groundtruth.GetSize()[2]),
                            resample_methold=sitk.sitkNearestNeighbor)
        groundtruth = sitk.GetArrayFromImage(groundtruth)

        result['name'].append(name)
        
        metirc = Metirc(predict, groundtruth)
        for m in metrics:
            result[m].append(eval(fun[m]))

    # 计算均值 todo: All arrays must be of the same length
    result['name'].append('means')
    for m in metrics:
        result[m].append(np.mean(result[m]))

    df = pd.DataFrame(result)
    df.to_csv(save_path, index=False)
    print(df)

