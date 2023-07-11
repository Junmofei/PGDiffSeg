import os
import sys;
sys.path.append('models')
sys.path.append('src')

import torch

from models.diffusion import diffusion_model
from src.data import load_test
from src.utils import set_parser
from src.advanced_convert import convert
from src.evaluation import evaluation, Evaluation

import SimpleITK as sitk

from tqdm import tqdm
import datetime


def sample(args):    
    model = diffusion_model(args)
    model.load_state_dict(torch.load(args.use_model, map_location=torch.device(args.device)))
    model.to(args.device)

    suffix = f'_{args.label}' if args.label else ''
    test_loader = load_test(args)

    path = f"predict/{args.use_model[:-4].replace('log/', '')}"
    print(path)
    os.makedirs(path, exist_ok=True)
    e = Evaluation(path, choice=args.choice, Global=True)
    model.eval()
    with tqdm(total=len(test_loader), desc='sample', unit='img') as pbar:
        for names, image_ , label in test_loader:
            image_ = image_.to(args.device)
            #print('----------------------', image.shape)
            samples = model.sample(image_)
            # save
            samples = samples.numpy().clip(-1, 1)

            for mask, name, label_ in zip(samples, names, label.numpy()):
                mask = mask.squeeze()
                name = name[:-3] + 'nii'
                sitk.WriteImage(sitk.GetImageFromArray(mask), path + '/' + name)

                e.cacu(name, mask, label_.squeeze())

            pbar.update()  # 更新进度
    print('single:')
    e.save(f'result{suffix}.csv')
    return path


def set_args():
    parser = set_parser()
    # convert
    start = 105  # start_index
    end = 131  # end_index
    # data = 'Lits'  # or abdomen dataset name

    # DSC
    choice = 0
    resize = 128
    n_classes = 2

    # test
    parser.add_argument('--use_model', required=True, type=str, help='log/train06/epoch-320-model.pth')
    parser.add_argument('--label', required=True, type=str, help='label1')
    parser.add_argument('--gt_dir', type=str, help='processed/label_1_3D')
    # parser.add_argument('--data', default=data, type=str)
    parser.add_argument('--choice', default=choice, type=int)
    parser.add_argument('--resize', default=resize, type=int)
    parser.add_argument('--n_classes', default=n_classes, type=int)
    parser.add_argument('--dice_3D', action='store_true')
    parser.add_argument('--test_index', default='test_index', type=str)

    return parser.parse_args()

def setting():
    args = set_args()
    args.unet_rate = eval(args.unet_rate)
    if args.res_base_channels is None:
        args.res_base_channels = args.base_channels
    if args.resnet_rate is None:
        args.resnet_rate = args.unet_rate
    else:
        args.resnet_rate = eval(args.resnet_rate)
    print(args)

    # device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    args.device = device
    # print(args)

    return args


if __name__ == '__main__':
    print('---', datetime.datetime.now(), '---')
    args = setting()

    path = sample(args)
    if args.dice_3D:
        assert args.gt_dir
        # path = 'predict/model4_train14/epoch-450-model'
        path_3D = convert(file_dir=path, data_path=args.root+args.data_path, test_index=args.test_index)
        # path_3D = 'predict3D/ISPY1_train01/epoch-300-model'
        if args.label:
            args.label = f'_{args.label}'
        print('mean')
        evaluation(save_path=path_3D.replace('predict3D', 'csv_save') + f'_result{args.label}.csv',
            pred_dir=path_3D,
            gt_dir=args.gt_dir,
            choice=args.choice, resize=args.resize)
