import argparse

import skimage.transform as trans
import numpy as np

import SimpleITK as sitk
import cv2

# setting
def set_parser():
    schedule_low=1e-4
    schedule_high=0.02
    loss_weight = 0.5  # classify loss weight
    noise_schedule="linear"

    # batch_size=30
    num_workers = 0

    # num_timesteps=200  # T
    # base_channels = 64
    # unet_rate = '[1,1,1,2,2,4,4]'

    # data
    

    parser = argparse.ArgumentParser(description='initial')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--use_sigmoid', action='store_true', help='use sigmoid after resnet18')
    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--schedule_low', default=schedule_low, type=float)
    parser.add_argument('--schedule_high', default=schedule_high, type=float)
    parser.add_argument('--loss_weight', default=loss_weight, type=float)
    parser.add_argument('--noise_schedule', default=noise_schedule, type=str)

    parser.add_argument('--num_timesteps', '-t', required=True, type=int)
    parser.add_argument('--base_channels', required=True, type=int)
    parser.add_argument('--unet_rate', required=True, type=str)
    
    
    # classify
    parser.add_argument('--load_resnet', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--use_classify', action='store_true')
    parser.add_argument('--classify_model', help='model name')
    parser.add_argument('--res_base_channels', type=int)
    parser.add_argument('--resnet_rate', type=str)
    parser.add_argument('--res_block', type=str)  # default: BasicBlock
    parser.add_argument('--num_blocks', type=str)  # default: [2]*len(rate)
    parser.add_argument('--classes', default=2, type=int, help='class number')
    parser.add_argument('--img_size', default=128, type=int)
    # select model flexibly
    parser.add_argument('--denoise_model', required=True, type=str, help='denoise_model1')
    parser.add_argument('--super_resnet_deep', default=4, type=int)
    # this is for dense net
    parser.add_argument('--growth_rate', default=16, type=int)
    parser.add_argument('--num_layers', default=4, type=int)

    parser.add_argument('--map_down', action='store_true', help='down sample the classify feature map before add to bottleneck')

   
    
    
    parser.add_argument('--root', default='')
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--batch_size', '-bs', required=True, type=int)
    parser.add_argument('--num_workers', default=num_workers, type=int)
    parser.add_argument('--train_list', default='train_list', type=str)
    parser.add_argument('--test_list', default='test_list', type=str)

    return parser


def read_pics(*paths):
    pics = []
    for path in paths:
        if path.endswith('.nii') or path.endswith('.dcm') or path.endswith('.nii.gz'):
            pic = sitk.ReadImage(path)
            pic = sitk.GetArrayFromImage(pic)
        else:
            pic = cv2.imread(path, 0)
        pics.append(pic)
    return pics if len(pics) > 1 else pics[0]


def resample_3D_nii_to_Fixed_size(nii_image, image_new_size, resample_methold=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()

    image_original_size = nii_image.GetSize()  # 原始图像的尺寸
    image_original_spacing = nii_image.GetSpacing()  # 原始图像的像素之间的距离
    image_new_size = np.array(image_new_size, float)
    factor = image_original_size / image_new_size
    image_new_spacing = image_original_spacing * factor
    image_new_size = image_new_size.astype(np.int)

    resampler.SetReferenceImage(nii_image)  # 需要resize的图像（原始图像）
    resampler.SetSize(image_new_size.tolist())
    resampler.SetOutputSpacing(image_new_spacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resample_methold)

    return resampler.Execute(nii_image)


def nii_resize_2D(image, label, shape):
    """
    type of image,label: Image or array or None

    :return: array or None
    """
    # image
    if isinstance(image, sitk.SimpleITK.Image):  # image need type array, if not, transform it
        image = sitk.GetArrayFromImage(image)
    if image is not None:
        image = trans.resize(image, (shape, shape))
    # label
    if isinstance(label, np.ndarray):
        label = sitk.GetImageFromArray(label)  # label1 need type Image
    if label is not None:
        label = resample_3D_nii_to_Fixed_size(label, (shape, shape),
                                              resample_methold=sitk.sitkNearestNeighbor)
        label = sitk.GetArrayFromImage(label)
    return image, label

