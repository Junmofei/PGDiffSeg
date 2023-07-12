import os
import numpy as np
import natsort
import skimage.transform as trans
import SimpleITK as sitk
import cv2


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

def norm(img, label):  # norm to [-1,1]
    if img is not None:
        Min, Max = np.min(img), np.max(img)
        img = (img - Min) / (Max - Min) if (Max - Min > 0) else np.zeros_like(img)
        img = img.astype('float32')*2 - 1
        
    if label is not None:
        Min, Max = np.min(label), np.max(label)
        label = (label - Min) / (Max - Min) if (Max - Min > 0) else np.zeros_like(label)
        label = label.astype('float32')*2 - 1

    return img, label

def merge(m):
    mask = []
    for i in m:
        mk = cv2.imread(f'{dir}/{i}', 0)
        # print(dir, mk.shape)
        mask.append(mk)
    return sum(np.array(mask), 0)

dirs = ['normal', 'benign', 'malignant']
cc = [0,1,2]
to = '/root/ffy/PGDiffSeg/processed/BUSI'
os.makedirs(to, exist_ok=True)
train, val, test = [], [], []

for dir, c in zip(dirs, cc):
    file_list = natsort.natsorted(os.listdir(dir))
    img_list = list(filter(lambda x: 'mask' not in x, file_list))
    mask_list = list(filter(lambda x: 'mask' in x, file_list))
    names = []
    for idx, file in enumerate(img_list):  # normal (1).png
        img = cv2.imread(f'{dir}/{file}')  # (699*1304*3)
        assert img[:,:,0].all()==img[:,:,1].all()==img[:,:,2].all()
        img = img[:,:,0]
        mask = file_list
        m = list(filter(lambda x: file[:-4] in x, mask_list))
        if len(m)>1:
            print(m)
            mask = merge(m)
        else:
            mask = cv2.imread(f'{dir}/{file[:-4]}_mask.png', 0)
        img, mask = nii_resize_2D(img, mask, 128)
        img, mask = norm(img, mask)
        # npy = [img[np.newaxis,:]*-1, mask[np.newaxis,:], c]
        npy = [img[np.newaxis,:], mask[np.newaxis,:], c]
        name = f'{dir}_{idx}.npy'
        np.save(f"{to}/{name}", np.array(npy, dtype=object))
        names.append(name)
    l = len(names)
    print(dir, l)
    train.extend(names[:int(l*0.7)])
    val.extend(names[int(l*0.7):int(l*0.8)])
    test.extend(names[int(l*0.8):])

str = '\n'
with open(f"{to}/train_list.txt","w") as f:
    f.write(str.join(train))
with open(f"{to}/val_list.txt","w") as f:
    f.write(str.join(val))
with open(f"{to}/test_list.txt","w") as f:
    f.write(str.join(test))


'''
normal 133
['benign (4)_mask.png', 'benign (4)_mask_1.png']
['benign (25)_mask.png', 'benign (25)_mask_1.png']
['benign (54)_mask.png', 'benign (54)_mask_1.png']
['benign (58)_mask.png', 'benign (58)_mask_1.png']
['benign (83)_mask.png', 'benign (83)_mask_1.png']
['benign (92)_mask.png', 'benign (92)_mask_1.png']
['benign (93)_mask.png', 'benign (93)_mask_1.png']
['benign (98)_mask.png', 'benign (98)_mask_1.png']
['benign (100)_mask.png', 'benign (100)_mask_1.png']
['benign (163)_mask.png', 'benign (163)_mask_1.png']
['benign (173)_mask.png', 'benign (173)_mask_1.png']
['benign (181)_mask.png', 'benign (181)_mask_1.png']
['benign (195)_mask.png', 'benign (195)_mask_1.png', 'benign (195)_mask_2.png']
['benign (315)_mask.png', 'benign (315)_mask_1.png']
['benign (346)_mask.png', 'benign (346)_mask_1.png']
['benign (424)_mask.png', 'benign (424)_mask_1.png']
benign 437
['malignant (53)_mask.png', 'malignant (53)_mask_1.png']
malignant 210

'''