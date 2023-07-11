import os
import sys;
sys.path.append('models')
# sys.path.append('src')

import torch
from torch.optim import Adam

from models.diffusion import diffusion_model
from src.data import load_train
from src.utils import set_parser
# from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime

import SimpleITK as sitk
import numpy as np

import cv2

def main(args):
    # model initial
    print(model)
    model = diffusion_model(args)
    
    '''
    # get perturb
    p = '/root/ffy/diffusion_image_segmentation/processed/breast/MRI1_T2/UCSF-BR-24_35.npy'
    a = np.load(p, allow_pickle=True)
    x = a[1][np.newaxis,:]
    x = torch.tensor(x)
    noise = torch.randn_like(x)
    for t in range(100, 1000, 10):
        t_ = torch.tensor([t])
        perturbed_x = model.perturb_x(x, t_, noise)
        perturbed_x = perturbed_x.numpy().clip(-1, 1)
        perturbed_x = (perturbed_x+1)/2
        perturbed_x = np.uint8(255 * perturbed_x[0][0])
        perturbed_x = np.broadcast_to(perturbed_x[:,:, np.newaxis],(128,128,3))
        cv2.imwrite(f"UCSF-BR-24_35_{t}.jpg", perturbed_x)
    exit()
    '''
    
    
    model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=args.device))
        optimizer.load_state_dict(torch.load(args.load.replace('-model', '-optim')))
    
    print(next(model.parameters()).device)
    
    # dataset
    train_loader = load_train(args)

    # 模型路径
    os.makedirs(f"{args.log_dir}/{args.description}", exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{args.description}")
    for epoch in range(args.epoch_start, args.epochs+1):
        model.train()
        total_loss = 0
        # with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
        for idx, (image, mask) in enumerate(train_loader):
            mask = mask.to(args.device)
            image = image.to(args.device)
            #print('-------', image.dtype, mask.dtype)
            #exit()
            loss = model(mask, image)
            writer.add_scalar(f'epoch_loss/{epoch}', loss.item(), idx)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

                # pbar.update()  # 更新进度
                # pbar.set_postfix(**{'loss (batch)': loss.item()})  # 设置后缀
        ave_loss = total_loss/len(train_loader)
        writer.add_scalar('all_epoch/loss', ave_loss, epoch)
        writer.flush()
        print(f'epoch {epoch}, loss=', ave_loss, datetime.datetime.now())    
        # 保存模型
        if epoch % args.checkpoint_rate_pth == 0:
            model_filename = f"{args.log_dir}/{args.description}/epoch-{epoch}-model.pth"
            optim_filename = f"{args.log_dir}/{args.description}/epoch-{epoch}-optim.pth"
            torch.save(model.state_dict(), model_filename)
            torch.save(optimizer.state_dict(), optim_filename)

    if epoch % args.checkpoint_rate_pth != 0:  # 模型尚未保存
        model_filename = f"{args.log_dir}/{args.description}/epoch-{epoch}-model.pth"
        optim_filename = f"{args.log_dir}/{args.description}/epoch-{epoch}-optim.pth"
        torch.save(model.state_dict(), model_filename)
        torch.save(optimizer.state_dict(), optim_filename)


def set_args():
    epochs = 500
    epoch_start = 1  # index
    lr = 2e-4  # learning_rate
    # checkpoint_rate = 10 # 每隔多少个epoch记录下结果
    # aug = 0
    # save
    description = 'train02'
    val_dir = 'val_sample'
    log_dir = 'log'  # save pth for every epoch
        
    parser = set_parser()
    parser.add_argument('--epochs', default=epochs, type=int)
    parser.add_argument('--epoch_start', default=epoch_start, type=int)
    
    # parser.add_argument('--checkpoint_rate', default=checkpoint_rate, type=int)
    parser.add_argument('--checkpoint_rate_pth', default=25, type=int)
    # parser.add_argument('--checkpoint_rate_val', default=50, type=int)
    
    parser.add_argument('--lr', default=lr, type=float)
    parser.add_argument('--weight_decay', '-w_d', default=0, type=float)
    # parser.add_argument('--aug', default=aug, type=int)
    parser.add_argument('--description', '-name', default=description, type=str)
    # parser.add_argument('--val_dir', default=val_dir, type=str)
    parser.add_argument('--log_dir', default=log_dir, type=str)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = set_args()
    args.unet_rate = eval(args.unet_rate)
    # args.attention = eval(args.attention)
    if args.res_base_channels is None:
        args.res_base_channels = args.base_channels
    if args.resnet_rate is None:
        args.resnet_rate = args.unet_rate
    else:
        args.resnet_rate = eval(args.resnet_rate) 
    if args.load:
        assert args.epoch_start > 1, 'you have used a pre trained model, reset epoch_start please'
    else: assert args.epoch_start == 1, 'epoch_start should be one'
    assert args.load_resnet is not None, 'you should give a trained resnet model'
    # device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    args.device = device
    print(args)

    main(args)