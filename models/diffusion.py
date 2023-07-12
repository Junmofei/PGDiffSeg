import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from model import get_denoise_model
import cv2

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        # scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas

    """
    def __init__(
        self,
        model,
        img_size,
        img_channels,
        betas,
        device
    ):
        super().__init__()

        self.model = model
        # self.ema_model = deepcopy(model)

        # self.ema = EMA(ema_decay)

        self.img_size = img_size
        self.img_channels = img_channels
        self.device = device
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    @torch.no_grad()
    def remove_noise(self, x, y, t):

        return (
            (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, y, t)) *
            extract(self.reciprocal_sqrt_alphas, t, x.shape)
        )

    @torch.no_grad()
    def sample(self, y, see=False):
        x = torch.randn_like(y)  # TODO: device
        #         x = torch.randn(y.shape[0], self.img_channels, *self.img_size, device=self.device)
        if see:
            print('sample, the first:', torch.min(x), torch.max(x))
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=self.device).repeat(y.shape[0])
            x = self.remove_noise(x, y, t_batch)
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            if see:
                print(f'sample, in time {t}:', torch.min(x), torch.max(x))
        
        return x.cpu().detach()
    

    def cam(self, y, save, layer):
        image_weight, heatmap_weight= 0.7, 0.6
        x = torch.randn_like(y)  # TODO: device
        #         x = torch.randn(y.shape[0], self.img_channels, *self.img_size, device=self.device)
        for t in range(self.num_timesteps - 1, -1, -1):  # -1
            t_batch = torch.tensor([t], device=self.device).repeat(y.shape[0])
            # remove_noise
            # x = self.remove_noise(x, y, t_batch)
            self.model.zero_grad()
            activations, predictions =  self.model.cam(x, y, t_batch, layer)
            predictions.backward(torch.ones_like(predictions))  # , retain_graph=True
            

            x_= x.detach()
            
            # exit()
            weights = np.mean(self.model.unet.gradients.detach().cpu().numpy(), axis=(2, 3))[0, :]
            cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations[-1].detach().cpu().numpy(), axis=0)
            cam = np.maximum(cam, 0)
            if cam.max != 0:
                cam = cam / cam.max()
            # 将 CAM 可视化到原始图像上
            cam = cv2.resize(cam, (128, 128))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            
            noise= x_.cpu().numpy()[0][0].clip(-1, 1)
            noise = np.uint8(255 *(noise+1)/2 )
            noise = np.broadcast_to(noise[:,:, np.newaxis],(128,128,3))
            # print(heatmap.shape, img.shape, np.max(heatmap), np.min(heatmap))
            img =  y.detach().cpu().numpy()[0][0]
            img = np.uint8(255 * (img+1)/2)
            img = np.broadcast_to(img[:,:, np.newaxis],(128,128,3))
            result2 = heatmap * heatmap_weight + img*image_weight
            if layer.endswith('_2'):
                result = heatmap * heatmap_weight + noise*image_weight
                cv2.imwrite(f"{save}/{t}_map.jpg", result.clip(0, 255))
            cv2.imwrite(f"{save}/{t}_map2.jpg", result2.clip(0, 255))
            cv2.imwrite(f"{save}/{t}_noise.jpg", noise.clip(0, 255))

            x= (
                (x - extract(self.remove_noise_coeff, t_batch, x.shape) * predictions) *
                extract(self.reciprocal_sqrt_alphas, t_batch, x.shape)
            )
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            x= x.detach()
            


        # return x.cpu().detach()
    

    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )   

    def get_losses(self, x, y, t):
        noise = torch.randn_like(x)

        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, y, t)

        return F.mse_loss(estimated_noise, noise)  # TODO: 

    def forward(self, x, y):
        b, c, h, w = x.shape
        device = x.device

        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[0]:
            raise ValueError("image width does not match diffusion parameters")
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, y, t)

def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)

def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return np.array(betas)


def diffusion_model(args):
    model = get_denoise_model(args)
    #model.to(args.device)

    if args.noise_schedule == "linear":
        betas = generate_linear_schedule(
                args.num_timesteps,
                args.schedule_low * 1000 / args.num_timesteps,
                args.schedule_high * 1000 / args.num_timesteps
            )
    elif args.noise_schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        raise NotImplementedError(f"unknown beta schedule: {args.noise_schedule}")

    return GaussianDiffusion(model, (args.img_size, args.img_size), 1, betas, args.device)

# if __name__ == '__main__':
#     a = 'tst'
#     b = get_denoise_model(a)
#     c = b('best')
#     print(c)