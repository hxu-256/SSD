##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                          Hunan University                            ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import cv2
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage import img_as_ubyte
import sys
import numpy as np
from pytorch_msssim import ssim
from scipy.signal import convolve2d
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def calculate_tv(x):
    N = x.shape
    idx = torch.arange(1, N[0]+1)
    idx[-1] = N[0]-1
    ir = torch.arange(1, N[1]+1)
    ir[-1] = N[1]-1

    x1 = x[:,ir,:] - x
    x2 = x[idx,:,:] - x
    tv = torch.abs(x1) + torch.abs(x2)
    return tv.mean()


def calculate_stv(x):
    N = x.shape
    ir = torch.arange(1, N[2]+1)
    ir[-1] = N[2]-1

    x1 = x[:,:,ir] - x
    tv = torch.abs(x1)
    return tv.mean()

    
def ssim_(data, recon):
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    data = data.astype(np.float64)
    recon = recon.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(data, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(recon, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(data ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(recon ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(data * recon, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(data, recon, border=0):
    if not data.shape == recon.shape:
        raise ValueError('Data size must have the same dimensions!')
    if not data.dtype == recon.dtype:
        data, recon = data.float(), recon.float()
        
    h, w = data.shape[:2]
    data = data[border:h - border, border:w - border]
    recon = recon[border:h - border, border:w - border]
    if data.ndim == 2:
        return ssim_(data, recon)
    elif data.ndim == 3:
        return ssim(torch.unsqueeze(data, 0).permute(3, 0, 1, 2), torch.unsqueeze(recon, 0).permute(3, 0, 1, 2), data_range=1).data
     

def calculate_psnr(reference, data):
    temp_psnr = 0.
    for i in range(reference.shape[2]):
        mse = torch.mean((reference[:, :, i] - data[:, :, i])**2)
        if mse == 0:
            temp_psrn = 100
        Pixel_max = torch.max(reference[:, :, i])
        temp_psnr += 20 * torch.log10(Pixel_max / torch.sqrt(mse))
    return temp_psnr / reference.shape[2]


def clip(x):
    if not isinstance(x, np.ndarray):
        x = x.clone().cpu().numpy()
    return x.clip(0., 1.)


def ssim_index(im1, im2):
    if im1.ndim == 2:
        out = structural_similarity(im1, im2, data_range=255, gaussian_weights=True,
                                                    use_sample_covariance=False, multichannel=False)                                        
    elif im1.ndim == 3:
        out = structural_similarity(im1, im2, data_range=255, gaussian_weights=True,
                                                     use_sample_covariance=False, multichannel=True)
    else:
        sys.exit('Please input the corrected images')
    return out


def cssim(img, img_clean):
    if isinstance(img, torch.Tensor):
        img = img.data.cpu().numpy()
    if isinstance(img_clean, torch.Tensor):
        img_clean = img_clean.data.cpu().numpy()
    img = img_as_ubyte(img)
    img_clean = img_as_ubyte(img_clean)
    SSIM = ssim_index(img, img_clean)
    return SSIM


def cpsnr(img, img_clean):
    if isinstance(img, torch.Tensor):
        img = img.data.cpu().numpy()
    if isinstance(img_clean, torch.Tensor):
        img_clean = img_clean.data.cpu().numpy()
    img = img_as_ubyte(img)
    img_clean = img_as_ubyte(img_clean)
    PSNR = peak_signal_noise_ratio(img, img_clean, data_range=255)
    return PSNR
    

def get_input(tensize, const=10.0):
    inp = torch.rand(tensize)/const
    inp = torch.autograd.Variable(inp, requires_grad=True).to(device)
    inp = torch.nn.Parameter(inp)
    return inp
    

def TV_denoiser(x, _lambda, n_iter_max):
    dt = 0.25
    N = x.shape
    idx = torch.arange(1, N[0]+1)
    idx[-1] = N[0]-1
    iux = torch.arange(-1, N[0]-1)
    iux[0] = 0
    ir = torch.arange(1, N[1]+1)
    ir[-1] = N[1]-1
    il = torch.arange(-1, N[1]-1)
    il[0] = 0
    p1 = torch.zeros_like(x)
    p2 = torch.zeros_like(x)
    divp = torch.zeros_like(x)

    for i in range(n_iter_max):
        z = divp - x*_lambda
        z1 = z[:,ir,:] - z
        z2 = z[idx,:,:] - z
        denom_2d = 1 + dt*torch.sqrt(torch.sum(z1**2 + z2**2, 2))
        denom_3d = torch.unsqueeze(denom_2d, 2).repeat(1, 1, N[2])
        p1 = (p1+dt*z1)/denom_3d
        p2 = (p2+dt*z2)/denom_3d
        divp = p1-p1[:,il,:] + p2 - p2[iux,:,:]
    u = x - divp/_lambda;
    return u


def tensor_downsampling(img, scale):
    lr_img = img[:, :, 0::scale, 0::scale]
    img = F.interpolate(lr_img, scale_factor=scale, mode='bilinear')
    return img, lr_img


from torchvision import transforms
def weight_conv_denoise(img):
    transform_1 = transforms.GaussianBlur(3, 3)
    return transform_1(img)


def norm_perpoint(img):
    img = img.to(device)
    nL, nC, H, W = img.shape
    for l in range(nL):
        for i in range(H):
            for j in range(W):
                temp_point = img[l, :, i, j]
                img[l, :, i, j] = (temp_point - torch.min(temp_point)) / (torch.max(temp_point) - torch.min(temp_point))
    return img


def shrink(x, _lambda):
    u = torch.sign(x) * torch.clamp(x - 1/_lambda, 0)
    return u