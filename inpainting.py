from dataset import load_cifar_dataset, load_transformed_dataset, load_specific_image, BATCH_SIZE, IMG_SIZE
from utils import T, apply_noise, save_tensor_image, apply_random_mask, show_tensor_image, apply_noise_from_previous, apply_center_mask
from main import sample_timestep, sample_plot_image, device
from torch.optim import Adam
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import random
import os
import numpy as np
import cv2
from lpips import LPIPS
from pytorch_msssim import SSIM
     
@torch.no_grad()
def sample_plot_image_for_inpainting_with_resampling_faces(mask, original_img, img, resampling_steps = 1, jumping_steps = 1):
    img_size = IMG_SIZE
    if img is None:
        img = torch.randn((1, 3, img_size, img_size), device=device)
    img = apply_noise(img, torch.tensor([499], device = device, dtype = torch.long), device)[0] # get x_T
    if len(img.shape) == 3:
                img = img.unsqueeze(0)
    u = 0
    for i in range(0,T)[::-1]:
        u += 1
        t = torch.full((1,), i, device=device, dtype=torch.long)
        if (i != T-2 and u == jumping_steps ):
            u = 0
            img = sample_timestep(img, t)
            img = img * (1 - mask) + mask * apply_noise(original_img, t, device)[0]
            for j in range(resampling_steps - 1):
                for k in range(jumping_steps - 1): 
                    current_t = torch.full((1,), i + k + 1, device=device, dtype=torch.long)
                    img = apply_noise_from_previous(img, current_t, device)[0]
                for k in range(jumping_steps-1): 
                    current_t = torch.full((1,), i + jumping_steps - 1 - k - 1, device=device, dtype=torch.long)
                    img = sample_timestep(img, current_t)
                    img = img * (1 - mask) + mask * apply_noise(original_img, current_t, device)[0]
        else:
            img = sample_timestep(img, t)
            img = img * (1 - mask) + mask * apply_noise(original_img, t, device)[0]
    return img * (1-mask) + mask * original_img       


def tensor_to_numpy(tensor):
    """Convert a tensor image to numpy array in range [0, 255]"""
    # Convert from [-1, 1] to [0, 1]
    img = (tensor + 1) / 2
    # Convert to numpy and scale to [0, 255]
    img = (img.cpu().numpy() * 255).astype(np.uint8)
    # Handle different input shapes
    if len(img.shape) == 4:  # [B, C, H, W]
        img = img[0]  # Take first image
    if img.shape[0] == 3:  # [C, H, W]
        img = np.transpose(img, (1, 2, 0))  # Convert to [H, W, C]
    return img

def calculate_psnr(original, reconstructed):
    """
    Calculate PSNR between original and reconstructed images.
    
    Args:
        original: Original image tensor in range [-1, 1]
        reconstructed: Reconstructed image tensor in range [-1, 1]
    
    Returns:
        PSNR value in dB
    """
    # Convert tensors to numpy arrays
    original_np = tensor_to_numpy(original)
    reconstructed_np = tensor_to_numpy(reconstructed)
    
    # Calculate MSE
    mse = np.mean((original_np - reconstructed_np) ** 2)
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    return psnr

def calculate_metrics(original, reconstructed):
    """
    Calculate PSNR, SSIM, and LPIPS between original and reconstructed images.
    
    Args:
        original: Original image tensor in range [-1, 1]
        reconstructed: Reconstructed image tensor in range [-1, 1]
    
    Returns:
        Dictionary containing PSNR, SSIM, and LPIPS scores
    """
    # Initialize metrics
    lpips_fn = LPIPS(net='alex').to(device)
    ssim_fn = SSIM(data_range=2.0, size_average=True, channel=3).to(device)
    
    # Ensure images are in correct format for metrics
    original = original.to(device)
    reconstructed = reconstructed.to(device)
    
    # Ensure both images have the same format [B, C, H, W]
    if len(original.shape) == 3:
        original = original.unsqueeze(0)
    if len(reconstructed.shape) == 3:
        reconstructed = reconstructed.unsqueeze(0)
    
    # Calculate PSNR
    psnr = calculate_psnr(original, reconstructed)
    
    # Calculate SSIM
    ssim = ssim_fn(original, reconstructed).item()
    
    # Calculate LPIPS
    lpips = lpips_fn(original, reconstructed).item()
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'lpips': lpips
    }



    
