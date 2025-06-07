import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random
def sigmoid_beta_schedule(timesteps, start = -3, end = 3):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start).sigmoid()
    v_end = torch.tensor(end).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start)).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 10e-4,0.02).to(torch.float32)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
def apply_noise_from_previous(x_t_minus_one, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_t_minus_one)
    beta_t = get_index_from_list(betas, t, x_t_minus_one.shape)
    sqrt_one_minus_beta_t = torch.sqrt(1. - beta_t)

    return sqrt_one_minus_beta_t.to(device) * x_t_minus_one.to(device) \
    + beta_t.to(device) * noise.to(device), noise.to(device)

    
def apply_noise(x_0, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_barred, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_barred, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)
# Define beta schedule
T = 500
betas = sigmoid_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_barred = torch.cumprod(alphas, axis=0)
alphas_barred_prev = F.pad(alphas_barred[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_barred = torch.sqrt(alphas_barred)
sqrt_one_minus_alphas_barred = torch.sqrt(1. - alphas_barred)
posterior_variance = betas * (1. - alphas_barred_prev) / (1. - alphas_barred)

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))
def save_tensor_image(image, path, name):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    reverse_transforms(image).save(path + "\\" + name + ".png")
def apply_random_mask(image, max_mask_area=0.5):
    """
    Apply a randomly placed rectangular mask on the image.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W)
        max_mask_area (float): Maximum percentage of the image to mask (default: 50%)
        
    Returns:
        torch.Tensor: Masked image
    """
    _, H, W = image.shape  # Get image dimensions
    
    # Compute max mask size based on max_mask_area
    max_mask_pixels = int(H * W * max_mask_area)
    
    # Randomly determine mask width and height (ensuring it doesn't exceed max_mask_pixels)
    mask_h = random.randint(H // 8, H // 3)  # Height of mask (within reasonable bounds)
    mask_w = min(max_mask_pixels // mask_h, W // 3)  # Width, ensuring total area ≤ max_mask_pixels

    # Randomly select top-left corner of the mask
    top = random.randint(0, H - mask_h)
    left = random.randint(0, W - mask_w)

    # Create a binary mask (1 for visible, 0 for masked)
    mask = torch.ones_like(image)
    mask[:, top:top + mask_h, left:left + mask_w] = 0  # Apply the mask

    # Apply the mask to the image
    masked_image = image * mask

    return masked_image, mask

def apply_center_mask(image, mask_size=0.3):
    """
    Apply a mask in the center of the image.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W)
        mask_size (float): Size of the mask relative to image dimensions (default: 30%)
        
    Returns:
        torch.Tensor: Masked image
        torch.Tensor: Binary mask
    """
    _, H, W = image.shape  # Get image dimensions
    
    # Calculate mask dimensions
    mask_h = int(H * mask_size)
    mask_w = int(W * mask_size)
    
    # Calculate center position
    top = (H - mask_h) // 2
    left = (W - mask_w) // 2

    # Create a binary mask (1 for visible, 0 for masked)
    mask = torch.ones_like(image)
    mask[:, top:top + mask_h, left:left + mask_w] = 0  # Apply the mask

    # Apply the mask to the image
    masked_image = image * mask

    return masked_image, mask