import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from utils import apply_noise, get_index_from_list, show_tensor_image, T, betas, sqrt_one_minus_alphas_barred, sqrt_recip_alphas, posterior_variance
from improvedmodel import SimpleUnet
from dataset import  BATCH_SIZE, IMG_SIZE, load_transformed_dataset, load_cifar_dataset
from torch.optim import Adam
from tqdm import tqdm
import os
import cv2

model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
def get_loss(model, x_0, t):
    x_noisy, noise = apply_noise(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return 0.9 * F.l1_loss(noise, noise_pred) + 0.1 * F.mse_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_Ts = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_barred_t = get_index_from_list(
        sqrt_one_minus_alphas_barred, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_Ts * model(x, t) / sqrt_one_minus_alphas_barred_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image(img = None):
    img_size = IMG_SIZE
    if img is None:
        img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img[0].cpu())
    plt.show()
    return img            


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
CHECKPOINT_PATH = "model_new_arch (3).pth"
data_loader = load_transformed_dataset()
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Resume from the next epoch
    print(f"Resuming training from epoch {start_epoch}")
epochs = 100 
for epoch in range(start_epoch,epochs):
    epoch_loss = 0
    with tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
        for step, batch in pbar:
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, batch[0], t)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, CHECKPOINT_PATH)
    print(f"Checkpoint saved at epoch {epoch+1}")

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")