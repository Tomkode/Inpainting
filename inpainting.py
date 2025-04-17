from dataset import load_cifar_dataset, load_transformed_dataset, load_specific_image, BATCH_SIZE, IMG_SIZE
from utils import T, apply_noise, save_tensor_image, apply_random_mask, show_tensor_image, apply_noise_from_previous
from main import sample_timestep, sample_plot_image, device
from torch.optim import Adam
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import random
import os
import numpy as np
import cv2
@torch.no_grad()
def sample_plot_image_for_inpainting(mask, original_img, img):
    # Sample noise
    img_size = IMG_SIZE
    if img is None:
        img = torch.randn((1, 3, img_size, img_size), device=device)
    # plt.figure(figsize=(15,15))
    # plt.axis('off')
    # num_images = 10
    # stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        if len(img.shape) == 3:
                img = img.unsqueeze(0)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = img * (1 - mask) + mask * apply_noise(original_img, t, device)[0] if i != 0 else img * (1 - mask) + mask * original_img
        img = torch.clamp(img, -1.0, 1.0)
        
        # if i % stepsize == 0:
        #     plt.subplot(1, num_images, int(i/stepsize)+1)
        #     show_tensor_image(img[0].cpu())
    # plt.show()
    return img            
@torch.no_grad()
def sample_plot_image_for_inpainting_with_resampling(mask, original_img, img, res_steps = 10, res_freq = 0.5):
    # Sample noise
    img_size = IMG_SIZE
    if img is None:
        img = torch.randn((1, 3, img_size, img_size), device=device)
    resampling_steps = res_steps
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)
    img = apply_noise(img, torch.tensor([499], device = device, dtype = torch.long), device)[0] # get x_T
    for i in range(0,T-1)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        print(i)
        last_t = torch.full((1,), i + 1, device=device, dtype=torch.long)
        upper_bound = 1 if i % (resampling_steps * res_freq) != 0  else resampling_steps
        for j in range(upper_bound):
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            img = sample_timestep(img, t)
            # show_tensor_image(img[0].cpu())
            # plt.show()
            # Edit: This is to maintain the natural range of the distribution
            img = img * (1 - mask) + mask * apply_noise(original_img, t, device)[0] if i != 0 else img * (1 - mask) + mask * original_img
            if j < upper_bound - 1:
                img = apply_noise(img, last_t, device)[0]
        img = torch.clamp(img, -1.0, 1.0)
        
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img[0].cpu())
    plt.show()
    return img           
@torch.no_grad()
def sample_plot_image_for_inpainting_with_resampling_faces(mask, original_img, img, resampling_steps = 1, jumping_steps = 1):
    # Sample noise
    img_size = IMG_SIZE
    if img is None:
        img = torch.randn((1, 3, img_size, img_size), device=device)
    # plt.figure(figsize=(15,15))
    # plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)
    img = apply_noise(img, torch.tensor([499], device = device, dtype = torch.long), device)[0] # get x_T
    u = 0
    for i in range(0,T)[::-1]:
        print(i)
        u += 1
        t = torch.full((1,), i, device=device, dtype=torch.long)
        if len(img.shape) == 3:
                img = img.unsqueeze(0)
        if (i != T-2 and u == jumping_steps ): # i is 489 for example
            u = 0
            img = sample_timestep(img, t)
            img = img * (1 - mask) + mask * apply_noise(original_img, t, device)[0] if i != 0 else img * (1 - mask) + mask * original_img
            for j in range(resampling_steps - 1):
                for k in range(jumping_steps - 1): # 0-9
                    
                    current_t = torch.full((1,), i + k + 1, device=device, dtype=torch.long)
                    img = apply_noise_from_previous(img, current_t, device)[0]
                    print(i + k + 1) # the timestep of img
                for k in range(jumping_steps-1): #0 - 9
                    
                    current_t = torch.full((1,), i + jumping_steps - 1 - k - 1, device=device, dtype=torch.long)
                    img = sample_timestep(img, current_t)
                    img = img * (1 - mask) + mask * apply_noise(original_img, current_t, device)[0] if i != 0 else img * (1 - mask) + mask * original_img
                    print(i + jumping_steps - 1 - k - 1)
            img = torch.clamp(img, -1.0, 1.0)
        else:
            img = sample_timestep(img, t)
            img = img * (1 - mask) + mask * apply_noise(original_img, t, device)[0] if i != 0 else img * (1 - mask) + mask * original_img
        
        if i == 0:
            img = torch.clamp(img, -1.0, 1.0)
        
    return img           
@torch.no_grad()
def sample_plot_image_for_inpainting_with_resampling_faces_old(mask, original_img, img, resampling_steps = 1, jumping_steps = 1):
    # Sample noise
    img_size = IMG_SIZE
    if img is None:
        img = torch.randn((1, 3, img_size, img_size), device=device)
    # plt.figure(figsize=(15,15))
    # plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)
    img = apply_noise(img, torch.tensor([499], device = device, dtype = torch.long), device)[0] # get x_T
    for i in range(0,T-1)[::-1]:
        u = 0
        while True:
            u += 1
            t = torch.full((1,), i, device=device, dtype=torch.long)
            print(i)
            last_t = torch.full((1,), i + 1, device=device, dtype=torch.long)
            
            # if upper_bound > 1 and (j == 0 or j == upper_bound - 1):
            #     show_tensor_image(img.cpu())
            #     plt.show()
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            img = sample_timestep(img, t)
            # show_tensor_image(img[0].cpu())
            # plt.show()
            # Edit: This is to maintain the natural range of the distribution
            # mask * img returns the known pixels
            img = img * (1 - mask) + mask * apply_noise(original_img, t, device)[0] if i != 0 else img * (1 - mask) + mask * original_img
            if i % jumping_steps != 0:
                break
            if u == resampling_steps:
                # if u != 1:
                #     img = torch.clamp(img, -1.0, 1.0)
                break
            img = apply_noise_from_previous(img, last_t, device)[0] # img is x t-1, last_t is t

        if i % 150 == 0:
            img = torch.clamp(img, -1.0, 1.0)
            
        # if i % stepsize == 0:
        #     plt.subplot(1, num_images, int(i/stepsize)+1)
        #     show_tensor_image(img[0].cpu())
    #plt.show()
    return img           
data_loader = load_transformed_dataset()
#data_loader = load_cifar_dataset()
#image = next(iter(data_loader))[0][0]
#image = load_specific_image("thomas.jpeg")
# image, mask = apply_random_mask(image)
# show_tensor_image(image.cpu())
# plt.show()
# img = image.clone()
# img = sample_plot_image_for_inpainting_with_resampling_faces(mask.to(device), image.to(device), img.to(device), 5, 10)
# show_tensor_image(img.cpu())
# plt.show()

for i in range(10):
    image = next(iter(data_loader))[0][0]
    #sample_plot_image()
    #image = load_specific_image("thomas.jpeg")
    image, mask = apply_random_mask(image)
    #show_tensor_image(image.cpu())
    img = image.clone()
    #plt.show()
    plt.figure(figsize=(15,15))
    plt.axis('off')
    parameters = [ (2,10), (2,15), (5,10), (5,15), (5,20), (10,10), (10,15), (10,20)]
    img1 = sample_plot_image_for_inpainting_with_resampling_faces(mask.to(device), image.to(device), img.to(device), parameters[0][0], parameters[0][1])
    img2 = sample_plot_image_for_inpainting_with_resampling_faces(mask.to(device), image.to(device), img.to(device), parameters[1][0], parameters[1][1])
    img3 = sample_plot_image_for_inpainting_with_resampling_faces(mask.to(device), image.to(device), img.to(device), parameters[2][0], parameters[2][1])
    img4 = sample_plot_image_for_inpainting_with_resampling_faces(mask.to(device), image.to(device), img.to(device), parameters[3][0], parameters[3][1])
    img5 = sample_plot_image_for_inpainting_with_resampling_faces(mask.to(device), image.to(device), img.to(device), parameters[4][0], parameters[4][1])
    img6 = sample_plot_image_for_inpainting_with_resampling_faces(mask.to(device), image.to(device), img.to(device), parameters[5][0], parameters[5][1])
    img7 = sample_plot_image_for_inpainting_with_resampling_faces(mask.to(device), image.to(device), img.to(device), parameters[6][0], parameters[6][1])
    img8 = sample_plot_image_for_inpainting_with_resampling_faces(mask.to(device), image.to(device), img.to(device), parameters[7][0], parameters[7][1])
    images = [image,img1, img2, img3, img4, img5, img6, img7, img8]
    for index,img in enumerate(images):
        plt.subplot(2,len(images) // 2 + 1,index + 1)
        show_tensor_image(img.cpu())
        if index >= 1:
            plt.title(f"r = {parameters[index-1][0]}, j = {parameters[index-1 ][1]}")
    plt.savefig(f"E:\\Thesis Experiments 2\\Faces_{i}_Inpainted.png")
    print(f"Saved figure {i}")
#plt.show()


# for i in range(1, 15):
#     image = next(iter(data_loader_test))[0][0]
#     image, mask = apply_random_mask(image)
#     save_tensor_image(image.cpu(), file_path, f"Figure_{i}_CIFAR_test_normalfreq_vs_doublefreq")
#     #show_tensor_image(image)
#     #plt.show()
#     # plt.figure(figsize=(15,15))
#     # plt.axis('off')
#     # num_images = 10
#     # stepsize = int(T/num_images)
#     # sample_plot_image()
#     # for idx in range(0, T, stepsize):
#     #     t = torch.Tensor([idx]).type(torch.int64)
#     #     #plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
#     #     img, noise = apply_noise_with(image, t, mask)
#     #     #show_tensor_image(img)
#     img = image.clone()
#     # for i in range(20):
#     #     #show_tensor_image(image.cpu())
#     #     apply = apply_noise(img.to(device), torch.tensor([499], device = device, dtype = torch.long), device) # get x_t
#     #     img = apply[0].unsqueeze(0) if i == 0 else apply[0]
#     #     img = sample_plot_image_for_inpainting(mask.to(device), image.to(device), img)
#     #     #show_tensor_image(img.cpu())
#     #     #plt.show()
#     #     image = img
#     img = sample_plot_image_for_inpainting_with_resampling(mask.to(device), image.to(device), img.to(device), 40, 1)
#     save_tensor_image(img.cpu(), file_path, f"Figure_{i}_CIFAR_Inpainted_test_40_freq_1")
#     img = image.clone()
#     img = sample_plot_image_for_inpainting_with_resampling(mask.to(device), image.to(device), img.to(device), 20, 0.5)
#     save_tensor_image(img.cpu(), file_path, f"Figure_{i}_CIFAR_Inpainted_test_20_freq_0.5")
#     #plt.show()