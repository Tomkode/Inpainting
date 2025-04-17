import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from utils import apply_noise, get_index_from_list, show_tensor_image, T, betas, sqrt_one_minus_alphas_barred, sqrt_recip_alphas, posterior_variance
from model import SimpleUnet
from dataset import data_loader, BATCH_SIZE, IMG_SIZE
from torch.optim import Adam
from tqdm import tqdm
from main import sample_plot_image
import os

model = SimpleUnet()

CHECKPOINT_PATH = "model_ffhq_full.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
# Load existing checkpoint if available
start_epoch = 0  # Default: start from scratch
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Resume from the next epoch
    print(f"Resuming training from epoch {start_epoch}")

for i in range(10):
    sample_plot_image()