from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from inpainting import sample_plot_image_for_inpainting_with_resampling_faces
from main import device
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
from utils import show_tensor_image
from torchvision import transforms

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...)
):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # [-1,1]
    ])
    image_tensor = preprocess(pil_image)

    _, H, W = image_tensor.shape
    mask = torch.ones((H, W), dtype=image_tensor.dtype, device=device)
 
    mask[y1:y2, x1:x2] = 0
    mask = mask.unsqueeze(0).expand_as(image_tensor)
    img = image_tensor.clone()
    img = sample_plot_image_for_inpainting_with_resampling_faces(
        mask.to(device), image_tensor.to(device), img.to(device), 5, 15
    )
    
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
    ])
    img = reverse_transforms(img[0].cpu())
    print(JSONResponse(content={"image": img.tolist()}))
    return JSONResponse(content={"image": img.tolist()})