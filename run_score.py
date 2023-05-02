#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

# standard libraries
import copy
import os
import sys
from operator import itemgetter
sys.path.append('/home/gm2724/captcha_net/anomaly_detector/')
import numpy as np
import pandas as pd
import torch
torch.multiprocessing.set_start_method('spawn')
import torch.nn as nn
from torch.utils.data import DataLoader
# external libraries
import skimage.io as io 
import warnings
import glob 
from torchvision import transforms
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt 
from utils import get_visualizer, heat_map, EarlyStopping
from loguru import logger
import argparse
import cv2
from scipy import interpolate
from utils import utils
from utils.loss_utils import cross_entropy
from tqdm import tqdm
from utils import cmd_args_utils, RetinaGetFace
import argparse 
import pandas as pd 


from pathlib import Path

from dataset import *
import model as M
sys.path.append("./utils")
from efficientnet_pytorch import EfficientNet

from utils import get_visualizer, heat_map
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets



def increase_momentum(model):
    if isinstance(model, nn.BatchNorm2d):
        model.momentum = 0.1

class Detector(nn.Module):
    def __init__(self, num_classes=2, path2pretrained=None):
        super().__init__()
        num_classes = [num_classes, 1]
        self.backbone = EfficientNet.from_pretrained("efficientnet-b4",
                            advprop=True,
                            num_classes=num_classes,
                            in_channels=3)
        self.backbone.apply(increase_momentum)
        if path2pretrained and os.path.exists(path2pretrained):
            logger.info(f"Load pretrained from {path2pretrained}")
            checkpoint = torch.load(path2pretrained)
            self.backbone.load_state_dict(checkpoint["model_state_dict"])
            cnn_sd = {key.replace("net.", ""): value for key, value in
                      cnn_sd.items()}
        else:
            logger.info("No pretrained")
            
    def forward(self, img):
        logits = self.backbone(img)
        if isinstance(logits, tuple):
            logits, score = logits
            return logits, score
       
        return logits



batch_size=128
concat_segment=False
dataset_dir='/scratch/jy3694/data/faceforensics2/original_sequences/youtube/raw/frames/'
dataset_path='/vast/jy3694/all_dataset3'
deeplabv3_ckpt='./runs/pretrained/deeplab_v3/latest_deeplabv3plus_mobilenet_face_occlude_os16.pth'
deeplabv3_model='deeplabv3plus_mobilenet'
deeplabv3_num_classes=2
devices='cuda'
exp_folder='runs'
img_shape=224
log_every=10
lpips=True
lr=0.001
lr_decay=0.1
model_dir='regress_occlusion'
no_val=False
num_classes=2
num_dataset=-1
num_epochs=400
num_workers=0
occluder_img_dataset=['/face_occlusion_generation/object_image_sr/face_occlusion_generation/11k_hands_img/']
occluder_mask_dataset=['/face_occlusion_generation/object_mask_x4/face_occlusion_generation/11k_hands_masks/']
output_stride=16
path2pretrained='./runs/pretrained/ours/sbi.tar'
plot_graph=True
print_every=10
print_header_every=100
pseudo_labels=False
real_path='/vast/jy3694/captcha_dataset'
restart=False
save_every=100
seed=0
segment_ckpt='./checkpoints/latest_deeplabv3plus_mobilenet_face_occlude_os16'
segment_model='deeplabv3plus_mobilenet'
shuffle=True
test=False
test_visualization_dir='e1'
testset_dir='../assets/test_samples3'
use_cutout=False
use_deeplab=False
use_facenet=False
use_keypoint=False
use_occlusion=False
use_regression=True
use_retinafacenet=True
use_segment=False
use_self_blended=False
val_every=100
val_num_images=256
visualization_type='eigencam'
visualized=False
image_size = 224



model_dir = "runs/8_1/dfdc_2k_more_skewed"
device = "cuda"
ckpt = os.path.join(model_dir, "ckpt.pt")

model = Detector(path2pretrained=None)


if os.path.exists(ckpt):
    checkpoint = torch.load(ckpt, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    
model = model.to(device)
model.eval()
retina = RetinaGetFace(image_size=512, device=device)



class CropTransform:
    """Rotate by one of the given angles."""
    @staticmethod
    def crop_image_only_outside(img, tol=0):
        # img is 2D or 3D image data
        # tol  is tolerance
        mask = img>tol
        if img.ndim==3:
            mask = mask.all(2)
        m,n = mask.shape
        mask0,mask1 = mask.any(0),mask.any(1)
        col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
        row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
        return img[row_start:row_end,col_start:col_end]
    
    def __call__(self, image):
        image = np.array(image)
        return self.crop_image_only_outside(image, tol=5)
    
img_transform = transforms.Compose([CropTransform(), transforms.ToTensor(), transforms.Resize((image_size, image_size), )])

def get_features(filename):
    image = Image.open(filename)
    img = retina(image, return_pil=True)
    return img_transform(img)



from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader


class dataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.imgs = list(Path(img_dir).glob("*.jpg"))
        self.transform = transform
        self.labels = []
        for img_path in self.imgs:
            if "fake" in str(img_path):
                self.labels.append(1)
            else:
                self.labels.append(0)
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        features = get_features(self.imgs[idx])
        label = self.labels[idx]
        return features, label
    




def infer_batch(dataloader):
    with torch.no_grad():
        scores = []
        preds = []
        for batch, labels in dataloader:
            img = batch.to(devices)
            logits = model(img)
            logits, reg_score = logits
            reg_score = torch.sigmoid(reg_score)
            pred = logits.softmax(1)[:, 1]
            scores.append(reg_score.squeeze().cpu().numpy())
            preds.append(pred.cpu().numpy())
    return np.concatenate(scores), np.concatenate(preds)



import time 
results_path = "/vast/jy3694/results_simswap"
imp = int(sys.argv[1])
imp += 47
print("impersonator = ", imp)
for chal in range(16):
    for sub in range(47):
        if os.path.isfile(os.path.join(results_path, f"{imp}_{chal}_{sub}.npz")):
            continue
        try:
            t = time.time()
            #path = f"/challenges/{imp}/{chal}/{sub}/fake"
            #path = f"/vast/jy3694/all_dataset_simswap/{imp}/{chal}/{sub}"
            path = f"/vast/jy3694/all_dataset_simswap_celeb/{imp}/{chal}/{sub}"
            dataset = dataset(img_dir=path)
            dataloader = DataLoader(dataset, batch_size=128, num_workers=0, shuffle=False)
            s, p = infer_batch(dataloader)
            np.savez(os.path.join(results_path, f"{imp}_{chal}_{sub}.npz"), scores=s, preds=p)
            print(time.time() - t)
        except:
            print (f"skipped - {imp} - {chal} - {sub}")
        
       


