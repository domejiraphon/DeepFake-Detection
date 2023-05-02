# standard libraries
import copy
import os
import sys
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# external libraries
from torch.utils.tensorboard import SummaryWriter
import skimage.io as io 
import warnings
import glob 
from torchvision import transforms
warnings.filterwarnings("ignore")
import dataset
import model as M
import matplotlib.pyplot as plt 
from utils import get_visualizer, heat_map, EarlyStopping
from loguru import logger
from PIL import Image
import argparse
import cv2
import dataset
from scipy import interpolate
# internal libraries
from utils import utils
from utils.loss_utils import cross_entropy
from tqdm import tqdm
from utils import cmd_args_utils, RetinaGetFace
import argparse 
import pandas as pd 

from utils import Segment, Face_Cutout, GetFace, Face_Mesh, RandomDownScale, \
    RetinaGetFace, DeepLabV3_Segment, Occlusion

from mmseg.apis import init_segmentor, inference_segmentor
import torchvision.models
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', 
            type=str, 
            default="/vast/jy3694/captcha_dataset",
            help="dataset to use as an attacker")
parser.add_argument('--test', 
            choices=["face_detector", 
                "face_keypoints",
                "face_segmentation"], 
            default="face_detector", 
            type=str,
            help="select which module to attack")
parser.add_argument('--segment_cfg_file', type=str, 
            default='utils/openmmlab/deeplabv3plus_celebA_train_wo_natocc_wsot.py',
            help="config fule from the one govind as ked in face occlusion issue")
parser.add_argument('--segment_pth_file', type=str, 
            default="utils/openmmlab/iter_27600.pth",
            help="pretrained model from the one govind asked in face occlusion issue")
parser.add_argument('--dst_seg_path', 
            type=str, 
            default="/vast/jy3694/captcha_segment_dataset",
            help="dataset to store the ground truth for segmentation map\
                In this case, I used the model from face occlusion to do so")
parser.add_argument('--dst_path', 
            type=str, 
            default="degrade_result",
            help="where to store the output")
parser.add_argument('--img_shape', 
            type=int, 
            default=512,
            help="image size")
parser.add_argument('--batch_size', 
            type=int, 
            default=1,
            help="batch size to attack")
parser.add_argument('--job', 
            type=int, 
            default=0,
            help="the index of impersonator")
parser.add_argument('--num_workers', 
            type=int, 
            default=8,
            help="Number of workers for the dataloader")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retina = RetinaGetFace(image_size=512, 
                            device=device)

class dataset(Dataset):
    def __init__(self, image_path):
        """Initialize and preprocess the Swapping dataset."""
        self.image_path = image_path
        self.preprocess()
        img_transform = [transforms.ToTensor()]
        self.img_transform = transforms.Compose(img_transform)
      
    def preprocess(self, sel_every=None):
        """Preprocess the Swapping dataset."""
        def find_path(args):
            subfolder = os.path.join(self.image_path, args)
            paths = sorted(glob.glob(subfolder))[1:]

            return paths 
        if args.test == "face_segmentation":
            self.dataset = sorted(glob.glob(os.path.join(self.image_path, "*")))
            self.dataset = self.dataset[int(args.job*len(self.dataset)/30): int((args.job+1)*len(self.dataset)/30)]
            return
        paths = find_path("*/")
       
        self.dataset = []
        for i, dir_item in tqdm(enumerate(paths)):
            join_path = sorted(glob.glob(os.path.join(dir_item, '*')))
            self.dataset += join_path
    
        if sel_every:
            self.dataset = self.dataset[::sel_every]
       

    def __getitem__(self, index):
        filename = self.dataset[index]

        image = Image.open(filename)

        return {"img": self.img_transform(image),
                "name": filename}
      

    def __len__(self):
        """Return the number of images."""
        return len(self.dataset)

class method:
    def __init__(self):
        if args.test == "face_detector":
            self.model = GetFace(image_size=args.img_shape, 
                        device=device)
        elif args.test == "face_keypoints":
            self.model = Face_Mesh()
        elif args.test == "face_segmentation":
            self.model = init_segmentor(args.segment_cfg_file, 
                args.segment_pth_file, device=device)
           
    def run(self, path):
        count = []
        with torch.no_grad():
            dataloader = self.load_data(path)
            if args.test == "face_detector":
                for batch in dataloader:
                    img = batch["img"].to(device)
                    img *= 255
                    img = img.permute([0, 2, 3, 1])
                    out = self.model(img, degrade=True)
                    count.append(1) if out is not None  else count.append(0)
                return count
            
            elif args.test == "face_keypoints":
                for batch in dataloader:
                    img = batch["img"].permute([0, 2, 3, 1]).numpy().squeeze()
                    img = Image.fromarray(img).convert('RGB')
                   
                    keypoints = self.model(img)
                    if keypoints is None:
                        count.append(0)
                        continue
                 
                    img = np.array(img) / 255 
                   
                    count.append(keypoints.shape[0])
                return count
            elif args.test == "face_segmentation":
                for batch in dataloader:
                    
                    img = batch["img"].permute([0, 2, 3, 1]).numpy().squeeze()
                    img = np.uint8(255 * img)
                
                    out = inference_segmentor(self.model, img)
                    print(batch["name"][0].split('/')[-1])
                    plt.imsave(os.path.join(args.dst_seg_path, batch["name"][0].split('/')[-1]), out[0])
                  

    @staticmethod
    def load_data(path):
        if not os.path.exists(path): return None 
     
        testdata = dataset(
                            image_path=path,
                            )
        testset = DataLoader(testdata,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers)
        return testset


def plot_graph(x, y, xlabel, ylabel, title, path, name):
    plt.plot(x, y, 'r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if not os.path.exists(path):
        os.system(f"mkdir -p {path}")
    plt.savefig(os.path.join(path, name))
    plt.clf()

@logger.catch 
def main():
    model = method()
    for chal in sorted(glob.glob(os.path.join(args.dataset_path, "*")))[6:7]:
        for sub in sorted(glob.glob(os.path.join(chal, "*"))):
         
            out = model.run(sub)
            if args.test == "face_detector": 
                i, j = sub.split('/')[-2:]
                plot_graph(np.arange(len(out)), 
                        out, "frames", "detect", 
                        f"Face detector: Challenge {i}, Subject: {j}", 
                        os.path.join(args.dst_path, i, j), "detect_face.jpg")
                np.savez(os.path.join(args.dst_path, i, j, f"face_detector.npz"), detect=out)
            elif args.test == "face_keypoints":
                i, j = sub.split('/')[-2:]
                plot_graph(np.arange(len(out)), 
                        out, "frames", "detect", 
                        f"Face detector: Challenge {i}, Subject: {j}", 
                        os.path.join(args.dst_path, i, j), "keypoints.jpg")
                np.savez(os.path.join(args.dst_path, i, j, f"face_keypoints.npz"), detect=out)



if __name__=='__main__':
    main_seg()

