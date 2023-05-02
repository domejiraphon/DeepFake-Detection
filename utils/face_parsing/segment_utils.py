import sys

sys.path.append("./utils/face_parsing")
from utils.face_parsing.model import BiSeNet

import torch
import torchvision.transforms.functional as F
import os
import numpy as np
from PIL import Image
import PIL
import torchvision.transforms as transforms
import cv2
from loguru import logger


class Segment(torch.nn.Module):
    def __init__(self, img_shape, device):
        super().__init__()
        n_classes = 19
        self.device = device
        self.img_shape = img_shape
        self.net = BiSeNet(n_classes=n_classes).to(self.device)
        parent_path = "/".join(os.path.realpath(__file__).split("/")[:-1])
        ckpt_path = os.path.join(parent_path, 'res/cp/79999_iter.pth')
        logger.info("Load BiSeNet for segmentation map.")
        self.net.load_state_dict(torch.load(ckpt_path))
        self.net.eval()

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def forward(self, img, return_for_visualization=False):

        if type(img) in [PIL.JpegImagePlugin.JpegImageFile,
                         PIL.Image.Image]:
            image = self.to_tensor(img)[None]
        else:
            image = img[None]

        image = F.resize(image, (512, 512))
        image = image.to(self.device)
        out = self.net(image)[0]
        out = F.resize(out, (self.img_shape, self.img_shape))

        if return_for_visualization:
            parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
            segment_map = self.vis_parsing_maps(img, parsing, stride=1)
            return segment_map
        else:
            # out = torch.argmax(out, dim=1, keepdims=True)[0]
            out = out[0]
            return out

    def vis_parsing_maps(self, im, parsing_anno, stride):
        # Colors for all 20 parts
        part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 0, 85], [255, 0, 170],
                       [0, 255, 0], [85, 255, 0], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]

        im = np.array(im)
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride,
                                      fy=stride,
                                      interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros(
            (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        return vis_parsing_anno_color
