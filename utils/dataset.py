# standard libraries
import glob
import os
import random
from typing import Dict 
import albumentations as alb
import cv2
import numpy as np
# external libraries
import torch
from PIL import Image
from loguru import logger
from torch.utils.data import Dataset
# import imgaug.augmenters as iaa
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt 
import torchvision.transforms.functional as F
import random
# internal libraries
from utils import Segment, Face_Cutout, GetFace, Face_Mesh, RandomDownScale, \
    RetinaGetFace, DeepLabV3_Segment, Occlusion


class AffectnetDataset(Dataset):
    def __init__(self, image_path, image_size, num_dataset=1000,
                 use_keypoint=False, use_facenet=False, use_segment=False,
                 device="cuda"):
        # use_facenet=False
        self.image_path = image_path
        self.use_facenet = use_facenet
        self.num_dataset = num_dataset
        self.use_keypoint = use_keypoint
        self.use_segment = use_segment

        img_transform = [transforms.ToTensor(),
                         transforms.Resize((image_size, image_size), )]

        if self.use_facenet:
            img_transform = [GetFace(image_size=image_size, device=device),
                             *img_transform]

        if self.use_segment:
            self.segment = Segment(device=device)
            self.resize = transforms.Compose(
                [transforms.Resize((image_size, image_size))])

        if self.use_keypoint:
            self.face_mesh = Face_Mesh()
        # self.mtcnn = MTCNN(image_size=image_size, device="cuda", post_process=False)#.cuda()
        self.img_transform = transforms.Compose(img_transform)
        self.preprocess()

    def preprocess(self):
        """Preprocess the Swapping dataset."""
        logger.info("processing Affectnet dataset images...")

        subfolder = os.path.join(self.image_path, '*/')
        paths = sorted(glob.glob(subfolder))
        self.expression = [folder_name.split("/")[-2] for folder_name in paths]
        self.dataset, self.labels = [], []
        for i, dir_item in tqdm(enumerate(paths)):
            join_path = sorted(glob.glob(os.path.join(dir_item, '*')))
            self.labels.append(i * torch.ones(len(join_path)))
            self.dataset += join_path

        self.labels = torch.cat(self.labels, 0).long()
        if self.num_dataset > 0:
            sel_index = np.random.randint(0, len(self.dataset),
                                          self.num_dataset).astype(int)
            self.dataset = np.array(self.dataset)[sel_index]
            self.labels = self.labels[sel_index]

        logger.info(f"Total dirs number: {len(self.dataset)}")

    def __getitem__(self, index):
        filename = self.dataset[index]

        image = Image.open(filename)
        labels = self.labels[index]
        features = {"img": self.img_transform(image),
                    "expression_labels": labels}
        if self.use_segment:
            segment_map = self.segment(image)
            features["segmentation"] = self.resize(segment_map)
        if self.use_keypoint:
            keypoints = self.face_mesh(image)
            features["keypoints"] = keypoints
        return features

    def __len__(self):
        """Return the number of images."""
        return len(self.dataset)

class CaptchaDataset(Dataset):
    def __init__(self,
                 image_path, image_size, num_dataset=-1,
                 use_keypoint=False, use_facenet=False, use_segment=False,
                 use_self_blended=False, use_cutout=False,
                 use_retinafacenet=False, device="cuda", batch_size=-1,
                 test=False, config=None, data_type=None, 
                 verbose=True, **kwargs):
        """Initialize and preprocess the Swapping dataset."""
        self.image_path = image_path
        self.use_facenet = use_facenet
        self.use_retinafacenet = use_retinafacenet
        self.num_dataset = num_dataset
        self.use_keypoint = use_keypoint
        self.use_segment = use_segment
        self.use_self_blended = use_self_blended
        self.use_cutout = use_cutout
        self.image_size = image_size
        self.data_type = data_type
        self.config = config
        if 'sel_every' in kwargs:
            self.preprocess(kwargs['sel_every'])
        else:
            self.preprocess()
        img_transform = [transforms.ToTensor(),
                         transforms.Resize((image_size, image_size), )]

        if self.use_facenet:
            self.getface = GetFace(image_size=image_size, device=device)
        elif "retinaface" in kwargs:
            self.getface = kwargs["retinaface"]
          
        elif self.use_retinafacenet:
            get_org = np.array(Image.open(self.dataset[0]))
            self.getface = RetinaGetFace(image_size=max(get_org.shape), device=device)

        if self.use_segment:
            self.segment = Segment(device=device, img_shape=self.image_size)

        if self.config.use_deeplab:
            self.segment = DeepLabV3_Segment(device=device,
                                             img_shape=self.image_size,
                                             model=self.config.deeplabv3_model,
                                             num_classes=self.config.deeplabv3_num_classes,
                                             output_stride=self.config.output_stride,
                                             ckpt=self.config.deeplabv3_ckpt)
            # self.resize = transforms.Compose([transforms.Resize((image_size, image_size))])
        if self.config.use_occlusion:
            self.occlusion = Occlusion(config, self.segment, image_size)

        if self.use_keypoint: self.face_mesh = Face_Mesh()

        if self.use_self_blended:
            self.np2tensor = transforms.Compose([transforms.ToTensor()])
            self.resize = transforms.Compose([transforms.Resize(
                                                  (image_size, image_size))])

        if self.use_cutout:
            self.face_cutout = Face_Cutout(image_size=self.image_size)

        self.img_transform = transforms.Compose(img_transform)
        
    def preprocess(self, sel_every=None):
        """Preprocess the Swapping dataset."""
        def find_path(args):
            subfolder = os.path.join(self.image_path, args)
            paths = glob.glob(subfolder)
            return paths 
        paths = find_path("*/")
        if len(paths) == 0: 
            self.dataset = find_path("*")
        else:
            self.dataset = []
            for i, dir_item in tqdm(enumerate(paths)):
                join_path = sorted(glob.glob(os.path.join(dir_item, '*')))
                self.dataset += join_path
        if self.data_type == "train":
            self.dataset = self.dataset[:-self.config.val_num_images]
            #self.dataset = self.dataset[:64]
        elif self.data_type == "val":
            self.dataset = self.dataset[-self.config.val_num_images:]
            #self.dataset = self.dataset[:64]
        """
        elif self.data_type == "test":
            real_path = [p for p in self.dataset if "real" in p]
            fake_path = [p for p in self.dataset if "fake" in p]
            self.dataset = []
            for (real_p, fake_p) in zip(real_path, fake_path):
                self.dataset.append(real_p)
                self.dataset.append(fake_p)
        """
        if sel_every:
            self.dataset = self.dataset[::sel_every]
       
        if self.num_dataset > 0 and len(self.dataset) > self.num_dataset:
            sel_index = np.random.randint(0, len(self.dataset),
                                          self.num_dataset).astype(int)
            self.dataset = np.array(self.dataset)[sel_index]
        
        logger.info(
            f'Finished preprocessing the {self.data_type}set, total dirs number: {len(self.dataset)}...')

    def __getitem__(self, index):
        filename = self.dataset[index]

        image = Image.open(filename)

        features = {}
        #if self.data_type == "test":
        if True:
            img = self.getface(image)
            if img is None:
                img = image
                features['detect_face'] = 0
            else:
                features['detect_face'] = 1
            features['img'] = self.img_transform(img)
            
            features['labels'] = 0 if "real" in filename else 1
            
            features['name'] = filename
            if self.use_segment or self.config.use_deeplab:
                segment_map = self.segment(features['img'])
                # segment_map = self.segment(image)
                features["segmentation"] = segment_map
            if self.config.use_regression:
                features['pseudo_labels'] = 0 if "real" in filename else 1
            return features
        found_face = True 
        if self.use_facenet:
            image = self.getface(image)
        elif self.use_retinafacenet:
            image = self.getface(image, return_pil=True)
        if image is None: 
            image = Image.open(filename)
            found_face = False
        if self.use_keypoint and (not self.use_self_blended):
            keypoints = self.face_mesh(image)
            features["keypoints"] = keypoints

        use_occlusion = False
        if self.use_self_blended and self.use_keypoint and found_face:
            rand = np.random.choice(2, p=[0.5, 0.5])

            img, image_blended, mask = self.self_blending(image,
                                                          self.image_size)
            if img is None: found_face = False
            if found_face:
                img = img if rand == 0 else image_blended
                if self.config.use_occlusion:
                    if np.random.choice(2) == 1:
                        use_occlusion = True
                        img, features['segmentation'] = self.occlusion(img)
                        if img is None:
                            found_face = False 
               
                if found_face:
                    features['img'] = img
                    features['labels'] = 0 if rand == 0 else 1
                    features['detect_face'] = 1
        if not found_face:
            features['img'] = self.img_transform(image)
            features['labels'] = -1
            features['detect_face'] = 0

        if self.use_cutout:
            features['img'] = self.np2tensor(self.face_cutout(features['img']))

        if (self.use_segment or self.config.use_deeplab) and (
        not use_occlusion):
            segment_map = self.segment(features['img'])
            # segment_map = self.segment(image)
            features["segmentation"] = segment_map
        
        return features


    def __len__(self):
        """Return the number of images."""
        return len(self.dataset)

    def get_torch_transforms(self, img):
        hue = np.random.uniform(low=-0.02, high=0.02)
       
        sat, bright, sharp = np.random.uniform(low=0.9, high=1.1, size=3)

        rgb_shifted = 20/255 * (torch.rand(3)*2-1).to(img.get_device())
        
        img = torch.clamp(img + rgb_shifted[:, None, None] * torch.ones_like(img), 0.0, 1.0)
        
        img = F.adjust_hue(img, hue_factor=hue)
        img = F.adjust_saturation(img, saturation_factor=sat)
        
        img = F.adjust_brightness(img, brightness_factor=bright)
        
        if np.random.uniform() < 0.5:
            img = F.autocontrast(img)

        if np.random.uniform() < 0.5:
            img = F.adjust_sharpness(img, sharpness_factor=sharp)
        else:
            kernel_size = random.choice([3, 5, 7, 9])
            img = F.gaussian_blur(img, kernel_size=kernel_size)
        
        return torch.clamp(img, 0.0, 1.0) 

    def randaffine(self, img, mask):
        """
        f = alb.Affine(
            translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
            scale=[0.95, 1 / 0.95],
            fit_output=False,
            p=1)
        """
        np_img = img.permute([1, 2, 0]).detach().cpu().numpy()
        g = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )
        H, W = img.shape[1:]

        img_translate = list(np.random.randint(low=0, high=int(0.03*W), size=2))
        img_scale = np.random.choice([0.95, 1 / 0.95])
        img = F.affine(
            img=img, 
            angle=0,
            translate=img_translate,
            scale=img_scale,
            shear=0,
        )
        
        np_mask = mask.detach().cpu().numpy()
        transformed = g(image=np_img, mask=np_mask)
        mask = self.np2tensor(transformed['mask']).to(img.get_device())
        
        return img, mask

    def self_blending(self, img, image_size, device="cuda"):
        img = self.np2tensor(img).to(device)
        img = self.resize(img)
        
       
        H, W = img.shape[1:]
        np_img = (img.permute([1, 2, 0]).detach().cpu().numpy() * 255).astype(np.uint8)
        
        mask = self.face_mesh.create_mask(np_img)
        
        if mask is None: return None, None, None
        mask = torch.from_numpy(mask).float()
        source = torch.clone(img)
      
        if np.random.rand() < 0.5:
            source = self.get_torch_transforms(source)
        else:
            img = self.get_torch_transforms(img)
           
    

        def get_torch_blend_mask(mask):
            H, W = mask.shape[1:]
            size_h = np.random.randint(low=192, high=257)
            size_w = np.random.randint(low=192, high=257)
            
            mask = F.resize(mask, [size_w, size_h])
            kernel_1 = random.randrange(5, 26, 2)
            kernel_1 = (kernel_1, kernel_1)
            kernel_2 = random.randrange(5, 26, 2)
            kernel_2 = (kernel_2, kernel_2)

            mask_blured = F.gaussian_blur(mask, kernel_size=kernel_1)
            mask_blured = mask_blured / (mask_blured.max())
            mask_blured[mask_blured < 1] = 0

            mask_blured = F.gaussian_blur(mask_blured, kernel_2,
                                            np.random.randint(5, 46))
            mask_blured = mask_blured / (mask_blured.max())
            mask_blured = F.resize(mask_blured, (W, H))
            return mask_blured

        def dynamic_blend(source, target, mask):
            mask_blured = get_torch_blend_mask(mask)
            blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
            blend_ratio = blend_list[np.random.randint(len(blend_list))]
            mask_blured *= blend_ratio
            img_blended = (mask_blured * source + (1 - mask_blured) * target)
            return img_blended, mask_blured
        #source [3. h, w]
        #mask [1, h, w]
        #img [3, h, w]
        source, mask = self.randaffine(source, mask)
        img_blended, mask = dynamic_blend(source, img, mask)

        return (img.detach().cpu(), 
            img_blended.detach().cpu(), 
            mask.squeeze().detach().cpu())

class SBI_Dataset(Dataset):
    def __init__(self, image_path, phase='train', image_size=224, 
                n_frames=8, config=None, device="cuda", ):
        
        assert phase in ['train','val','test']
        
        image_list,label_list=self.init_ff(image_path, phase,'frame',n_frames=n_frames)
       
        path_lm='/landmarks/' 
        label_list=[label_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
        image_list=[image_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
        if phase == "train":
            image_list = image_list[:-32]
            label_list = image_list[:-32]
        elif phase == "val":
            image_list = image_list[-32:]
            label_list = image_list[-32:]
        self.path_lm=path_lm
        print(f'SBI({phase}): {len(image_list)}')
       

        self.image_list=image_list

        self.image_size=(image_size,image_size)
        self.phase=phase
        self.n_frames=n_frames

        self.transforms=self.get_transforms()
        self.source_transforms = self.get_source_transforms()
        self.config = config 
        
        if self.config.use_deeplab:
            self.segment = DeepLabV3_Segment(device=device,
                                             img_shape=image_size,
                                             model=self.config.deeplabv3_model,
                                             num_classes=self.config.deeplabv3_num_classes,
                                             output_stride=self.config.output_stride,
                                             ckpt=self.config.deeplabv3_ckpt)
        if self.config.use_occlusion:
            self.occlusion = Occlusion(config, self.segment, image_size)
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        flag=True

        while flag:
            try:
                filename=self.image_list[idx]
                img=np.array(Image.open(filename))
                landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
                bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
                bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
                iou_max=-1
                for i in range(len(bboxes)):
                    iou=self.IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
                    if iou_max<iou:
                        bbox=bboxes[i]
                        iou_max=iou

                landmark=self.reorder_landmark(landmark)
                if self.phase=='train':
                    if np.random.rand()<0.5:
                        img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
                        
                img,landmark,bbox,__=self.crop_face(img,landmark,bbox,margin=True,crop_by_bbox=False)
                if self.config.use_regression:
                    (img_r, img_f, mask_f, 
                        transforms_dict) = self.self_blending(img.copy(),landmark.copy())
                else:
                    img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy())
                
                if self.phase=='train':
                    transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
                    img_f=transformed['image']
                    img_r=transformed['image1']

                
                (img_f,_,__,___,y0_new,
                    y1_new,x0_new,x1_new)=self.crop_face(img_f,landmark,bbox,margin=False,
                                crop_by_bbox=True,abs_coord=True,phase=self.phase)
                
                img_r=img_r[y0_new:y1_new,x0_new:x1_new]
                
                img_f=cv2.resize(img_f, self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                img_r=cv2.resize(img_r, self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                
                if self.config.use_occlusion:
                    
                    if random.random() < 0.5:
                        if self.config.use_regression:
                        
                            (occlude_f, segment_f,
                                occlude_r, segment_r) = self.occlusion(img_f, to_tensor=False, img2=img_r)
                            
                        else:
                            occlude_f, segment_f = self.occlusion(img_f, to_tensor=False)
                            occlude_r, segment_r = self.occlusion(img_r, to_tensor=False)
                        if not(occlude_f is None or occlude_r is None): 
                            img_f = occlude_f
                            img_r = occlude_r
                

                img_f=img_f.transpose((2,0,1))
                img_r=img_r.transpose((2,0,1))
                        
            
                
                
                flag=False
            except Exception as e:
                idx=torch.randint(low=0,high=len(self),size=(1,)).item()
               
        if self.config.use_regression:
            pseudo_label = 0
            for key, value in transforms_dict.items():
                pseudo_label += value
          
            pseudo_label = pseudo_label / (len(transforms_dict) - 1)
        
            return img_f, img_r, torch.tensor([1]).long(), pseudo_label, 0
        else:
            return img_f,img_r, torch.tensor([1]).long()

    def get_torch_transforms(self, img, device="cuda"):
  
        img = torch.from_numpy(img) / 255 
        img = img.permute([2, 0, 1]).to(device)
        config_dict = {"hue_min": -0.02,
                       "hue_no": 0,
                       "hue_max": 0.02,
                       "sat_bright_min": 0.9,
                       "sat_bright_no": 1,
                       "sat_bright_max": 1.1,
                       "rgb_shifted_min": -20/255,
                       "rgb_shifted_no": 0,
                       "rgb_shifted_max": 20/255,
                       "sharp_min": 0.9,
                       "sharp_no": 1,
                       "sharp_max": 1.1,
                       }
       
        hue = np.random.uniform(low=config_dict["hue_min"], high=config_dict["hue_max"])

        sat, bright = np.random.uniform(low=config_dict["sat_bright_min"], 
                                high=config_dict["sat_bright_max"], size=2)

        rgb_shifted = torch.zeros(3).to(img.get_device())
        if np.random.uniform() > 0.5:
            rgb_shifted = config_dict["rgb_shifted_max"] * (torch.rand(3)*2-1).to(img.get_device())
            img = torch.clamp(img + rgb_shifted[:, None, None] * torch.ones_like(img), 0.0, 1.0)
     
        img = F.adjust_hue(img, hue_factor=hue)
        img = F.adjust_saturation(img, saturation_factor=sat)

        img = F.adjust_brightness(img, brightness_factor=bright)

        contrast_factor = 0
        if np.random.uniform() > 0.5:
            img = F.autocontrast(img)
            contrast_factor = 1
        
        if np.random.uniform() > 0.5:
            sharp = np.random.uniform(low=config_dict["sharp_min"], 
                    high=config_dict["sharp_max"], size=1).item()
            img = F.adjust_sharpness(img, sharpness_factor=sharp)
            down = 0
        else:
            sharp = 1
            ratio_list = [2, 4]
            org_h, org_w = img.shape[1:]
            r = ratio_list[np.random.randint(len(ratio_list))]
            img = F.resize(img, (int(org_h / r), int(org_w/r)),
                    transforms.InterpolationMode.AREA)
            img = F.resize(img, (int(org_h), int(org_w)),
                    transforms.InterpolationMode.AREA)
            down = r / ratio_list[-1]
        img = torch.clamp(img, 0.0, 1.0).permute([1,2,0]).detach().cpu().numpy()
        img = img * 255
      
        if self.config.use_regression:
            r, g, b = rgb_shifted.detach().cpu().numpy()
           
            def map_score(score, factor):
                distance = np.abs(score - config_dict[f"{factor}_no"])
                distance = 0.5 * distance / (config_dict[f"{factor}_no"] - config_dict[f"{factor}_min"])
                if (score - config_dict[f"{factor}_no"]) > 0:
                    distance += 0.5
                return distance
            hue_factor = map_score(hue, "hue")
            sat_factor = map_score(sat, "sat_bright")
           
            bright_factor = map_score(bright, "sat_bright")

            sharp_factor = map_score(sharp, "sharp")
            down_factor = down 
            r_factor = map_score(r, "rgb_shifted")
            g_factor = map_score(g, "rgb_shifted")
            b_factor = map_score(b, "rgb_shifted")
            transforms_dict = {"hue_factor": hue_factor,
                                "sat_factor": sat_factor,
                                "bright_factor": bright_factor,
                                "sharp_factor": sharp_factor,
                                "r_factor": r_factor,
                                "g_factor": g_factor,
                                "b_factor": b_factor,
                                "contrast_factor": contrast_factor,
                                "down_factor": down_factor, 
                                }
            return img, transforms_dict
        return img

    def get_source_transforms(self):
        return alb.Compose([
                alb.Compose([
                        alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
                        alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
                        alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
                    ],p=1),

                alb.OneOf([
                    RandomDownScale(p=1),
                    alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                ],p=1),
                
            ], p=1.)
      
    def get_transforms(self):
        return alb.Compose([
            
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
            
        ], 
        additional_targets={f'image1': 'image'},
        p=1.)

    def randaffine(self,img,mask):
        f=alb.Affine(
                translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
                scale=[0.95,1/0.95],
                fit_output=False,
                p=1)
            
        g=alb.ElasticTransform(
                alpha=50,
                sigma=7,
                alpha_affine=0,
                p=1,
            )

        transformed=f(image=img,mask=mask)
        img=transformed['image']
        
        mask=transformed['mask']
        transformed=g(image=img,mask=mask)
        mask=transformed['mask']
        return img,mask
		
    def self_blending(self,img,landmark):
        H,W=len(img),len(img[0])
        if np.random.rand()<0.25:
            landmark=landmark[:68]
        exist_bi=False
        if exist_bi:
            logging.disable(logging.FATAL)
            mask=random_get_hull(landmark,img)[:,:,0]
            logging.disable(logging.NOTSET)
        else:
            mask=np.zeros_like(img[:,:,0])
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)


        source = img.copy()
        
        if np.random.rand() < 0.5:
            source = self.get_torch_transforms(source)
        else:
            img = self.get_torch_transforms(img)
        if isinstance(source, tuple):
            source, transforms_dict = source
        if  isinstance(img, tuple):
            img, transforms_dict = img 
        
        """
        if np.random.rand()<0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']
        """
        source, mask = self.randaffine(source,mask)
       
        img_blended,mask=self.dynamic_blend(source,img,mask)
        #img_blended,mask=self.static_blend(source,img,mask)
      
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)
        if self.config.use_regression:
            return img, img_blended, mask, transforms_dict
        else:
            return img, img_blended, mask
   
    def static_blend(self, source, target, mask):
        img_blended=(mask * source + (1 - mask) * target)
        return img_blended,mask

    def dynamic_blend(self, source,target,mask): 
        mask_blured = self.get_blend_mask(mask)  
        blend_list=[0.25,0.5,0.75,1,1,1]
        blend_ratio = blend_list[np.random.randint(len(blend_list))]
        mask_blured*=blend_ratio
        img_blended=(mask_blured * source + (1 - mask_blured) * target)
        return img_blended,mask_blured

    def get_blend_mask(self, mask):
        H,W=mask.shape
        size_h=np.random.randint(192,257)
        size_w=np.random.randint(192,257)
        mask=cv2.resize(mask,(size_w,size_h))
        kernel_1=random.randrange(5,26,2)
        kernel_1=(kernel_1,kernel_1)
        kernel_2=random.randrange(5,26,2)
        kernel_2=(kernel_2,kernel_2)

        mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
        mask_blured = mask_blured/(mask_blured.max())
        mask_blured[mask_blured<1]=0

        mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
        mask_blured = mask_blured/(mask_blured.max())
        mask_blured = cv2.resize(mask_blured,(W,H))
        return mask_blured.reshape((mask_blured.shape+(1,)))

    def reorder_landmark(self,landmark):
        landmark_add=np.zeros((13,2))
        for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
            landmark_add[idx]=landmark[idx_l]
        landmark[68:]=landmark_add
        return landmark

    def hflip(self,img,mask=None,landmark=None,bbox=None):
        H,W=img.shape[:2]
        landmark=landmark.copy()
        bbox=bbox.copy()

        if landmark is not None:
            landmark_new=np.zeros_like(landmark)

            
            landmark_new[:17]=landmark[:17][::-1]
            landmark_new[17:27]=landmark[17:27][::-1]

            landmark_new[27:31]=landmark[27:31]
            landmark_new[31:36]=landmark[31:36][::-1]

            landmark_new[36:40]=landmark[42:46][::-1]
            landmark_new[40:42]=landmark[46:48][::-1]

            landmark_new[42:46]=landmark[36:40][::-1]
            landmark_new[46:48]=landmark[40:42][::-1]

            landmark_new[48:55]=landmark[48:55][::-1]
            landmark_new[55:60]=landmark[55:60][::-1]

            landmark_new[60:65]=landmark[60:65][::-1]
            landmark_new[65:68]=landmark[65:68][::-1]
            if len(landmark)==68:
                pass
            elif len(landmark)==81:
                landmark_new[68:81]=landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:,0]=W-landmark_new[:,0]
            
        else:
            landmark_new=None

        if bbox is not None:
            bbox_new=np.zeros_like(bbox)
            bbox_new[0,0]=bbox[1,0]
            bbox_new[1,0]=bbox[0,0]
            bbox_new[:,0]=W-bbox_new[:,0]
            bbox_new[:,1]=bbox[:,1].copy()
            if len(bbox)>2:
                bbox_new[2,0]=W-bbox[3,0]
                bbox_new[2,1]=bbox[3,1]
                bbox_new[3,0]=W-bbox[2,0]
                bbox_new[3,1]=bbox[2,1]
                bbox_new[4,0]=W-bbox[4,0]
                bbox_new[4,1]=bbox[4,1]
                bbox_new[5,0]=W-bbox[6,0]
                bbox_new[5,1]=bbox[6,1]
                bbox_new[6,0]=W-bbox[5,0]
                bbox_new[6,1]=bbox[5,1]
        else:
            bbox_new=None

        if mask is not None:
            mask=mask[:,::-1]
        else:
            mask=None
        img=img[:,::-1].copy()
        return img,mask,landmark_new,bbox_new

    def collate_fn(self,batch):
        if self.config.use_regression:
            img_f, img_r, detect_face, label_f, label_r = zip(*batch)
        else:
            img_f, img_r, detect_face = zip(*batch)
        data={}
     
      
        data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
        
        data['labels']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
        data['detect_face']=torch.tensor(detect_face).repeat([2]).long()
        if self.config.use_regression:
            data['pseudo_labels'] = torch.cat([torch.tensor(label_r), 
                                            torch.tensor(label_f)], 0)
            
        return data
    
    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id) 
    
    def init_ff(self, dataset_path, phase, level="frame", n_frames=8):
        image_list=[]
        label_list=[]
        folder_list = sorted(glob.glob(dataset_path+'*')) 
        if level =='video':
            label_list=[0]*len(folder_list)
            return folder_list,label_list
        for i in range(len(folder_list)):
            images_temp=sorted(glob.glob(folder_list[i]+'/*.png'))
            if n_frames<len(images_temp):
                images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
            image_list+=images_temp
            label_list+=[0]*len(images_temp)
        
        return image_list,label_list
    
    def IoUfrom2bboxes(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
    
    def crop_face(self, img,landmark=None,bbox=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
        assert phase in ['train','val','test']

        #crop face------------------------------------------
        H,W=len(img),len(img[0])

        assert landmark is not None or bbox is not None

        H,W=len(img),len(img[0])

        if crop_by_bbox:
            x0,y0=bbox[0]
            x1,y1=bbox[1]
            w=x1-x0
            h=y1-y0
            w0_margin=w/4#0#np.random.rand()*(w/8)
            w1_margin=w/4
            h0_margin=h/4#0#np.random.rand()*(h/5)
            h1_margin=h/4
        else:
            x0,y0=landmark[:68,0].min(),landmark[:68,1].min()
            x1,y1=landmark[:68,0].max(),landmark[:68,1].max()
            w=x1-x0
            h=y1-y0
            w0_margin=w/8#0#np.random.rand()*(w/8)
            w1_margin=w/8
            h0_margin=h/2#0#np.random.rand()*(h/5)
            h1_margin=h/5



        if margin:
            w0_margin*=4
            w1_margin*=4
            h0_margin*=2
            h1_margin*=2
        elif phase=='train':
            w0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
            w1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
            h0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
            h1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()	
        else:
            w0_margin*=0.5
            w1_margin*=0.5
            h0_margin*=0.5
            h1_margin*=0.5
                
        y0_new=max(0,int(y0-h0_margin))
        y1_new=min(H,int(y1+h1_margin)+1)
        x0_new=max(0,int(x0-w0_margin))
        x1_new=min(W,int(x1+w1_margin)+1)

        img_cropped=img[y0_new:y1_new,x0_new:x1_new]
        if landmark is not None:
            landmark_cropped=np.zeros_like(landmark)
            for i,(p,q) in enumerate(landmark):
                landmark_cropped[i]=[p-x0_new,q-y0_new]
        else:
            landmark_cropped=None
        if bbox is not None:
            bbox_cropped=np.zeros_like(bbox)
            for i,(p,q) in enumerate(bbox):
                bbox_cropped[i]=[p-x0_new,q-y0_new]
        else:
            bbox_cropped=None

        if only_img:
            return img_cropped
        if abs_coord:
            return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1),y0_new,y1_new,x0_new,x1_new
        else:
            return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1)

