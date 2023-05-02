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
# external libraries
from torch.utils.tensorboard import SummaryWriter
import skimage.io as io 
import warnings
warnings.filterwarnings("ignore")
import dataset
import model
import lpips 
import matplotlib.pyplot as plt 
import lpips 
from typing import Dict 
from utils import get_visualizer, heat_map, EarlyStopping
# internal libraries
from utils import utils
from utils.loss_utils import cross_entropy
from tqdm import tqdm

class SourceExtractionPL(object):
    """
    The source extraction pipeline.

    A SourceExtractionPL object manages all pipeline components: data loading, log tensorboard
    
    To use, first create an object of this class. Then pass attributes to the trainer as below
    -------
    The first few steps of the pipeline are illustrated below.

    >>> pl = pipeline.SourceExtractionPL(parser)
    >>> trainer.train(pl.model, pl.train, pl.loss, pl.optimizer, pl.config, pl)
    """

    def __init__(self, config):
        """
        Initialize the pipeline attributes to be set later.
        """
        self.model = model.get_network(config)
        self.config = config.parse_args()

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                          lr=self.config.lr, momentum=0.9)
        #self.optimizer = torch.optim.Adam(self.model.parameters(),
        #                                  lr=self.config.lr)
    
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                step_size=int(5*self.config.num_epochs/4),
                                gamma=self.config.lr_decay)
        self.early_stop = EarlyStopping()

        self.model_path = os.path.join("runs", self.config.model_dir)
        if self.config.restart:
            os.system(f'rm -rf {self.model_path}')
        #self.model_path = "./runs/8_1/lpips"
        self.ckpt = os.path.join(self.model_path, "ckpt.pt")
        if os.path.exists(self.ckpt):
            start_epoch = utils.loadFromCheckpoint(self.ckpt, self.model,
                                                   self.optimizer, self.config)
        if "faceforensics" in self.config.dataset_dir:
            self.load_data_sbi()
        else:
            self.load_data_captcha()
        if self.config.visualized:
            self.visualize_model = copy.deepcopy(self.model.backbone)
            # print(self.visualize_model._modules)
            
            name = list(dict(self.visualize_model._modules.items()).keys())[2]
            #name = list(dict(self.visualize_model._modules.items()).keys())[7]
            num_block = list(dict(
                getattr(self.visualize_model, name)._modules.items()).keys())[
                -1]

            self.visualized_method = get_visualizer(
                vis_type=self.config.visualization_type.lower(),
                model=self.visualize_model,
                # target_layers=[self.visualize_model.layer3],
                target_layers=[
                    getattr(self.visualize_model, name)[int(num_block)]],
                    devices=self.config.devices)
        if self.config.lpips:
            self.lpips_loss = lpips.LPIPS(net='vgg').to(self.config.devices)
        if self.config.test:
            with torch.no_grad(): self.test(self.testset)
            exit()
        self.tensorboard()

    def load_data_captcha(self) -> None:
        """
        Load the data.

        :param train: DataLoader
            Training set.
        :param test: DataLoader
            Test set.
        """
        self.traindata = dataset.CaptchaDataset(
            image_path=self.config.dataset_dir,
            image_size=self.config.img_shape,
            num_dataset=self.config.num_dataset,
            use_keypoint=self.config.use_keypoint,
            use_segment=self.config.use_segment,
            use_facenet=self.config.use_facenet,
            use_self_blended=self.config.use_self_blended,
            use_cutout=self.config.use_cutout,
            use_retinafacenet=self.config.use_retinafacenet,
            device=self.config.devices,
            config=self.config,
            data_type="train")

        self.valdata = dataset.CaptchaDataset(
            image_path=self.config.dataset_dir,
            image_size=self.config.img_shape,
            num_dataset=self.config.num_dataset,
            use_keypoint=self.config.use_keypoint,
            use_segment=self.config.use_segment,
            use_facenet=self.config.use_facenet,
            use_self_blended=self.config.use_self_blended,
            use_cutout=self.config.use_cutout,
            use_retinafacenet=self.config.use_retinafacenet,
            device=self.config.devices,
            config=self.config,
            data_type="val")

        self.testdata = dataset.CaptchaDataset(
            image_path=self.config.testset_dir,
            image_size=self.config.img_shape,
            use_keypoint=self.config.use_keypoint,
            use_segment=self.config.use_segment,
            use_retinafacenet=True,
            use_facenet=self.config.use_facenet,
            use_self_blended=False,
            use_cutout=False,
            device=self.config.devices,
            config=self.config,
            num_dataset=self.config.num_dataset,
            data_type="test")
      
        
        self.train = DataLoader(self.traindata,
                                batch_size=self.config.batch_size,
                                shuffle=self.config.shuffle,
                                num_workers=self.config.num_workers)
        self.val = DataLoader(self.valdata,
                              batch_size=self.config.batch_size,
                              shuffle=True,
                              num_workers=self.config.num_workers)
        self.testset = DataLoader(self.testdata,
                                  batch_size=self.config.batch_size,
                                  shuffle=False,
                                  num_workers=self.config.num_workers)

    def load_data_sbi(self) -> None:
        """
        Load the data.

        :param train: DataLoader
            Training set.
        :param test: DataLoader
            Test set.
        """
        self.traindata = dataset.SBI_Dataset(
            image_path=self.config.dataset_dir,
            phase='train',
            image_size=self.config.img_shape,
            config=self.config)

        self.valdata = dataset.SBI_Dataset(
            image_path=self.config.dataset_dir,
            phase='val',
            image_size=self.config.img_shape,
            config=self.config)
        
        self.testdata = dataset.CaptchaDataset(
            image_path=self.config.testset_dir,
            image_size=self.config.img_shape,
            use_keypoint=self.config.use_keypoint,
            use_segment=self.config.use_segment,
            use_retinafacenet=True,
            use_facenet=self.config.use_facenet,
            use_self_blended=False,
            use_cutout=False,
            device=self.config.devices,
            config=self.config,
            num_dataset=self.config.num_dataset,
            data_type="test")
        
        self.train = DataLoader(self.traindata,
                        batch_size=self.config.batch_size//2,
                        shuffle=True,
                        collate_fn=self.traindata.collate_fn,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=True,
                        worker_init_fn=self.traindata.worker_init_fn
                        )
        self.val = DataLoader(self.valdata,
                        batch_size=8,
                        shuffle=True,
                        collate_fn=self.valdata.collate_fn,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=True,
                        worker_init_fn=self.traindata.worker_init_fn
                        )
       
        self.testset = DataLoader(self.testdata,
                                  batch_size=self.config.batch_size,
                                  shuffle=False,
                                  num_workers=self.config.num_workers)
        
    def tensorboard(self) -> SummaryWriter:
        """
        This method creates writer attributes to log training schedule 
        """
        self.writer = SummaryWriter(self.model_path)
        self.writer.add_text('command', ' '.join(sys.argv), 0)

    def log_tensorboard(self,
                        loss: Dict,
                        acc: torch.tensor,
                        epoch: int,
                        vis_info=None,
                        val_info=None,
                        test_info=None,
                        ) -> None:
        """
        This method logs loss, accuracy, and visualization to the tensorboard
        """
        for key, val in loss.items():
            self.writer.add_scalar(f'Train/{key}', val, epoch)
        self.writer.add_scalar('Train/Acc', acc, epoch)
        if val_info is not None:
            for key, val in val_info.items():
                self.writer.add_scalar(f"Val/{key}", val, epoch)
            for key, val in test_info.items():
                self.writer.add_scalar(f"test/{key}", val, epoch)
        if vis_info is not None:
            for i in range(vis_info[0].shape[0]):
                # name = self.data.expression[vis_info[1][i]]
                name = "real" if vis_info[1][i] == 0 else "fake"
                self.writer.add_images(f"Train/{name}/{i}",
                                       vis_info[0][i:i + 1],
                                       epoch)

    def visualize(self,
                  img: torch.tensor,
                  target: torch.tensor
                  ):
        """
        This function returns the heat map of the visualization of a trained neural network
        """
        return heat_map(self.config.visualization_type,
                        self.model.backbone, self.visualize_model,
                        self.visualized_method, img, target)

    def checkpoint(self, epoch):
        file = os.path.join(self.model_path, './ckpt.pt')
        utils.checkpoint(file, self.model, self.optimizer, epoch)

    def test(self, data, visualization=True, rand=False):
        if visualization:
            if self.config.test_visualization_dir == "e1": 
                self.config.test_visualization_dir = self.config.testset_dir + '_visualization'
            real_path = os.path.join(self.config.test_visualization_dir, 'real')
            fake_path = os.path.join(self.config.test_visualization_dir, 'fake')
            
            if not (os.path.exists(real_path) or os.path.exists(fake_path)):
                os.system(f"mkdir -p {real_path}")
                os.system(f"mkdir -p {fake_path}")
        names, scores = [], []

        loss, tp, fp, fn, tn, count, num_data = 0, 0, 0, 0, 0, 0, 0
       
        self.softmax = nn.Softmax(dim=-1)
        self.model.eval()
        criterion=nn.CrossEntropyLoss()

        for enum, features in enumerate(data):
            if rand:
                if enum == 5: break
            img = features["img"].to(self.config.devices)

            labels = features["labels"].to(self.config.devices)
            detect_face = features['detect_face'].to(self.config.devices)
            detect_face_idx = torch.where(detect_face == 1)[0]
          
            img = img[detect_face_idx]
            labels = labels[detect_face_idx]
           
            if (self.config.use_segment or
                self.config.use_deeplab) and self.config.concat_segment:
                segment_map = features['segmentation'].to(self.config.devices)
                segment_map = segment_map[detect_face_idx]

                #img = torch.cat([img, segment_map], 1)
                img = img *segment_map

            logits = self.model(img)
            
            if self.config.use_regression:
                logits, score = logits
                #score = logits
                if self.config.pseudo_labels:
                    pseudo_labels = features["pseudo_labels"].to(self.config.devices)[:, None]
                    pseudo_labels = pseudo_labels[detect_face_idx]
                    
                    out = torch.sigmoid(score)
                
                    loss = loss + torch.mean((out - pseudo_labels)**2)

                elif self.config.lpips and rand:
                    real_idx = torch.where(labels == 0)[0]
                    fake_idx = torch.where(labels == 1)[0]
                    with torch.no_grad():
                      
                        lpips_score = self.lpips_loss(img[real_idx], 
                                        img[fake_idx], normalize=True)
                        lpips_score = torch.cat([torch.zeros_like(lpips_score),
                                            lpips_score], 0)
                  
                    out = torch.sigmoid(score)

                    #regress_loss = torch.mean((out - lpips_score)**2)
                    #regress_loss = 0
                    #loss = loss + regress_loss
               
            loss += criterion(logits, labels.long())

            
            if visualization:
                names += ["/".join(n.split('/')[-2:]) for n in
                                        features["name"]]
           
            
            real_id = torch.where(labels == 0)[0]
            fake_id = torch.where(labels == 1)[0]
           
            out = self.softmax(logits)[:, 1]
            out_fake = out[fake_id]
            out_real = out[real_id]
        
            tp += torch.sum((out_real < 0.5).float())
            fp += torch.sum((out_fake > 0.5).float())
            fn += torch.sum((out_real > 0.5).float())
        
            num_data += img.shape[0]
           

            temp_score = np.zeros((features["img"].shape[0]))
            temp_score[
                detect_face_idx.detach().cpu().numpy()] = out.detach().cpu().numpy()
            no_face_idx = torch.where(detect_face == 0)[
                0].detach().cpu().numpy()
            temp_score[no_face_idx] = -1 * np.ones((no_face_idx.shape[0]))
            scores += list(temp_score)

            if visualization:
                vis_info = heat_map(self.config.visualization_type,
                        self.model.backbone, self.visualize_model,
                        self.visualized_method, img, labels, sel=-1)
                vis_map = vis_info[0].detach().cpu().permute([0, 2, 3, 1]).numpy()
                for i in range(vis_info[0].shape[0]):
                    path = real_path if vis_info[1][i] == 0 else fake_path
                    im = (vis_map[i] * 255).astype(np.uint8)
                    io.imsave(os.path.join(path,
                        features["name"][i].split('/')[-1]), im)
               
            
     
        pr = tp / (tp + fp)
        rc = tp / (tp + fn)
        f1 = 2 * (pr * rc) / (pr + rc + 1e-9)
        acc = (tp + fp) / num_data * 100
        self.model.train()
        if visualization:
            real_idx = [i for i, n in enumerate(names) if 'real' in n]
            fake_idx = [i for i, n in enumerate(names) if 'fake' in n]
            info_dict = {"Real Names": list(itemgetter(*real_idx)(names)),
                        "Real's Fakeness Score": list(
                            itemgetter(*real_idx)(scores)),
                        "Fake Names": list(itemgetter(*fake_idx)(names)),
                        "Fake's Fakeness Score": list(itemgetter(*fake_idx)(scores))}

            num = abs(len(info_dict['Real Names']) - len(info_dict['Fake Names']))
            if len(info_dict['Real Names']) > len(info_dict['Fake Names']):

                info_dict['Fake Names'] += ["Not found"]* num
                info_dict["Fake's Fakeness Score"] += [0]* num

            else:
                info_dict['Real Names'] += ["Not found"]* num
                info_dict["Real's Fakeness Score"] += [0]* num

            df = pd.DataFrame.from_dict(info_dict)
            print("===================================")
            print(df.to_string(index=False))
            print("-----------------------------------")
            print(df.mean().to_string())
            print(f"Accuracy: {round(acc.item(), 2)}")
            print(f"Precision: {round(pr.item(), 4)}")
            print(f"Recall: {round(rc.item(), 4)}")
            print(f"F1: {round(f1.item(), 4)}")
            # TODO performed an edit of visualization; bad hardcoding below; remove!

            df.to_csv('./anomaly_test.csv')
            

        else:
            return {"loss": loss / (enum+1),
                        "accuracy": acc,
                        "precision": pr,
                        "recall": rc,
                        "f1": f1}
