import os
import sys
import numpy as np 
import torch
import torch.nn as nn
import torchvision.models as models
sys.path.append("./utils")
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.efficient_utils import get_same_padding_conv2d
from loguru import logger
import utils 


def increase_momentum(model):
    if isinstance(model, nn.BatchNorm2d):
        model.momentum = 0.1

class Detector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.add_args(config)
        config = config.parse_args()
        """
        self.backbone = EfficientNet.from_pretrained("efficientnet-b4",
                            advprop=True,
                            num_classes=config.num_classes + 1 if config.use_regression else config.num_classes,
                            in_channels=3)
        """
        
        num_classes = [config.num_classes, 1] if config.use_regression else config.num_classes
        self.backbone = EfficientNet.from_pretrained("efficientnet-b4",
                            advprop=True,
                            num_classes=num_classes,
                            in_channels=3)
        self.backbone.apply(increase_momentum)
        """
       
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, config.num_classes)
        """
        """
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, config.num_classes)
        
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, config.num_classes)
        
        self.backbone = models.vgg16(pretrained=True)
        self.backbone.classifier = nn.Sequential(
                nn.Linear(25088, 256),
                nn.ReLU(),
                nn.Linear(256, config.num_classes)
        )
        """
      
        if os.path.exists(config.path2pretrained):
            logger.info(f"Load pretrained from {config.path2pretrained}")
            cnn_sd = torch.load(config.path2pretrained)["model"]

            cnn_sd = {key.replace("net.", ""): value for key, value in
                      cnn_sd.items()}

            #self.backbone.load_state_dict(cnn_sd, strict=True)

        else:
            logger.info("No pretrained")
        
        
        """
        if config.concat_segment:
            Conv2d = get_same_padding_conv2d(
                image_size=self.backbone._global_params.image_size)
            self.backbone._conv_stem = Conv2d(3,
                                              self.backbone._conv_stem.out_channels,
                                              kernel_size=3, stride=2,
                                              bias=False)
        """
    def forward(self, img):
        logits = self.backbone(img)
        if isinstance(logits, tuple):
            logits, score = logits
            return logits, score
       
        return logits

    def add_args(self, parser):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        # Model parameters

        parser.add_argument("--devices", default="cuda", type=str)
#         parser.add_argument("--batch_size", default=1, type=int)
        


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, checkpoint_fn, step):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            checkpoint_fn(step)
          
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                exit()
        else:
            self.best_score = score
            checkpoint_fn(step)
            self.counter = 0

