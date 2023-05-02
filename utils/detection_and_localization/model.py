import sys

import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append("./utils/detection_and_localization")
import network


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.add_FER_args(cfg)
        self.add_autoencoder_args(cfg)
        cfg = cfg.parse_args()
        """
    self.FER = fer.Ensemble(cfg.num_branches_trained_network, 
                    cfg.num_expressions)
    self.autoencoder = network.modeling.__dict__[cfg.model]\
            (num_classes=cfg.num_classes, output_stride=cfg.output_stride)
    """
        self.FER = models.resnet18(pretrained=True)
        self.FER.fc = nn.Linear(512, cfg.num_expressions)

    def forward2(self, img, fer_target, seg_target):
        out = self.FER(img)

        # all_prob [batch, branch, class]
        all_prob = torch.stack([out[i][0] for i in range(len(out))], 1)
        all_latent = torch.stack([out[i][1] for i in range(len(out))], 1)

        # sel_prob [batch, branch]
        sel_prob = all_prob[:, :, fer_target.squeeze(-1)][:, :, 0]

        # max_prob [batch], max_latent[ batch, channel, 1, 1]
        max_prob = torch.argmax(sel_prob, dim=1)
        max_latent = all_latent[:, max_prob][:, 0]

        seg, manipulate_prob = self.autoencoder(img, max_latent)

        return all_prob, seg, manipulate_prob

    def forward(self, img):
        out = self.FER(img)

        # all_prob [batch, branch, class]
        # all_prob = torch.stack([out[i][0] for i in range(len(out))], 1)

        return out

    def add_FER_args(self, parser):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        # Model parameters
        parser.add_argument("--lr", default=1e-3, type=float)
        parser.add_argument("--devices", default="cuda", type=str)
        parser.add_argument("--num_branches_trained_network", default=1,
                            type=int)
        parser.add_argument("--num_epochs", default=100, type=int)
        # parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--num_expressions", default=8, type=int)
        parser.add_argument(
            "--affectnet_dir",
            type=str,
            default="/affectnet",
            help="Path to affectnet folder",
        )

    def add_autoencoder_args(self, parser):
        parser.add_argument(
            "--num_classes", type=int, default=2,
            help="num classes (default: None)"
        )
        # Deeplab Options
        available_models = sorted(
            name
            for name in network.modeling.__dict__
            if name.islower()
            and not (name.startswith("__") or name.startswith("_"))
            and callable(network.modeling.__dict__[name])
        )
        parser.add_argument(
            "--model",
            type=str,
            default="deeplabv3plus_mobilenet",
            choices=available_models,
            help="model name",
        )
        parser.add_argument(
            "--separable_conv",
            action="store_true",
            default=False,
            help="apply separable conv to decoder and aspp",
        )
        parser.add_argument("--output_stride", type=int, default=16,
                            choices=[8, 16])

        # Train Options
        parser.add_argument("--test_only", action="store_true", default=False)
        parser.add_argument(
            "--save_val_results",
            action="store_true",
            default=False,
            help='save segmentation results to "./results"',
        )
        parser.add_argument(
            "--total_itrs", type=int, default=30e3,
            help="epoch number (default: 30k)"
        )

        parser.add_argument(
            "--lr_policy",
            type=str,
            default="poly",
            choices=["poly", "step"],
            help="learning rate scheduler policy",
        )
        parser.add_argument("--step_size", type=int, default=10000)
        parser.add_argument(
            "--crop_val",
            action="store_true",
            default=False,
            help="crop validation (default: False)",
        )
        parser.add_argument(
            "--val_batch_size",
            type=int,
            default=4,
            help="batch size for validation (default: 4)",
        )
        parser.add_argument("--crop_size", type=int, default=513)

        parser.add_argument(
            "--ckpt", default=None, type=str, help="restore from checkpoint"
        )
        parser.add_argument("--continue_training", action="store_true",
                            default=False)

        parser.add_argument(
            "--loss_type",
            type=str,
            default="cross_entropy",
            choices=["cross_entropy", "focal_loss"],
            help="loss type (default: False)",
        )
        parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-4,
            help="weight decay (default: 1e-4)",
        )
        parser.add_argument(
            "--random_seed", type=int, default=1,
            help="random seed (default: 1)"
        )
        parser.add_argument(
            "--print_interval",
            type=int,
            default=10,
            help="print interval of loss (default: 10)",
        )
        parser.add_argument(
            "--val_interval",
            type=int,
            default=100,
            help="epoch interval for eval (default: 100)",
        )
        parser.add_argument(
            "--download", action="store_true", default=False,
            help="download datasets"
        )
