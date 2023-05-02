# standard libraries

import tabulate
import numpy as np
# external libraires
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm 
# internal libraries
from utils.loss_utils import cross_entropy
import lpips
#torch.autograd.set_detect_anomaly(True)
columns = ["ep", "expression_loss", "manipulate_loss", "seg_loss",
           "total loss"]
def deactivate_batchnorm(model):
    if isinstance(model, nn.BatchNorm2d):
        model.reset_parameters()
        model.eval()
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.zero_()


        
def train(
        model: Module,
        trainset: DataLoader,
        optimizer: Optimizer,
        config,
        pipeline_obj,
) -> None:
    """
    The core of the training loop.

    Trains the given model for a single epoch on the given dataset using the loss criterion and optimizer.

    :param model: Module
        Model to be trained.
    :param train_iter: DataLoader
        The training set.
    :param criterion: Module
        The loss function.
    :param optimizer: Optimizer
        The optimizer.
    """
    model.train()
   
    columns = ["Step", "train loss", "train acc"]
    val_columns = ["Step", "Train Loss", "Train Acc",
                   "Val Loss", "Val Acc", "Val F1",
                   "Test Loss", "Test Acc", "Test F1"]
    if config.use_regression:
        val_columns = ["Step", "Train Loss", "Train Acc", "regression",
                   "Val Loss", "Val Acc", "Val F1",
                   "Test Loss", "Test Acc", "Test F1"]
    correct, num_trainset = 0.0, 0.0
    criterion=nn.CrossEntropyLoss()
    # TODO Add a tqdm or some progress bar
    # Can't add tqdm with table. The table looks weird.
    softmax = nn.Softmax(dim=-1)
    step = 0
    for epoch in range(1, config.num_epochs + 1):
        
        for _, features in enumerate(trainset):
            
            optimizer.zero_grad()
            img = features["img"].to(config.devices)
          
            labels = features["labels"].to(config.devices)
            detect_face = features['detect_face'].to(config.devices)

            detect_face_idx = torch.where(detect_face == 1)[0]
           
            img = img[detect_face_idx]
            labels = labels[detect_face_idx]
           
            if 'segmentation' in features.keys() and config.concat_segment:
                segment_map = features['segmentation'].to(config.devices)
                segment_map = segment_map[detect_face_idx]
                
                #img = torch.cat([img, segment_map], 1)
                img = img * segment_map
            # expression_prob [batch, branch, class]
            #model.apply(deactivate_batchnorm)
            logits = model(img)
            
            loss = 0
            
            if config.use_regression:
                if config.pseudo_labels:
                    pseudo_labels = features["pseudo_labels"].to(config.devices)[:, None]
                    logits, score = logits
                
                    out = torch.sigmoid(score)
                
                    #out = 1 - torch.exp(-torch.abs(score))

                    regress_loss = torch.mean((out - pseudo_labels)**2)
                    
                    loss = loss + regress_loss
                elif config.lpips:
                    real_idx = torch.where(labels == 0)[0]
                    fake_idx = torch.where(labels == 1)[0]
                  
                    with torch.no_grad():
                        lpips_score = 3*pipeline_obj.lpips_loss(img[real_idx], 
                                        img[fake_idx], normalize=True)
                       
                        lpips_score = torch.cat([torch.zeros_like(lpips_score),
                                            lpips_score], 0).squeeze()[:, None]
                     
                        """
                        def save(sel, path):
                            real = img[real_idx]
                            fake = img[fake_idx]
                            real = real[sel].cpu().detach().permute([1, 2, 0]).numpy()
                            fake = fake[sel].cpu().detach().permute([1, 2, 0]).numpy()
                            real = np.concatenate([real, fake], 0)
                            plt.imsave(f'test_{path}.jpg', real)
                        for i in range(10):
                            save(i, i)
                        exit()
                        """
                        
                        
                        
                    """
                    print('fc_regr')
                    print(torch.std(pipeline_obj.model.backbone._fc_regression.weight))
                    print('fc')
                    print(torch.std(pipeline_obj.model.backbone._fc.weight))
                    print('\n')
                    """
                    logits, score = logits

                    #logits = score
                    #score = logits
                    out = torch.sigmoid(score)
                   
                    #out = softmax(score)
                    #print(out.squeeze())
                    #print(out.squeeze())
                    #regress_loss = torch.mean(torch.abs(out - lpips_score))
                    #one_hot = F.one_hot(labels.long(), num_classes=2)
                   
                    regress_loss = torch.mean(torch.abs(out - lpips_score))
                    
                    #regress_loss = torch.mean(torch.abs(out - lpips_score))
                    
                    #regress_loss = criterion(score, labels.long())
                    loss = loss + regress_loss


                
            classification_loss = criterion(logits, labels.long())
            
            loss = loss + classification_loss
          
            loss.backward()
            optimizer.step()
          
            pred = logits.data.max(-1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
            temp_correct = pred.eq(labels.data.view_as(pred)).sum().item()
            num_trainset += img.shape[0]

            
            temp_acc = temp_correct / img.shape[0] * 100
            loss_dict = {"Total loss": loss}
            if config.use_regression:
                loss_dict["Regression loss"] = regress_loss
            if step % config.val_every == 0:
                if config.no_val:
                    values = [step, loss.item(), temp_acc]
                    if config.use_regression:
                        values = [step, loss.item(), temp_acc, regress_loss]
                    table = tabulate.tabulate([values], val_columns,
                                            tablefmt="simple",
                                            floatfmt="8.4f")
                    if step % config.print_every == 0:
                        if step % config.print_header_every == 0:
                            table = table.split("\n")
                            table = "\n".join([table[1]] + table)
                        else:
                            table = table.split("\n")[2]

                        print(table, flush=True)
                
                else:
                    vis_info = None
                    
                    with torch.no_grad():
                        val_info = pipeline_obj.test(pipeline_obj.val,
                                                    visualization=False,rand=True)
                        test_info = pipeline_obj.test(pipeline_obj.testset,
                                                    visualization=False)
                        values_tmp = [step, loss.item(), temp_acc,
                                    val_info["loss"].item(),
                                    val_info["accuracy"].item(),
                                    val_info["f1"].item(),
                                    test_info["loss"].item(),
                                    test_info["accuracy"].item(),
                                    test_info["f1"].item(), ]
                        if config.use_regression:
                            values_tmp = values_tmp[:4] + [regress_loss] +\
                                        values_tmp[4:]
                       
                            
                        val_table = tabulate.tabulate([values_tmp],
                                                    val_columns,
                                                    tablefmt="simple",
                                                    floatfmt="8.4f")
                        val_table = val_table.split("\n")
                        val_table = "\n".join([val_table[1]] + val_table)
                        print(val_table, flush=True)
                    pipeline_obj.log_tensorboard(loss_dict, temp_acc, step,
                                                vis_info,
                                                val_info, test_info)
                  
                        
                if pipeline_obj.config.visualized:
                    vis_info = pipeline_obj.visualize(img, labels)
                    pipeline_obj.log_tensorboard(loss_dict, temp_acc, step,
                                                vis_info)
            
                
                #if test_info["accuracy"].item() > 80:
                if step % config.save_every == 0 and step != 0:
                    pipeline_obj.checkpoint(step)
                
            step += 1
       
        pipeline_obj.scheduler.step()
        #pipeline_obj.early_stop(val_info["loss"], pipeline_obj.checkpoint, step)