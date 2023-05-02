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
import glob 
from torchvision import transforms
warnings.filterwarnings("ignore")
import dataset
import model as M
import matplotlib.pyplot as plt 
from utils import get_visualizer, heat_map, EarlyStopping
from loguru import logger
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

parser = argparse.ArgumentParser()
parser.add_argument('--real_path', type=str, default="/vast/jy3694/captcha_dataset")
parser.add_argument('--dataset_path', type=str, default="/vast/jy3694/all_dataset3")
parser.add_argument('--plot_graph', type=bool, default=True)
cmd_args_utils.add_common_flags(parser)

def load_model():
    model = M.get_network(parser)
    args = parser.parse_args()
    model_path = os.path.join("runs", args.model_dir)
    ckpt = os.path.join(model_path, "ckpt.pt")
    
    if os.path.exists(ckpt):
        _ = utils.loadFromCheckpoint(ckpt, model,
                        None, args)
    retina = RetinaGetFace(image_size=512, 
                            device=args.devices)
    return model, args, retina 

def load_data(path, args, retina):
    if not os.path.exists(path): return None 
    testdata = dataset.CaptchaDataset(
                        image_path=path,
                        image_size=args.img_shape,
                        use_keypoint=args.use_keypoint,
                        use_segment=args.use_segment,
                        use_retinafacenet=True,
                        use_facenet=args.use_facenet,
                        use_self_blended=False,
                        use_cutout=False,
                        device=args.devices,
                        config=args,
                        num_dataset=args.num_dataset,
                        data_type="test",
                        sel_every=2, 
                        verbose=False,
                        retinaface = retina,
                        )
    testset = DataLoader(testdata,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers)
    return testset

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

@logger.catch 
def main():
    model, args, retina = load_model()
    
   
    model.eval()
    
    score = {}
    eval_chal = {"Real's Fakeness score": [],
                "Fake's Fakeness score": [],
                "Accuracy": [],
                }
    with torch.no_grad():
        for i_chal, impersonator_path in enumerate(sorted(glob.glob(os.path.join(args.dataset_path, "*")))):
            if '.csv' in impersonator_path or '.jpg' in impersonator_path: continue
            impersonator_graph = []
            all_score = {"Real's Fakeness score": [],
                        "Fake's Fakeness score": [],
                        "Accuracy": [],
                        }
            for challenge_path in sorted(glob.glob(os.path.join(impersonator_path, "*"))):
                
                chal_name = challenge_path.split('/')[-1]
                
                if '.csv' in chal_name or '.jpg' in chal_name: continue
                score[chal_name] ={}
                avg_score = {"Real's Fakeness score": [],
                            "Fake's Fakeness score": [],
                            "Accuracy": [],
                            }
                score_graph = []
                fake_graph, real_graph = [], []
                for person in sorted(glob.glob(os.path.join(challenge_path, "*"))):
            
                    name = person.split('/')[-1]
                    
                    if '.csv' in name or'.jpg' in name: continue
                    score[chal_name][name] = {}
                    real_count, fake_count, num_data = 0, 0, 0
                    print(f"impersonator: {i_chal}, chal_name: {chal_name} person: {name}")
                    all_path = [p for p in sorted(glob.glob(os.path.join(person, "*"))) \
                                if not (p.endswith("csv") or p.endswith("jpg"))]
                    if len(all_path) == 1:
                        
                        real_path = os.path.join(args.real_path, 
                                        chal_name,
                                        name,
                                        "real")
                        all_path.append(real_path)
                   
                    for type_path in all_path:
                    
                        type_name = type_path.split('/')[-1]
                        score[chal_name][name][type_name] = []
                        testset = load_data(type_path, args, retina)
                        if testset is None: continue 
                    
                        for _, features in enumerate(testset):
                            img = features["img"].to(args.devices)

                            labels = features["labels"].to(args.devices)
                            detect_face = features['detect_face'].to(args.devices)
                            detect_face_idx = torch.where(detect_face == 1)[0]
                        
                            img = img[detect_face_idx]
                            labels = labels[detect_face_idx]
                        
                            logits = model(img)
                            #pred = model(img).softmax(1)[:,1]
                            if args.use_regression:
                                if args.pseudo_labels:
                                    logits, out_score = logits
                                    out_score = torch.sigmoid(out_score)
                                
                                elif args.lpips:
                                    logits, out_score = logits
                                    out_score = torch.sigmoid(out_score)
                                
                        
                            pred = logits.softmax(1)[:,1]
                        
                            if 'real' in type_path:
                                real_count += torch.sum((pred < 0.5).float())
                            else:
                                fake_count += torch.sum((pred > 0.5).float())
                        
                            if args.use_regression:
                                pred = out_score.cpu().data.numpy().tolist()
                            else:
                                pred = pred.cpu().data.numpy().tolist()

                            score[chal_name][name][type_name] += pred
                            
                            num_data += img.shape[0]
                        
                    acc = (real_count + fake_count) / num_data
                    acc = np.array(acc.item()).tolist()              
                    if 'real' in score[chal_name][name].keys():
                        if len(score[chal_name][name]['real'])  == 0:
                            real_score = np.array([-1]).mean().tolist()
                        else:
                            real_score = np.array(score[chal_name][name]['real']).mean().tolist()
                    fake_score = np.array(score[chal_name][name]['fake']).mean().tolist()
                    info_dict = {
                                "Fake's Fakeness score": [fake_score],
                                "Accuracy": [acc],
                                }
                    if 'real' in score[chal_name][name].keys():
                        info_dict["Real's Fakeness score"] = [real_score]
                        
                    
                    df = pd.DataFrame.from_dict(info_dict)
                    df.to_csv(os.path.join(person, 'anomaly_test.csv'))
                    for key, val in info_dict.items():
                        if val[0] != -1.0:
                            avg_score[key] += val
                    if args.plot_graph:
                        def spline(x, y, new_x, k, s, der):
                            tck = interpolate.splrep(x, y, k=3, s=2)
                            return interpolate.splev(new_x, tck, der=0)
                        """
                        fake_time = np.arange(len(score[chal_name][name]['fake']))
                        fake_scores = np.array(score[chal_name][name]['fake'])
                        real_time = np.arange(len(score[chal_name][name]['real']))
                        real_scores = np.array(score[chal_name][name]['real'])
                    
                        new_time = np.arange(min(len(score[chal_name][name]['fake']),
                                            len(score[chal_name][name]['real'])))
                        fake_scores = fake_scores[:new_time.shape[0]]
                        real_scores = real_scores[:new_time.shape[0]]
                        
                        fake_scores = spline(fake_time, fake_scores, new_time,
                                            k=5, s=0, der= 3)
                        real_scores = spline(real_time, real_scores, new_time,
                                            k=3, s=2, der= 3)
                        """
                        fake_scores = np.array(score[chal_name][name]['fake'])
                        new_time = np.arange(len(score[chal_name][name]['fake']))
                   
                        plt.plot(new_time, fake_scores, 'r', linestyle = 'dashed')
                        #plt.plot(new_time, real_scores, 'g')
                        
                        plt.xlabel("Frame")
                        plt.ylabel("Fake's Fakeness score")
                        #plt.legend(['fake', 'real'])
                        plt.legend(['fake'])
                        plt.title(f"Person: {name}, Challenge: {chal_name}")
                        plt.savefig(os.path.join(person, 'graph.jpg'))
                        plt.clf()
                        fake_graph.append(fake_scores)
                        """
                        fake = fake_scores- real_scores
                        score_graph.append(fake)
                        
                        real_graph.append(real_scores)
                        plt.plot(new_time, fake, 'b')
                        plt.xlabel("Frame")
                        plt.ylabel("Fake's Fakeness score")
                        #plt.legend(['fake', 'real'])
                        plt.legend(['diff'])
                        plt.title(f"Person: {name}, Challenge: {chal_name}")
                        plt.savefig(os.path.join(person, 'graph_diff.jpg'))
                        plt.clf()
                        """
                zero_index = [i for i in range(len(fake_graph)) if len(fake_graph[i]) == 0]
                
                for idx in zero_index:
                    del fake_graph[idx]
                min_seq = min(np.array([len(seq) for seq in fake_graph]))
                """
                score_graph = np.mean(np.array([seq[:min_seq] for seq in score_graph]), axis=0)
                real_graph = np.mean(np.array([seq[:min_seq] for seq in real_graph]), axis=0)
                """
                fake_graph = np.mean(np.array([seq[:min_seq] for seq in fake_graph]), axis=0)
                impersonator_graph.append(fake_graph)
                """
                plt.plot(np.arange(len(score_graph)), score_graph, 'b')
                impersonator_graph.append(score_graph)

                plt.xlabel("Frame")
                plt.ylabel(f"Average Difference of fake and real images")
            
                plt.title(f"Challenge: {chal_name}")
                plt.savefig(os.path.join(challenge_path, 'avg_diff.jpg'))
                plt.clf()
                """
                #plt.plot(np.arange(len(real_graph)), real_graph, 'g')
                plt.plot(np.arange(len(fake_graph)), fake_graph, 'r', linestyle = 'dotted')
            
                plt.xlabel("Frame")
                plt.ylabel(f"Fake score")
                plt.legend(['fake'])
                #plt.legend(['real', 'fake'])
                plt.title(f"Challenge: {chal_name}")
                plt.savefig(os.path.join(challenge_path, 'avg.jpg'))
                plt.clf()
                
                for key, val in avg_score.items():
                    avg_score[key] = [np.array(val).mean()]
                
                df = pd.DataFrame.from_dict(avg_score)
                df.to_csv(os.path.join(challenge_path, 'avg_anomaly_test.csv'))
                for key, val in avg_score.items():
                    all_score[key] += val
            df = pd.DataFrame.from_dict(all_score)
           
            df.to_csv(os.path.join(impersonator_path, 'all_anomaly_test.csv'))
            min_seq = min(np.array([len(seq) for seq in impersonator_graph]))
            """
            score_graph = np.mean(np.array([seq[:min_seq] for seq in score_graph]), axis=0)
            real_graph = np.mean(np.array([seq[:min_seq] for seq in real_graph]), axis=0)
            """
            impersonator_graph = np.mean(np.array([seq[:min_seq] for seq in impersonator_graph]), axis=0)

            plt.plot(np.arange(len(impersonator_graph)), impersonator_graph, 'r', linestyle = 'dotted')
            #plt.plot(new_time, real_scores, 'g')
            
            plt.xlabel("Frame")
            plt.ylabel("Fake's Fakeness score")
            #plt.legend(['fake', 'real'])
            plt.legend(['fake'])
            plt.title(f"Challenge: {chal_name}")
            plt.savefig(os.path.join(args.dataset_path, f'chal_{i_chal}.jpg'))
            plt.clf()
            for key, val in eval_chal.items():
                
                eval_chal[key] += [all_score[key]]
        #print(eval_chal)
        for key, val in eval_chal.items():
            eval_chal[key] = np.mean(np.array(val), axis = 0)
          
        df = pd.DataFrame.from_dict(eval_chal)
        df.to_csv(os.path.join(args.dataset_path, 'evaluation.csv'))
       
       


if __name__=='__main__':
    main()

