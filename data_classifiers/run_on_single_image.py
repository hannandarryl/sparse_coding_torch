import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from utils.load_data import load_bamc_data, load_bamc_clips
from feature_extraction.conv_sparse_model import ConvSparseLayer
from data_classifiers.small_data_classifier import SmallDataClassifier
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import random
from data_classifiers.pca import PCASparseActivations
import torchvision
from utils.video_loader import VideoGrayScaler, MinMaxScaler
from torchvision.datasets.video_utils import VideoClips

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', default=None, type=str)
    parser.add_argument('--kernel_height', default=15, type=int)
    parser.add_argument('--kernel_width', default=15, type=int)
    parser.add_argument('--kernel_depth', default=5, type=int)
    parser.add_argument('--num_kernels', default=64, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--max_activation_iter', default=200, type=int)
    parser.add_argument('--activation_lr', default=1e-1, type=float)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--sparse_checkpoint', default=None, type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    
    assert args.input_video
    
    if not os.path.exists(args.input_video):
        raise Exception('Please provide an input image.')
    
    frozen_sparse = ConvSparseLayer(in_channels=1,
                                   out_channels=args.num_kernels,
                                   kernel_size=(args.kernel_depth, args.kernel_height, args.kernel_width),
                                   stride=args.stride,
                                   padding=(0, 7, 7),
                                   convo_dim=3,
                                   rectifier=True,
                                   lam=args.lam,
                                   max_activation_iter=args.max_activation_iter,
                                   activation_lr=args.activation_lr)
    if args.sparse_checkpoint:
        sparse_param = torch.load(args.sparse_checkpoint, map_location=device)
        frozen_sparse.load_state_dict(sparse_param['model_state_dict'])
    frozen_sparse.to(device)

    predictive_model = torch.nn.DataParallel(SmallDataClassifier())
    predictive_model.to(device)
        
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        predictive_model.load_state_dict(checkpoint['model_state_dict'])
        
    transform = torchvision.transforms.Compose(
    [VideoGrayScaler(),
     MinMaxScaler(0, 255),
     torchvision.transforms.Normalize((0.2592,), (0.1251,)),
     torchvision.transforms.CenterCrop((100, 200))
    ])
    
    vc = VideoClips([args.input_video],
                        clip_length_in_frames=5,
                        frame_rate=20,
                       frames_between_clips=1)
    
    clip_predictions = []
    for i in tqdm(range(vc.num_clips())):
        clip, _, _, _ = vc.get_clip(i)
        clip = clip.swapaxes(1, 3).swapaxes(0, 1).swapaxes(2, 3).to(torch.float)
        clip = transform(clip)

        with torch.no_grad():
            clip = clip.unsqueeze(0).to(device)
            u_init = torch.zeros([1, frozen_sparse.out_channels] + frozen_sparse.get_output_shape(clip)).to(device)
            activations, _ = frozen_sparse(clip, u_init)

            # Note that you can get activations here
            pred, activations = predictive_model(activations)
            
            clip_predictions.append(torch.nn.Sigmoid()(pred).round().detach().cpu().flatten().to(torch.long))

                   
    final_pred = torch.mode(torch.tensor(clip_predictions))[0].item()
    if final_pred == 1:
        print('No Sliding')
    else:
        print('Sliding')