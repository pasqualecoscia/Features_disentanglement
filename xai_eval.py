import os
import time
import argparse
from datetime import datetime as dt
from random import randint
import urllib
import zipfile
import csv
import json
import glob
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, transforms as T, datasets
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler

from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.utils import make_grid

from utils.dataset_utils import load_tiny_dataset, MyDataset
from utils.model_utils import *
from utils.xai_utils import *
from utils.config_params import get_config_parser

from torchcam.utils import overlay_mask
from torch.nn.functional import softmax, interpolate

from utils.LR_finder import LRFinder, plot_lr_finder
from utils.logger import Logger

# Suppress the specific UserWarning about antialias parameter
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms")

def main():

    # Print the default configuration
    opt = get_config_parser()
    print(vars(opt))
    
    # Save path
    saved_model_path = f'{opt.model}_{opt.training_type}_{str(opt.num_classes)}_{str(opt.data_percentage)}_{str(opt.epochs)}.pt'
    results_path_folder = os.path.join(opt.results_path, saved_model_path.split('.')[0], opt.XAI_path, opt.cam_type, str(opt.segmentation_threshold))

    # Check if the folder exists, and create it if not
    if not os.path.exists(results_path_folder):
        os.makedirs(results_path_folder)

    # Create an instance of Logger
    logger = Logger(path=results_path_folder)

    if opt.dataset == 'tiny-image-net':
        print(f"Selected dataset: {opt.dataset}.")
        train_dataset, validation_dataset, test_dataset, _ = load_tiny_dataset(opt)
    else:
        raise NotImplementedError(opt.dataset)
    
    # Set options for device
    if opt.use_cuda:
        kwargs = {"pin_memory": opt.pin_memory, "num_workers": opt.num_workers}
    else:
        kwargs = {}
             
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, 
                        shuffle=True, 
                        **kwargs)
    
    validation_loader = DataLoader(validation_dataset, batch_size=opt.batch_size, 
                        shuffle=False, 
                        **kwargs)  
                        
    # Define the test_loader with batch size 1
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs) 

    if 'resnet' in opt.model:
        if opt.model == 'resnet18':
            model = models.resnet18()
        elif opt.model == 'resnet50':
            model = models.resnet50()
        else:
            model = models.resnet101()   
        IN_FEATURES = model.fc.in_features 
        OUTPUT_DIM = opt.num_classes
        model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    elif opt.model == 'vgg19':
        model = models.vgg19()
        IN_FEATURES = model.classifier[0].in_features 
        OUTPUT_DIM = opt.num_classes
        model.classifier = nn.Linear(IN_FEATURES, OUTPUT_DIM)        
    else:
        raise NotImplementedError(opt.model)

    model_path = os.path.join(opt.results_path, saved_model_path.split('.')[0], saved_model_path)
    # Remove encoders if dis_loss active
    if opt.dis_loss:
        # If disentangle loss is active re-define the model adding additional encoders
        model_ = copy.deepcopy(model)
        model_with_encoders = ModelWithEncoders(model_, opt)
        # Load weights
        model_with_encoders.load_state_dict(torch.load(model_path))
        # Copy weights
        copy_weights(model_with_encoders, model)
    else:
        model.load_state_dict(torch.load(model_path))
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device).eval()

    # XAI metrics
    jaccard_index_values = []
    precision_score_values = []
    recall_score_values = []
    accuracy_values = []
    dice_coef_values = []

    resize_tr = transforms.Resize(opt.image_size_dataset, interpolation=transforms.InterpolationMode.BICUBIC)
    
    # Initialize CAM extractor
    cam_extractor = extractCAMs(model=model, layers=opt.cam_layers, cam_type=opt.cam_type)

    for idx, (image, label, ann) in enumerate(test_loader):
        
        print(f'{str(idx)}/{len(test_loader)}')

        image = image.squeeze(0).to(device) # Image
        label = label.to(device) # Label
        
        # Model segmentation mask
        out = model(image.unsqueeze(0))
        preds_softmax = torch.nn.functional.softmax(out, dim=1)
        top_prob, top_pred = preds_softmax.max(dim=1)

        # Evaluate CAMs wrt label (GT) or top_pred
        if opt.label_toppred_evaluation == "label":
            item_to_evaluate = label.item()
        else:
            item_to_evaluate = top_pred.item()

        # Get CAM
        if opt.cam_type == 'CAM':
            cam = cam_extractor(class_idx=item_to_evaluate)[0] 
        elif opt.cam_type == 'GradCAM':
            cam = cam_extractor(class_idx=item_to_evaluate, scores=out)[0]
        elif opt.cam_type == 'GradCAMpp':
            cam = cam_extractor(class_idx=item_to_evaluate, scores=out)[0]
        elif opt.cam_type == 'ScoreCAM':
            cam = cam_extractor(class_idx=item_to_evaluate)[0]
        elif opt.cam_type == 'LayerCAM':
            cam = cam_extractor(class_idx=item_to_evaluate, scores=out)[0]            
        else:
            raise NotImplementedError(opt.cam_type) 

        cam = resize_tr(cam)

        # Apply thresholding
        model_mask = thresholding(cam, opt.segmentation_threshold)

        # GT segmantation mask
        gt_mask = fill_rectangle(opt.image_size_dataset, ann[0], ann[1], ann[2], ann[3], device)

        jaccard_index_values.append(jaccard_index(model_mask, gt_mask).item())
        precision_score_values.append(precision_score(model_mask, gt_mask).item())
        recall_score_values.append(recall_score(model_mask, gt_mask).item())
        accuracy_values.append(accuracy(model_mask, gt_mask).item())
        dice_coef_values.append(dice_coef(model_mask, gt_mask).item())

    xai_metrics = {'IoU': np.nanmean(jaccard_index_values), \
        'Precision': np.nanmean(precision_score_values), \
            'Recall': np.nanmean(recall_score_values), \
                'Accuracy': np.nanmean(accuracy_values), \
                    'Dice_coeff': np.nanmean(dice_coef_values)}


    print(f"Jaccard Index (IoU): {np.nanmean(jaccard_index_values):.3f}  \
        -- Percentage NaNs ({sum(np.isnan(jaccard_index_values))/len(jaccard_index_values)}%)")
    print(f"Precision Score:{np.nanmean(precision_score_values):.3f}  \
                -- Percentage NaNs ({sum(np.isnan(precision_score_values))/len(precision_score_values)}%)")
    print(f"Recall Score: {np.nanmean(recall_score_values):.3f}   \
                -- Percentage NaNs ({sum(np.isnan(recall_score_values))/len(recall_score_values)}%)")
    print(f"Accuracy: {np.nanmean(accuracy_values):.3f}   \
                -- Percentage NaNs ({sum(np.isnan(accuracy_values))/len(accuracy_values)}%)")
    print(f"Dice coefficient: {np.nanmean(dice_coef_values):.3f}  \
                -- Percentage NaNs ({sum(np.isnan(dice_coef_values))/len(dice_coef_values)}%)")


    # Save results
    save_xai_metrics_to_csv(results_path_folder, xai_metrics)

    cam_extractor.remove_hooks()

if __name__ == '__main__':
    main()




