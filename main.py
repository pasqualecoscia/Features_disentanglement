# Import dependencies
import os
import time
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import models, transforms as T
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from utils.dataset_utils import load_tiny_dataset
from utils.model_utils import *
from utils.config_params import get_config_parser
from utils.LR_finder import LRFinder, plot_lr_finder
from utils.logger import Logger

def main():
    # This file performs both training and evalution for the specified models

    # Print the default configuration
    opt = get_config_parser()
    print(vars(opt))

    # Save path
    saved_model_path = f'{opt.model}_{opt.training_type}_{str(opt.num_classes)}_{str(opt.data_percentage)}_{str(opt.epochs)}.pt'
    results_path_folder = os.path.join(opt.results_path, saved_model_path.split('.')[0])

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
    
    # # Plot augmented images SimCLR
    # NUM_IMAGES = 12
    # imgs = []
    # for idx in range(NUM_IMAGES):
    #     img1 = validation_dataset[idx][0]
    #     img2 = validation_dataset[idx][1]
    #     imgs.append(img1)
    #     imgs.append(img2)

    # img_grid = torchvision.utils.make_grid(torch.stack(imgs, dim=0), nrow=6, normalize=True, pad_value=0.9)
    # img_grid = img_grid.permute(1, 2, 0)

    # plt.figure(figsize=(10,5))
    # plt.title('Augmented image examples of the STL10 dataset')
    # plt.imshow(img_grid)
    # plt.axis('off')
    # plt.show()
    # plt.savefig('./test_aug_simclr.png')
    # plt.close()

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
                        
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, **kwargs) 
    
    # Create model
    model = model_creation(opt)

    if opt.dis_loss:
        # If disentangle loss is active re-define the model adding additional encoders
        model = ModelWithEncoders(model, opt)    

    if opt.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        

    device = torch.device('cuda' if (torch.cuda.is_available() and opt.use_cuda) else 'cpu')

    if opt.training_type == 'colorization':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    if opt.lr_finder and opt.training_type not in ['colorization', 'simclr']: # Find best learning rate
        # Not supported for colorization and simclr
        END_LR = 10
        NUM_ITER = 500

        lr_finder = LRFinder(model, optimizer, criterion, device, results_path_folder)
        lrs, losses = lr_finder.range_test(train_loader, END_LR, NUM_ITER)
        
        plot_lr_finder(lrs, losses, save_path=results_path_folder)

    
    if opt.scheduler == 'OneCycleLR':
        STEPS_PER_EPOCH = len(train_loader)
        TOTAL_STEPS = opt.epochs * STEPS_PER_EPOCH

        MAX_LRS = 5e-2

        scheduler = lr_scheduler.OneCycleLR(optimizer,
                                        max_lr = MAX_LRS,
                                        total_steps = TOTAL_STEPS)
    elif opt.scheduler == 'LambdaLR':
        lambda1 = lambda epoch: 0.95 ** epoch
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    print(scheduler)

    best_valid_loss = float('inf')

    train_losses = []
    valid_losses = []
    train_accs_1 = []
    valid_accs_1 = []


    for epoch in range(opt.epochs):
        
        start_time = time.monotonic()
        
        if opt.training_type == 'rotation':
            topk = False
        else:
            topk = True

        train_loss, train_acc_1, train_acc_5 = train(model, train_loader, optimizer, criterion, scheduler, device, opt, topk)
        valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, validation_loader, criterion, device, opt, topk)
            
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(results_path_folder, saved_model_path))

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if epoch % 10 == 0:
            #print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | Learning rate: {scheduler.get_last_lr()[0]:1.5f}')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ' \
                f'Train Acc @5: {train_acc_5*100:6.2f}% [Epoch: {epoch}]')
            print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ' \
                f'Valid Acc @5: {valid_acc_5*100:6.2f}%')
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs_1.append(train_acc_1)
        valid_accs_1.append(valid_acc_1)

    if opt.save_plots:
        plot_loss_and_accuracy(opt.epochs, train_losses, valid_losses, train_accs_1, valid_accs_1, save_path=results_path_folder)

    # Evaluation section
    model.load_state_dict(torch.load(os.path.join(results_path_folder, saved_model_path)))

    test_loss, test_acc_1, test_acc_5 = evaluate(model, test_loader, criterion, device, opt, topk)

    results = f'Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1*100:6.2f}% | ' \
        f'Test Acc @5: {test_acc_5*100:6.2f}%'

    print(results)
    save_results(results_path_folder, results)

if __name__ == '__main__':
    main()