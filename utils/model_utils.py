import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torch import nn
import torch.nn.functional as F

def calculate_topk_accuracy(y_pred, y, k = 5, topk=True):
    with torch.no_grad():
        
        if not topk:
            k = 1 # Do not compute topk

        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k

def train(model, iterator, optimizer, criterion, scheduler, device, opt, topk=True):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    # Disentangle loss
    dis_loss = 0

    model.train()
    
    for x,y,_ in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()

        loss, acc_1, acc_5 = inner_training_and_accuracy(x, y, opt, model, criterion, device, topk)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()


    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_5

def evaluate(model, iterator, criterion, device, opt, topk=True):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for x,y,_ in iterator:

            x = x.to(device)
            y = y.to(device)

            loss, acc_1, acc_5 = inner_training_and_accuracy(x, y, opt, model, criterion, device, topk)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()
            
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_5

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def smooth_fc(_list, smooth_f=0.2):
    ''' Smoothing list of elements
    '''
    l = _list.copy()
    for i in range(1, len(_list)):
        l[i] = smooth_f * l[i] + (1 - smooth_f) * l[i-1]
    
    return l

def inner_training_and_accuracy(x, y, opt, model, criterion, device, topk):
    '''
        Inner training and accuracy computation
    '''
    # For SimCLR training, use InfoNCELoss
    if opt.training_type == 'simclr':
        assert (not opt.dis_loss), "Disentagled loss not supported for SimCLR method."
        loss, acc_1, acc_5 = model.info_nce_loss([x,y])
    else:
        if opt.dis_loss:
            y_pred, b_out, c_out = model(x)
            
            loss1 = criterion(y_pred, y)

            cos = nn.CosineSimilarity(dim=-1)

            b_out_norm = F.normalize(b_out.view(b_out.size(0), opt.num_classes, -1), 
                                    p=2.0, dim=-1)
            c_out_norm = F.normalize(c_out.view(b_out.size(0), opt.num_classes, -1),
                                    p=2.0, dim=-1)                                    
            dis_loss = cos(b_out_norm, c_out_norm)
            dis_loss = dis_loss.sum().mean(dim=-1)
                                                      
            # b_out_avg = F.adaptive_avg_pool2d(input=b_out, output_size=(1,1)).squeeze()
            # c_out_avg = F.adaptive_avg_pool2d(input=c_out, output_size=(1,1)).squeeze()
            # loss2 = criterion(b_out_avg, y)
            # loss3 = criterion(c_out_avg, y)

            # Pairwise cosine similarities
            pairwise_cos_sim_b = F.cosine_similarity(b_out_norm[:,None, :,:], b_out_norm[:,:, None,:], dim=-1)
            loss2 = torch.triu(pairwise_cos_sim_b, diagonal=1).sum(dim=(1,2)).mean()                
            pairwise_cos_sim_c = F.cosine_similarity(c_out_norm[:,None, :,:], c_out_norm[:,:, None,:], dim=-1)
            loss3 = torch.triu(pairwise_cos_sim_c, diagonal=1).sum(dim=(1,2)).mean() 

            loss = loss1 + opt.lambda_dls * (dis_loss + loss2 + loss3)

        else:
            y_pred = model(x)
            loss = criterion(y_pred, y)
        
        if opt.training_type == 'colorization':
            acc_1, acc_5 = torch.zeros(1).to(device), torch.zeros(1).to(device)
        else:
            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y, 5, topk=topk)

    return loss, acc_1, acc_5 

def plot_loss_and_accuracy(EPOCHS, train_losses, valid_losses, train_accs_1, valid_accs_1, save_path=None):
    # Check if the folder exists, and create it if not
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(np.arange(EPOCHS), smooth_fc(train_losses))
    ax1.plot(np.arange(EPOCHS), smooth_fc(valid_losses))
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train Loss', 'Valid Loss'])
    ax1.grid(True)

    ax2.plot(np.arange(EPOCHS), smooth_fc(train_accs_1))
    ax2.plot(np.arange(EPOCHS), smooth_fc(valid_accs_1))
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train Acc@1', 'Valid Acc@1'])
    ax2.grid(True)

    if save_path:
        plt.savefig(os.path.join(save_path, 'plot_results.png'))
    else:
        plt.show()    

    return None

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    def __str__(self):
        attributes = '\n'.join(f'{key}: {value}' for key, value in self.__dict__.items())
        return f"Config:\n{attributes}"


def save_results(folder_path, results):
    # Check if the folder exists, and create it if not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create a filename based on the model name
    file_path = os.path.join(folder_path, 'results.txt')

    # Open the file in write mode and write the string
    with open(file_path, 'w') as file:
        file.write(results)

def copy_weights(source_model, target_model):

    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()

    new_state_dictionary = dict()
    # Filter out unnecessary keys
    for k,v in source_state_dict.items():
        if 'downsample' in k:
            k = k.replace('downsample.','')
        if k in target_state_dict:
           new_state_dictionary[k] = v 

    #source_state_dict = {k: v for k, v in source_state_dict.items() if k in target_state_dict}

    # Update the target model's state dict with the source model's weights
    target_state_dict.update(new_state_dictionary)

    # Load the updated state dict into the target model
    target_model.load_state_dict(target_state_dict)

    print("Source model copied to target model.")


class SimCLR(nn.Module):
    def __init__(self, model_name, temperature, hidden_dim):
        super().__init__()
        
        self.model_name = model_name
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.model = None

        assert self.temperature > 0.0, 'The temperature must be a positive float!'

        if self.model_name == 'resnet18':
            # Base model f(.)
            self.model = models.resnet18(num_classes=4*self.hidden_dim)  # Output of last linear layer
        elif self.model_name == 'resnet50':
            # Base model f(.)
            self.model = models.resnet50(num_classes=4*self.hidden_dim)  # Output of last linear layer
        elif self.model_name == 'resnet101':
            # Base model f(.)
            self.model = models.resnet101(num_classes=4*self.hidden_dim)  # Output of last linear layer
        elif self.model_name == 'vgg19':
            # Base model f(.)
            self.model = models.vgg19(num_classes=4*self.hidden_dim)  # Output of last linear layer
        else:
            raise NotImplementedError('Model not implemented!')

        if 'resnet' in self.model_name:
            # The MLP for g(.) consists of Linear->ReLU->Linear
            self.model.fc = nn.Sequential(
                self.model.fc,  # Linear(ResNet output, 4*hidden_dim)
                nn.ReLU(inplace=True),
                nn.Linear(4*self.hidden_dim, self.hidden_dim)
            )
        else: #VGG19
             # The MLP for g(.) consists of Linear->ReLU->Linear
            self.model.classifier = nn.Sequential(
                self.model.classifier,  # Linear(VGG output, 4*hidden_dim)
                nn.ReLU(inplace=True),
                nn.Linear(4*self.hidden_dim, self.hidden_dim)
            )           
    
    def forward(self, X):
        features = self.model(X)
        return features

    def info_nce_loss(self, batch):
        imgA, imgB = batch
        imgs = torch.cat((imgA, imgB), dim=0)

        # Encode all images
        feats = self.model(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Ranking metrics
        acc_top1 = (sim_argsort == 0).float().mean()
        acc_top5 = (sim_argsort < 5).float().mean() 

        return nll, acc_top1, acc_top5

class Colorizer(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        
        self.model_name = model_name

        if self.model_name == 'vgg19':

            self.downsample = models.vgg19()
            self.downsample.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.downsample.avgpool = nn.Identity()
            self.downsample.classifier = nn.Identity()  

            upsampling_layers_first = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )

        elif self.model_name in ['resnet50', 'resnet101']:
            self.downsample = models.resnet50() if model_name == 'resnet50' else models.resnet101()

            self.downsample.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            self.downsample.avgpool = nn.Identity()
            self.downsample.fc = nn.Identity()
            
            upsampling_layers_first = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )

        elif self.model_name == 'resnet18':
            self.downsample = models.resnet18()

            self.downsample.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            self.downsample.avgpool = nn.Identity()
            self.downsample.fc = nn.Identity()
            
            upsampling_layers_first = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )

        else:
            raise NotImplementedError('Model not implemented.')

        self.upsample = nn.Sequential(
              upsampling_layers_first,
              nn.Upsample(scale_factor=2),
              nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(256),
              nn.ReLU(),
              nn.Upsample(scale_factor=2),
              nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(128),
              nn.ReLU(),
              nn.Upsample(scale_factor=2),
              nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(64),
              nn.ReLU(),
              nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(32),
              nn.ReLU(),
              nn.Upsample(scale_factor=2),
              nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(16),
              nn.ReLU(),
              nn.Upsample(scale_factor=2),
              nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)
            )

    def forward(self, X):
        features = self.downsample(X)
        if self.model_name in ['resnet50', 'resnet101']:
            features = features.reshape(features.size(0), 2048, 7, 7) # Reshape due to a flatten operation
        else:
            features = features.reshape(features.size(0), 512, 7, 7) # Reshape due to a flatten operation
        upsampled = self.upsample(features)
        return upsampled

def model_creation(opt):

    saved_model_path = f'{opt.model}_{opt.training_type.split("_")[0]}_{str(opt.num_classes)}_{str(opt.data_percentage)}_{str(opt.epochs)}.pt'
    results_path_folder = os.path.join(opt.results_path, saved_model_path.split('.')[0])  

    if opt.model == 'resnet18':
        if opt.training_type in ['from_scratch', 'rotation']:
            model = models.resnet18() # No pre-training
        elif opt.training_type == 'imagenet_pretrained':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest classes should be False')
            model = models.resnet18(weights='IMAGENET1K_V1') # Pre-trained
            # Do not update the feature extraction section
            for param in model.parameters():
                param.requires_grad = False
        elif opt.training_type in 'rotation_finetuning':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest_classes should be False')
            model = models.resnet18() # No pre-training
            IN_FEATURES = model.fc.in_features 
            OUTPUT_DIM = len(list(opt.degrees))
            model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)           
            # Save path
            model.load_state_dict(torch.load(os.path.join(results_path_folder, saved_model_path)))
        elif opt.training_type == 'colorization_finetuning':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest_classes should be False')
            model = Colorizer(opt.model) # No pre-training
            # Load model
            model.load_state_dict(torch.load(os.path.join(results_path_folder, saved_model_path)))      

            model_resnet18 = models.resnet18()
            # Restore original layers
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            model.fc = nn.Linear(in_features=512, out_features=1000, bias=True)

            copy_weights(model, model_resnet18)

            # Create new resnet model
            IN_FEATURES = model_resnet18.fc.in_features 
            OUTPUT_DIM = opt.num_classes
            model_resnet18.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM) 
            return model_resnet18

        elif opt.training_type == 'simclr_finetuning':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest_classes should be False')
            model = SimCLR(opt.model, opt.temperature, opt.hidden_dim) # No pre-training
            # Load saved model
            model.load_state_dict(torch.load(os.path.join(results_path_folder, saved_model_path)))

            model_resnet18 = models.resnet18()
            IN_FEATURES = model_resnet18.fc.in_features
            OUTPUT_DIM = opt.num_classes 
            # Restore original layers
            model.model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

            model_resnet18.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

            copy_weights(model.model, model_resnet18)

            return model_resnet18

        elif opt.training_type == 'colorization':
            return Colorizer(opt.model)

        elif opt.training_type == 'simclr':
            return SimCLR(opt.model, opt.temperature, opt.hidden_dim)        
        else:
            raise NotImplementedError(f'{opt.training_type} not implemented.')

        IN_FEATURES = model.fc.in_features 
        
        if opt.training_type == 'rotation':
            OUTPUT_DIM = len(list(opt.degrees))
        else:
            OUTPUT_DIM = opt.num_classes
        
        model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    
    elif opt.model == 'resnet50':
        if opt.training_type in ['from_scratch', 'rotation']:
            model = models.resnet50() # No pre-training
        elif opt.training_type == 'imagenet_pretrained':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest classes should be False')
            model = models.resnet50(weights='IMAGENET1K_V1') # Pre-trained
            # Do not update the feature extraction section
            for param in model.parameters():
                param.requires_grad = False
        elif opt.training_type == 'rotation_finetuning':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest_classes should be False')
            model = models.resnet50() # No pre-training
            IN_FEATURES = model.fc.in_features 
            OUTPUT_DIM = len(list(opt.degrees))
            model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)           
            # Save path
            model.load_state_dict(torch.load(os.path.join(results_path_folder, saved_model_path)))                
        elif opt.training_type == 'colorization_finetuning':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest_classes should be False')
            model = Colorizer(opt.model) # No pre-training
            # Load model
            model.load_state_dict(torch.load(os.path.join(results_path_folder, saved_model_path)))      

            model_resnet50 = models.resnet50()
            # Restore original layers
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)

            copy_weights(model, model_resnet50)

            # Create new resnet model
            IN_FEATURES = model_resnet50.fc.in_features 
            OUTPUT_DIM = opt.num_classes
            model_resnet50.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM) 
            return model_resnet50
        elif opt.training_type == 'simclr_finetuning':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest_classes should be False')
            model = SimCLR(opt.model, opt.temperature, opt.hidden_dim) # No pre-training
            # Load saved model
            model.load_state_dict(torch.load(os.path.join(results_path_folder, saved_model_path)))

            model_resnet50 = models.resnet50()
            IN_FEATURES = model_resnet50.fc.in_features
            OUTPUT_DIM = opt.num_classes 
            # Restore original layers
            model.model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

            model_resnet50.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

            copy_weights(model.model, model_resnet50)

            return model_resnet50  

        elif opt.training_type == 'simclr':
            return SimCLR(opt.model, opt.temperature, opt.hidden_dim) 
        elif opt.training_type == 'colorization':
            return Colorizer(opt.model)
        else:
            raise NotImplementedError(f'{opt.training_type} not implemented.')

        IN_FEATURES = model.fc.in_features 
        
        if opt.training_type == 'rotation':
            OUTPUT_DIM = len(list(opt.degrees))
        else:
            OUTPUT_DIM = opt.num_classes
        
        model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

    elif opt.model == 'resnet101':
        if opt.training_type in ['from_scratch', 'rotation']:
            model = models.resnet101() # No pre-training
        elif opt.training_type == 'imagenet_pretrained':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest classes should be False')
            model = models.resnet101(weights='IMAGENET1K_V1') # Pre-trained
            # Do not update the feature extraction section
            for param in model.parameters():
                param.requires_grad = False
        elif opt.training_type == 'rotation_finetuning':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest_classes should be False')
            model = models.resnet101() # No pre-training
            IN_FEATURES = model.fc.in_features 
            OUTPUT_DIM = len(list(opt.degrees))
            model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)           
            # Save path
            model.load_state_dict(torch.load(os.path.join(results_path_folder, saved_model_path)))                
        elif opt.training_type == 'colorization_finetuning':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest_classes should be False')
            model = Colorizer(opt.model) # No pre-training
            # Load model
            model.load_state_dict(torch.load(os.path.join(results_path_folder, saved_model_path)))      

            model_resnet101 = models.resnet101()
            # Restore original layers
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)

            copy_weights(model, model_resnet101)

            # Create new resnet model
            IN_FEATURES = model_resnet101.fc.in_features 
            OUTPUT_DIM = opt.num_classes
            model_resnet101.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM) 
            return model_resnet101
        elif opt.training_type == 'simclr_finetuning':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest_classes should be False')
            model = SimCLR(opt.model, opt.temperature, opt.hidden_dim) # No pre-training
            # Load saved model
            model.load_state_dict(torch.load(os.path.join(results_path_folder, saved_model_path)))

            model_resnet101 = models.resnet101()
            IN_FEATURES = model_resnet101.fc.in_features
            OUTPUT_DIM = opt.num_classes 
            # Restore original layers
            model.model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

            model_resnet101.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

            copy_weights(model.model, model_resnet101)

            return model_resnet101 
                        
        elif opt.training_type == 'simclr':
            return SimCLR(opt.model, opt.temperature, opt.hidden_dim)
        elif opt.training_type == 'colorization':
            return Colorizer(opt.model)
        else:
            raise NotImplementedError(f'{opt.training_type} not implemented.')

        IN_FEATURES = model.fc.in_features 
        
        if opt.training_type == 'rotation':
            OUTPUT_DIM = len(list(opt.degrees))
        else:
            OUTPUT_DIM = opt.num_classes
        
        model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

    elif opt.model == 'vgg19':
        if opt.training_type in ['from_scratch', 'rotation']:
            model = models.vgg19() # No pre-training
        elif opt.training_type == 'imagenet_pretrained':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest classes should be False')
            model = models.vgg19(weights='IMAGENET1K_V1') # Pre-trained
            # Do not update the feature extraction section
            for param in model.parameters():
                param.requires_grad = False
        elif opt.training_type == 'rotation_finetuning':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest_classes should be False')
            model = models.vgg19() # No pre-training
            IN_FEATURES = model.classifier[0].in_features 
            OUTPUT_DIM = len(list(opt.degrees))
            model.classifier = nn.Linear(IN_FEATURES, OUTPUT_DIM)
            # Save path
            model.load_state_dict(torch.load(os.path.join(results_path_folder, saved_model_path)))
        elif opt.training_type == 'colorization_finetuning':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest_classes should be False')
            model = Colorizer(opt.model) # No pre-training
            # Load model
            model.load_state_dict(torch.load(os.path.join(results_path_folder, saved_model_path)))      

            model_vgg19 = models.vgg19()
            # Restore original layers
            model.downsample.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            model.downsample.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
            model.downsample.classifier = nn.Sequential(*list(model_vgg19.classifier.children()))

            copy_weights(model, model_vgg19)

            # Create new vgg19 model
            IN_FEATURES = model_vgg19.classifier[0].in_features 
            OUTPUT_DIM = opt.num_classes
            model_vgg19.classifier = nn.Linear(IN_FEATURES, OUTPUT_DIM) 
            return model_vgg19
        elif opt.training_type == 'simclr_finetuning':
            if opt.closest_classes:
                raise NotImplementedError('opt.closest_classes should be False')
            model = SimCLR(opt.model, opt.temperature, opt.hidden_dim) # No pre-training
            # Load saved model
            model.load_state_dict(torch.load(os.path.join(results_path_folder, saved_model_path)))

            model_vgg19 = models.vgg19()
            IN_FEATURES = model_vgg19.classifier[0].in_features
            OUTPUT_DIM = opt.num_classes 
            # Restore original layers
            model.model.classifier = nn.Linear(IN_FEATURES, OUTPUT_DIM)

            model_vgg19.classifier = nn.Linear(IN_FEATURES, OUTPUT_DIM)

            copy_weights(model.model, model_vgg19)

            return model_vgg19

        elif opt.training_type == 'simclr':
            return SimCLR(opt.model, opt.temperature, opt.hidden_dim)
        elif opt.training_type == 'colorization':
            return Colorizer(opt.model)                            
        else:
            raise NotImplementedError(f'{opt.training_type} not implemented.')

        IN_FEATURES = model.classifier.in_features if opt.training_type == 'rotation_finetuning' else model.classifier[0].in_features
        
        if opt.training_type == 'rotation':
            OUTPUT_DIM = len(list(opt.degrees))
        else:
            OUTPUT_DIM = opt.num_classes
        
        model.classifier = nn.Linear(IN_FEATURES, OUTPUT_DIM)

    return model


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x) 

class ModelWithEncoders(nn.Module):
    def __init__(self, base_model, opt):
        super(ModelWithEncoders, self).__init__()

        # Add the base model (previously created model)
        self.base_model = base_model

        self.opt = opt

        if opt.model in ['resnet18', 'vgg19']:
            self.in_channels = 512
        elif opt.model in ['resnet50', 'resnet101']:
            self.in_channels = 2048
        else:
            raise ValueError('in_channels not computed!.')    
        # Add the first encoder
        self.background_encoder = EncoderBlock(in_channels=self.in_channels, out_channels=opt.num_classes)

        # Add the second encoder
        self.content_encoder = EncoderBlock(in_channels=self.in_channels, out_channels=opt.num_classes)

        # Placeholder for the extracted features
        self.extracted_features = None

        # Target layer to extract features from
        self.target_layer_name = opt.target_layer_name

    def extract_features(self, x):
        # Dictionary to store features at each layer
        features_by_layer = {}

        output_ = x
        # Forward pass through the base model layer by layer
        for name, layer in self.base_model.named_children():
            input_ = output_
            if name == 'classifier': #for vgg19
                size_ = input_.size()
                input_ = input_.reshape(x.size(0), -1)
            output_ = layer(input_.squeeze()) # Squeeze x for FC layer

            # Store features if the current layer matches the target layer
            if name == self.target_layer_name:
                features_by_layer[name] = input_.clone()

        # Set the extracted features
        self.extracted_features = features_by_layer.get(self.target_layer_name)

        # Ensure that extracted features are set
        if self.extracted_features is None:
            raise ValueError("Extracted features are not set. Check if the target layer name is correct.")

        return output_

    def forward(self, x):
        # Extract features layer by layer
        out = self.extract_features(x)

        # Forward pass through the first encoder
        b_enc = self.background_encoder(self.extracted_features)

        # Forward pass through the second encoder
        c_enc = self.content_encoder(self.extracted_features)

        return out, b_enc, c_enc