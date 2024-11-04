import os
import requests
import zipfile
import io
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.datasets import ImageFolder
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from .transforms_utils import *
from nltk.corpus import wordnet
from itertools import product

from skimage.color import rgb2lab, rgb2gray, lab2rgb

def load_tiny_dataset(opt):

    DATA_DIR = 'datasets/tiny-imagenet-200'  # Original images come in shapes of [3,64,64]
    TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    if not (os.path.exists(DATA_DIR) and os.path.isdir(DATA_DIR)):
        make_tiny_image_net(DATA_DIR)

    classes = [item for item in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, item)) ]

    # Randomly select num_classes
    np.random.seed(opt.np_random_seed) 
    targets = np.random.choice(len(classes), opt.num_classes, replace=False)

    selected_classes = [classes[i] for i in targets]    

    # Load data
    transform = transforms.Compose([transforms.ToTensor()])

    # Save class names (for corresponding labels) as dict from words.txt file
    class_to_name_dict = dict()
    class_to_name_dict_reverse = dict()

    fp = open(os.path.join(DATA_DIR, 'words.txt'), 'r')
    data = fp.readlines()
    for line in data:
        words = line.strip('\n').split('\t')
        class_to_name_dict[words[0]] = words[1].split(',')[0]
    fp.close()

    # Reverse dictionary
    class_to_name_dict_reverse = {value: key for key, value in class_to_name_dict.items()}

    # Print selected classes
    print("Selected classes from TinyImageNet:")
    for cl in selected_classes:
        print(f"{cl}: {class_to_name_dict[cl]}")

    # Check if the training type is rotation or colorization
    if opt.training_type in ['rotation', 'colorization', 'simclr']:
        # Check if closest_classes option is set to False
        if not opt.closest_classes:
            # Raise NotImplementedError if closest_classes is False
            raise NotImplementedError("Rotation/Colorization training without closest classes is not implemented.")

    if opt.closest_classes:
        # Select closest classes to the selected ones for pre-training
        selected_classes_names = []
        for i, cl in enumerate(selected_classes):
            selected_classes_names.append(class_to_name_dict[cl])
        
        reduced_classes = remove_elements(selected_classes, classes)

        reduced_classes_names_dict = dict()
        for i, cl in enumerate(reduced_classes):
            reduced_classes_names_dict[cl] = class_to_name_dict[cl]

        reduced_classes_names_reverse_dict = {value:key for key, value in reduced_classes_names_dict.items()}
        reduced_classes_names = [value for key, value in reduced_classes_names_dict.items()]

        # Find similarity between all pairs
        similarity_pairs = find_similarity_pairs(selected_classes_names, reduced_classes_names)

        # Sort the list based on the similarity value in descending order
        sorted_data = sorted(similarity_pairs, key=lambda x: x[1], reverse=True)

        # Extract the sorted names from the tuples
        sorted_names = [tup[0][1] for tup in sorted_data]

        # Remove duplicated names
        sorted_names_ = []
        [sorted_names_.append(name) for name in sorted_names if name not in sorted_names_]
        assert len(sorted_names_) >= opt.num_closest_classes, "Closest classes not enough."

        closest_classes_names = sorted_names_[:opt.num_closest_classes]
        closest_classes = [reduced_classes_names_reverse_dict[class_] for class_ in closest_classes_names]
        
        if opt.training_type == 'rotation':
            degrees = list(opt.degrees)
            custom_dataset = CustomDatasetRotation(TRAIN_DIR, closest_classes, transform, degrees)
        elif opt.training_type == 'colorization':
            transform_colorization = colorization_transformations() # 224 image_size hard-coded
            custom_dataset = CustomDatasetColorization(TRAIN_DIR, closest_classes, transform_colorization)
        elif opt.training_type == 'simclr':
            transform_simclr = simclr_transformations() # 224 image_size hard-coded
            custom_dataset = CustomDatasetSimCLR(TRAIN_DIR, closest_classes, transform_simclr)                         
        else:
            custom_dataset = CustomDataset(TRAIN_DIR, closest_classes, transform, 'train')
        
        # Overwrite selected classes
        selected_classes = closest_classes

        # Print selected classes
        print("Selected closest classes from TinyImageNet:")
        for cl in selected_classes:
            print(f"{cl}: {class_to_name_dict[cl]}")

    else:
        custom_dataset = CustomDataset(TRAIN_DIR, selected_classes, transform, 'train')

    labels = [custom_dataset.samples[i][1] for i in range(len(custom_dataset.samples))]
    train_idx, validation_idx = train_test_split(np.arange(len(custom_dataset)),
                                                test_size=0.1,
                                                random_state=opt.random_state_split,
                                                shuffle=True,
                                                stratify=labels)

    labels_arr = [labels[i] for i in train_idx]
    
    if opt.training_type == 'rotation':
        num_samples_per_class_before = len(labels_arr)/len(degrees)
    else:
        if opt.closest_classes:
            num_samples_per_class_before = len(labels_arr)/opt.num_closest_classes
        else:
            num_samples_per_class_before = len(labels_arr)/opt.num_classes

    num_samples_per_class_after = np.floor(num_samples_per_class_before * opt.data_percentage /100).astype(int)

    train_idx_new = []
    for k in np.unique(labels):
        idx = np.array(train_idx)[np.array(labels_arr)==k]
        l = np.random.choice(idx, num_samples_per_class_after, replace=False)
        train_idx_new.extend(l)
    # Subset dataset for train and val
    train_dataset_ = torch.utils.data.Subset(custom_dataset, train_idx_new)
    validation_dataset_ = torch.utils.data.Subset(custom_dataset, validation_idx)

    if opt.training_type not in ['colorization', 'simclr']:
        # Dataloader for train and val
        train_loader = DataLoader(train_dataset_, batch_size=1, shuffle=True)
        validation_loader = DataLoader(validation_dataset_, batch_size=1, shuffle=False)

        # Compute mean and std from training set
        means = torch.zeros(3)
        stds = torch.zeros(3)

        print("Computing normalization values...")
        for img, label, _ in train_loader:
            means += torch.mean(img[0], dim = (1,2))
            stds += torch.std(img[0], dim = (1,2))

        means /= len(train_loader)
        stds /= len(train_loader)
            
        print(f'Calculated means: {means}')
        print(f'Calculated stds: {stds}')
    else:
        means = torch.zeros(1,3)
        stds = torch.zeros(1,3)
        
    #Import transformations
    if opt.training_type == 'rotation':
        preprocess_transform_train = val_transformations(means, stds) # Do not apply data augmentation
    elif opt.training_type in ['colorization', 'simclr']:
        pass
    else:
        preprocess_transform_train = train_transformations(means, stds)

    if opt.training_type not in ['colorization', 'simclr']:
        preprocess_transform_val = val_transformations(means, stds)
        preprocess_transform_test = test_transformations(means, stds)

    if opt.training_type == 'colorization':
        train_dataset = MyDataset(train_dataset_, transform=transform_colorization)
        validation_dataset = MyDataset(validation_dataset_, transform=transform_colorization)
    elif opt.training_type == 'simclr':
        train_dataset = MyDataset(train_dataset_, transform=None)
        validation_dataset = MyDataset(validation_dataset_, transform=None)               
    else:
        train_dataset = MyDataset(train_dataset_, transform=preprocess_transform_train)
        validation_dataset = MyDataset(validation_dataset_, transform=preprocess_transform_val)

    if opt.training_type == 'rotation':
        test_dataset = CustomDatasetRotation(VALID_DIR, selected_classes, preprocess_transform_test, degrees)
    elif opt.training_type == 'colorization':
        test_dataset = CustomDatasetColorization(VALID_DIR, selected_classes, transform_colorization)
    elif opt.training_type == 'simclr':
        test_dataset = CustomDatasetSimCLR(VALID_DIR, selected_classes, transform_simclr)            
    else:
        test_dataset = CustomDataset(VALID_DIR, selected_classes, preprocess_transform_test, 'val')

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(validation_dataset)}")
    print(f"Number of testing images: {len(test_dataset)}")

    if opt.training_type not in  ['colorization', 'simclr']:
        unique, counts = get_statistics(train_dataset)
        print(f"N. images (train set): {dict(zip(unique, counts))}")
        unique, counts = get_statistics(validation_dataset)
        print(f"N. images (validation set): {dict(zip(unique, counts))}")
        unique, counts = get_statistics(test_dataset)
        print(f"N. images (test set): {dict(zip(unique, counts))}")

    return train_dataset, validation_dataset, test_dataset, (means, stds, class_to_name_dict, class_to_name_dict_reverse)

def make_tiny_image_net(DATA_DIR):
    # Create the directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download the file
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    # Extract the contents into the specified directory
    zip_file.extractall(DATA_DIR)

    # Move up-folder validation annotations
    VAL_ANNOTATIONS_PATH = f'{VALID_DIR}/val_annotations.txt'
    VAL_ANNOTATIONS_PATH_ = os.path.join(os.path.dirname(VALID_DIR), 'val_annotations.txt')
    os.rename(VAL_ANNOTATIONS_PATH, VAL_ANNOTATIONS_PATH_)
    print("VAL annotations file moved!")
    #Create separate validation subfolders for the validation images based on
    #their labels indicated in the val_annotations txt file
    val_img_dir = os.path.join(VALID_DIR, 'images')

    # Open and read val annotations text file
    fp = open(VAL_ANNOTATIONS_PATH_, 'r')
    data = fp.readlines()

    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    for img, folder in val_img_dict.items():
        newpath = (os.path.join(os.path.dirname(val_img_dir), folder, 'images'))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

    try:
        os.rmdir(val_img_dir)
        # or os.removedirs(folder_path)
        print(f"Folder '{val_img_dir}' removed successfully.")
    except OSError as e:
        print(f"Error: {e}")

class CustomDataset(Dataset):
    # Custom tiny image net dataset
    def __init__(self, root_dir, selected_classes, transform=None, set_type='train'):
        self.root_dir = root_dir
        self.selected_classes = selected_classes
        self.transform = transform
        self.set_type = set_type

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.selected_classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.selected_classes)}

        self.dataset = ImageFolder(root=root_dir, transform=transform)

        # Remap
        self.class_to_idx_ext = {cls: idx for idx, cls in enumerate(self.dataset.classes)} # Extended to all classes
        self.idx_to_class_ext = {v: k for k, v in self.class_to_idx_ext.items()}

        l = []
        for key in self.selected_classes:
            l.append(self.class_to_idx_ext.get(key))

        self.map = {cls:idx for idx, cls in enumerate(sorted(l))}
        self.map_swap = {v: k for k, v in self.map.items()}
        
        # Filter out samples not in selected classes
        self.samples = [(path, label) for path, label in self.dataset.samples if self.dataset.classes[label] in self.selected_classes]

    def __getitem__(self, index):
        path, label = self.samples[index]
        label = self.map[label] # Remap
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        # Load annotations
        df = read_annotations(path, self.set_type) # Bounding boxes as list (x1, y1, x2, y2)
        return img, label, df

    def __len__(self):
        return len(self.samples)

    def loader(self, path):
        return Image.open(path).convert('RGB')

class CustomDatasetRotation(Dataset):
    # Custom tiny image net dataset
    def __init__(self, root_dir, selected_classes, transform, degrees):
        self.root_dir = root_dir
        self.selected_classes = selected_classes
        self.transform = transform
        self.degrees = degrees

        self.dataset = ImageFolder(root=root_dir, transform=transform)
       
        # Filter out samples not in selected classes
        self.samples = [(path, label_rotation) for label_rotation in list(range(len(self.degrees))) for path, label in self.dataset.samples if self.dataset.classes[label] in self.selected_classes]

    def __getitem__(self, index):
        path, label = self.samples[index]
        degree = self.degrees[label]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        rotated_img = F.rotate(img, degree)
        label = torch.tensor(label)
        df = [0,0,0,0] # Fake annotation to maintain compatibility with the original dataset version (image+label+annotations)
        return rotated_img, label, df

    def __len__(self):
        return len(self.samples)

    def loader(self, path):
        return Image.open(path).convert('RGB')

class CustomDatasetColorization(Dataset):
    def __init__(self, root_dir, selected_classes, transform=None):
        self.root_dir = root_dir
        self.selected_classes = selected_classes
        self.transform = transform
        self.dataset = ImageFolder(root=root_dir, transform=transform)       
        # Filter out samples not in selected classes
        self.samples = [(path, label) for path, label in self.dataset.samples if self.dataset.classes[label] in self.selected_classes]

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img_orig = self.transform(img)  # apply transforms
            img_orig = np.asarray(img_orig)  # convert to numpy array
            img_lab = rgb2lab(img_orig)  # convert RGB image to LAB
            img_ab = img_lab[:, :, 1:3]  # separate AB channels from LAB
            img_ab = (img_ab + 128) / 255  # normalize the pixel values
            # transpose image from HxWxC to CxHxW and turn it into a tensor
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
            img_l = rgb2gray(img_orig)  # convert RGB to grayscale
            # add a channel axis to grascale image and turn it into a tensor
            img_l = torch.from_numpy(img_l).unsqueeze(0).float()
        # Load annotations
        return img_l, img_ab, label

    def __len__(self):
        return len(self.samples)

    def loader(self, path):
        return Image.open(path).convert('RGB')

class CustomDatasetSimCLR(Dataset):
    def __init__(self, root_dir, selected_classes, transform=None):
        self.root_dir = root_dir
        self.selected_classes = selected_classes
        self.transform = transform
        self.dataset = ImageFolder(root=root_dir, transform=transform)       
        # Filter out samples not in selected classes
        self.samples = [(path, label) for path, label in self.dataset.samples if self.dataset.classes[label] in self.selected_classes]
        # Create views
        if transform is not None:
            self.transformations = SimCLRTransformations(self.transform, n_views=2)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transformations(img)  # apply transforms
        # Load annotations
        return img[0], img[1], label

    def __len__(self):
        return len(self.samples)

    def loader(self, path):
        return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y, d = self.subset[index] # Image, label and annotations (x1, y1, x2, y2)
        if self.transform:
            x = self.transform(x)
        return x, y, d
        
    def __len__(self):
        return len(self.subset)

def read_annotations(file_path, set_type):
    '''
    It returns the annotations (bounding boxes) for the input file as a data frame
    Inputs
    ------
    file_path: string
        Path of the file
    set_type: string
        It represents the train, val or test set
    '''
    if set_type not in ['train', 'val', 'test']:
        raise ValueError('Set_type not recognized.')

    # Get class from file path
    class_ = file_path.split('/')[-3]
    file_name = file_path.split('/')[-1]

    df = pd.read_csv(f'./datasets/tiny-imagenet-200/{set_type}/{class_}/{class_}_boxes.txt', \
        sep='\t', header=None, names=['Filename', 'x1', 'y1', 'x2', 'y2'])
    # Set the 'Filename' column as the index
    df.set_index('Filename', inplace=True)

    return list(df.loc[file_name])

def get_statistics(dataset):
    # Return statistics (number of elements per class for the input dataset)
    data = []
    for i in range(len(dataset)):
        data.append(dataset.__getitem__(i)[1])        
    unique, counts = np.unique(data, return_counts=True)
    return unique, counts

def remove_elements(first_list, second_list):
    # Remove elements from the second list that are present in the first list
    result_list = [name for name in second_list if name not in first_list]
    return result_list

def compute_similarity(word1, word2):
    # Get the synsets for each word
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    
    # Compute the similarity between synsets
    max_similarity = 0
    for synset1, synset2 in product(synsets1, synsets2):
        similarity = synset1.wup_similarity(synset2)
        if similarity is not None and similarity > max_similarity:
            max_similarity = similarity

    return max_similarity

def find_similarity_pairs(list1, list2):
    similarity_pairs = []

    for name1, name2 in product(list1, list2):
        similarity = compute_similarity(name1, name2)
        similarity_pairs.append(((name1, name2), similarity))

    return similarity_pairs