from torchvision import transforms as T

# ImageNet, normalize with mean=[0.485, 0.456, 0.406], 
# std=[0.229, 0.224, 0.225])
def train_transformations(means, stds):
    preprocess_transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((256, 256), antialias=True),
                    T.RandomCrop(224), # Center crop image
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomRotation(15),
                    T.ColorJitter(brightness=.5, hue=.3),
                    T.ToTensor(),  # Converting cropped images to tensors
                    T.Normalize(mean=means.tolist(), 
                                std=stds.tolist()) 
    ])
    return preprocess_transform

def val_transformations(means, stds):

    preprocess_transform_val = T.Compose([
                    T.ToPILImage(),
                    T.Resize((224, 224), antialias=True), # Resize images
                    T.ToTensor(),  # Converting cropped images to tensors
                    T.Normalize(mean=means.tolist(), 
                                std=stds.tolist()) 
    ])
    return preprocess_transform_val

def test_transformations(means, stds):
    preprocess_transform_test = T.Compose([
                    T.ToTensor(),  # Converting cropped images to tensors
                    T.Resize((224, 224), antialias=True), # Resize images
                    T.Normalize(mean=means.tolist(), 
                                std=stds.tolist()) 
    ]) 
    return preprocess_transform_test

def colorization_transformations():
    transform = T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip()
    ])
    return transform

class SimCLRTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

def simclr_transformations():

    contrast_transforms = T.Compose([T.RandomHorizontalFlip(),
                                            T.Resize(256),
                                            T.RandomResizedCrop(size=224),
                                            T.RandomApply([
                                                T.ColorJitter(brightness=0.5,
                                                                        contrast=0.5,
                                                                        saturation=0.5,
                                                                        hue=0.1)
                                            ], p=0.8),
                                            T.RandomGrayscale(p=0.2),
                                            T.GaussianBlur(kernel_size=9),
                                            T.ToTensor(),
                                            T.Normalize((0.5,), (0.5,))
                                            ])

    return contrast_transforms