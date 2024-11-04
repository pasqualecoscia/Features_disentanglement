import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config_parser():
    parser = argparse.ArgumentParser(description='Configuration parameters')

    # Model parameters
    # resnet18
    parser.add_argument('--model', type=str, default='resnet18', choices=['vgg19', 'resnet18', 'resnet50', 'resnet101'], help='Model to be selected.')
    # from_scratch, fine-tuning
    parser.add_argument('--training_type', type=str, default='from_scratch', choices=['from_scratch', 
                    'imagenet_pretrained', 'rotation', 'rotation_finetuning', 'colorization', 'colorization_finetuning', 'simclr', 'simclr_finetuning'], help='Training type')
    parser.add_argument('--degrees', nargs='+', type=int, default=[0, 90, 180, 270], help='A list of degrees')
    # num_classes for tiny image net
    parser.add_argument('--num_classes', type=int, default=5, help='Number of selected classes from TinyImageNet.')
    parser.add_argument('--lr_finder', type=str2bool, default=False, help='Find best learning rate.')
    parser.add_argument('--data_percentage', type=int, default=100, help='Percentage of training data to use.')
    parser.add_argument('--save_plots', type=str2bool, default=True, help='Save figures.')
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--optim', type=str, default="Adam", choices=['Adam', 'SGD'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='OneCycleLR', help='Type of scheduler to be used.')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay rate')
    parser.add_argument('--results_path', type=str, default="./results", help='Path for results')
    # SimCLR hyperparameters
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature InfoNCELoss')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension network')
    # Optimization parameters
    parser.add_argument('--use_cuda', type=str2bool, default=True, help='Use cuda')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--pin_memory', type=str2bool, default=True, help='Pin memory')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='tiny-image-net', choices=['tiny-image-net'], help='dataset')
    parser.add_argument('--np_random_seed', type=int, default=12, help='NumPy random seed')
    parser.add_argument('--random_state_split', type=int, default=123, help='Random state for data split')
    # XAI parameters
    parser.add_argument('--cam_layers', nargs='+', default=["layer4"], help='List of CAM layers') # or features for VGG16
    parser.add_argument('--cam_type', type=str, default="GradCAM", help='Type of CAM')
    parser.add_argument('--segmentation_threshold', type=float, default=0.5, help='Segmentation threshold')
    parser.add_argument('--image_size_dataset', type=int, default=64, help='Image size for the dataset')
    parser.add_argument('--XAI_path', type=str, default="XAI", help='Path for XAI')
    # Laber or toppred for extracting the CAM
    parser.add_argument('--label_toppred_evaluation', type=str, default="label", help='Label for top prediction evaluation')
    parser.add_argument('--XAI_single_plot', type=str, default=True, help='Plot single images or combined.')
    # Fine-tuning parameters
    parser.add_argument('--closest_classes', type=str2bool, default=False, help='Closest classes')
    parser.add_argument('--num_closest_classes', type=int, default=5, help='Number of closest classes')
    # Disentanglement loss
    parser.add_argument('--dis_loss', type=str2bool, default=False, help='Disentangle loss')
    parser.add_argument('--lambda_dls', type=float, default=0.5, help='Disentangle loss weight')
    parser.add_argument('--target_layer_name', type=str, default='avgpool', help='Target layer for extracting features (the layer before it is considered)')
    
    return parser.parse_args()
