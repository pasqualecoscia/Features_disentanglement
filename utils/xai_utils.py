import torch
from torchcam.methods import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM, ISCAM, XGradCAM, LayerCAM
import os
import csv
import matplotlib.patches as patches
from torchvision import transforms

def jaccard_index(pred_mask, gt_mask):
    '''
        This function computes the Jaccard Index (or IntersectionOverUnion - IoU)
        
                    TP
        J =  ---------------
               TP + FP + FN

        Parameters
        ----------
        pred_mask: torch.Tensor
            Input predicted mask
        gt_mask: torch.Tensor
            Input ground-truth mask            
        Returns
        -------
        iou: float
            IntersectionOverUnion

    '''
    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask)>0  # Logical OR
    iou = overlap.sum() / float(union.sum())

    return iou

def precision_score(pred_mask, gt_mask):
    '''
        This function computes the precision score
            
                 TP
        P =  -----------
               TP + FP

        Parameters
        ----------
        pred_mask: torch.Tensor
            Input predicted mask
        gt_mask: torch.Tensor
            Input ground-truth mask            
        Returns
        -------
        precision_score: float
            precision score

    '''
    overlap = pred_mask * gt_mask  # Logical AND
    total_pixel_pred = pred_mask.sum()
    precision = overlap.sum() / total_pixel_pred

    return precision 

def recall_score(pred_mask, gt_mask):
    '''
        This function computes the recall score
                
                 TP
        R =  ----------
               TP + FN

        Parameters
        ----------
        pred_mask: torch.Tensor
            Input predicted mask
        gt_mask: torch.Tensor
            Input ground-truth mask            
        Returns
        -------
        recall_score: float
            recall score

    '''
    overlap = pred_mask * gt_mask  # Logical AND
    total_pixel_gt = gt_mask.sum()
    recall = overlap.sum() / total_pixel_gt

    return recall

def accuracy(pred_mask, gt_mask):
    '''
        This function computes the accuracy
        
                    TP + TN
        A =  --------------------
               TP + TN + FN + FP
        
        Parameters
        ----------
        pred_mask: torch.Tensor
            Input predicted mask
        gt_mask: torch.Tensor
            Input ground-truth mask            
        Returns
        -------
        accuracy: float
            accuracy

    '''
    tot = (gt_mask==pred_mask)
    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask)>0  # Logical OR
    acc = tot.sum() / (tot.sum() + union.sum() - overlap.sum())

    return acc

def dice_coef(pred_mask, gt_mask):
    '''
        This function computes the dice coefficient
        
                    2*TP
        D =  ------------------
               2*TP + FP + FN
        
        Parameters
        ----------
        pred_mask: torch.Tensor
            Input predicted mask
        gt_mask: torch.Tensor
            Input ground-truth mask            
        Returns
        -------
        dice_c: float
            Dice coefficient

    '''
    overlap = pred_mask * gt_mask  # Logical AND
    dice_c = 2 * overlap.sum() / (pred_mask.sum() + gt_mask.sum()) 

    return dice_c

def create_patch(df, linewidth=7, edgecolor='r', facecolor='None'):
    # Create a Rectangle patch
    
    check_tensors = all(isinstance(elem, torch.Tensor) for elem in df)
    if (check_tensors):
        df = [elem.item() for elem in df]

    rect = patches.Rectangle((df[0], df[1]), df[2] - df[0], df[3] - df[1],
                                linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)
    return rect

def fill_rectangle(image_size, x1, y1, x2, y2, device):
    """
    Create the segmentation mask from x1, y1 and x2, y2
    
    Parameters:
    - image_size: int
        Square image to create
    - x1, y1, x2, y2: int
        The coordinates of the rectangle's two extremes.
    """
    image = torch.zeros((image_size, image_size))
    image[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1] = 1
    return image.to(device)

def extractCAMs(model, layers, cam_type):
    '''
        Initialize CAM extractor from model 
        Input image should be normalized
        Input model in evaluation mode
    '''

    if cam_type == "CAM":
        cam_extractor = CAM(model)
    elif cam_type == "GradCAM":
        cam_extractor = GradCAM(model, layers)
    elif cam_type == "GradCAMpp":
        cam_extractor = GradCAMpp(model, layers)
    elif cam_type == "SmoothGradCAMpp":
        cam_extractor = SmoothGradCAMpp(model, layers)
    elif cam_type == "ScoreCAM":
        cam_extractor = ScoreCAM(model)
    elif cam_type == "SSCAM":
        cam_extractor = SSCAM(model)
    elif cam_type == "ISCAM":
        cam_extractor = ISCAM(model)
    elif cam_type == "XGradCAM":
        cam_extractor = XGradCAM(model, layers)  
    elif cam_type == "LayerCAM":
        cam_extractor = LayerCAM(model, layers)  
    else:
        raise ValueError('cam_type not recognized.')
    
    return cam_extractor

def thresholding(map_, thr):
    '''
        This function applies a thresholding operation to transform the input heatmap
        to a binary segmentation mask
        
        Parameters
        ----------
        heatmap: torch.Tensor
            Input heatmap
        thr: float
            threshold value
        Returns
        -------
        segmentation_mask: torch.Tensor
            Segmentation mask

    '''
    # Check input thr value
    if not (0 <= thr <= 1):
        raise ValueError("Invalid thresholding value. Input must be between 0 and 1.")

    map_ /= map_.max()
    return (map_ > thr).to(map_.dtype)

def save_xai_metrics_to_csv(folder_path, xai_metrics):

    # Create a filename based on the model name
    filename = os.path.join(folder_path, f'XAI_metrics.csv')

    # Write XAI metrics to CSV
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Metric', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write each metric and its value to the CSV file
        for metric, value in xai_metrics.items():
            writer.writerow({'Metric': metric, 'Value': value})

def denormalization_and_resizing(image, means, stds, size_img=64):
    '''
        Denormalize and resize input image using input means and stds
    '''

    denormalization = transforms.Normalize((-means / stds).tolist(), (1.0 / stds).tolist())
    resize = transforms.Resize(size_img)
    return resize(denormalization(image))