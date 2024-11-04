# Features Disentanglement - PyTorch

### Overview
This repository contains the PyTorch implementation of Features Disentanglement for Explainable Convolutional Neural Networks (ICIP24).

### Usage Example

To run the main training script, use the following command:

```bash
python3 main.py --model resnet101 --training_type simclr_finetuning --closest_classes False --data_percentage 100 --num_classes 5 --epochs 250 --num_closest_classes 5 --batch_size 64
```
To run the explainability evaluation script, use the following command:

```bash
python3 xai_eval.py --model resnet18 --training_type from_scratch --closest_classes False --data_percentage 100 --num_classes 5 --epochs 250 --num_closest_classes 5 --cam_type LayerCAM
```
For more information about input parameters, refer to the utils/config_params.py file.

> **Citation:**  
> If you use this code for your research, please cite our ICIP24 paper:
> 
> ```bibtex
> @INPROCEEDINGS{10647568,
>   author={Coscia, Pasquale and Genovese, Angelo and Scotti, Fabio and Piuri, Vincenzo},
>   booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
>   title={Features Disentanglement For Explainable Convolutional Neural Networks}, 
>   year={2024},
>   volume={},
>   number={},
>   pages={514-520},
>   keywords={Measurement;Visualization;Image recognition;Semantic segmentation;Decision making;Convolutional neural networks;Security;Explainable AI (XAI);ResNet;self-supervised learning (SSL);disentanglement},
>   doi={10.1109/ICIP51287.2024.10647568}}
> ```
