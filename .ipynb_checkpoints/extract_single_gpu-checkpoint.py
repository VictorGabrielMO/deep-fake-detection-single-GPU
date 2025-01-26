import torch
import os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
#from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
#import torch.distributed as dist  # If distributed training is being used
#import wandb  # If Weights & Biases is being used for logging

from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
#import timm 

import torchvision.models as vis_models

from dataset import *
#from augment import ImageAugmentor
from mask import *
#from utils import *
from networks.resnet import resnet50
from networks.resnet_mod import resnet50 as _resnet50, ChannelLinear

#from networks.clip_models import CLIPModel
import time

#from torch.nn.parallel import DistributedDataParallel
#from torch.utils.data.distributed import DistributedSampler

#os.environ['NCCL_BLOCKING_WAIT'] = '1'
#os.environ['NCCL_DEBUG'] = 'WARN'

def extract_evaluation_features(
    model_name,
    data_type,
    mask_type, 
    ratio,
    dataset_path,
    dataset_name,
    batch_size,
    checkpoint_path, 
    device,
    args,
    label=0
    ):

    # Depending on the mask_type, create the appropriate mask generator
    if mask_type == 'spectral':
        mask_generator = FrequencyMaskGenerator(ratio=ratio)
    else:
        mask_generator = None


    #test_opt = {
    #    'rz_interp': ['bilinear'],
    #    'loadSize': 256,
    #    'blur_prob': 0.1,  # Set your value
    #    'blur_sig': [(0.0 + 3.0) / 2],
    #    'jpg_prob': 0.1,  # Set your value
    #    'jpg_method': ['pil'],
    #    'jpg_qual': [int((30 + 100) / 2)]
    #}

    #test_transform = test_augment(ImageAugmentor(test_opt), mask_generator, args)
  

    if data_type == 'ArtiFact':
        test_dataset = ArtiFact(dataset_path, label, load_percentage=5)
    else:
        raise ValueError("wrong dataset input")

    #test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    if model_name == 'RN50_mod':
        model = _resnet50(pretrained=False, stride0=1)
        model.fc = ChannelLinear(model.fc.in_features, 1)
    else:
        raise ValueError(f"Model {model_name} not recognized!")

    model = model.to(device)
    #model = DistributedDataParallel(model, find_unused_parameters=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')['model_state_dict']

    # Check if model was trained with DataParallel or DistributedDataParallel
    if 'module.' in list(checkpoint.keys())[0]:  # Check if the weights are wrapped in 'module.'
        # Create a new dictionary without the 'module.' prefix
        new_checkpoint = {}
        for k, v in checkpoint.items():
            new_checkpoint[k.replace('module.', '')] = v
        checkpoint = new_checkpoint

    # Load the weights into the model
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval() 
    
    features = np.empty((0, 2048))
    pred_list = []
    labels_list = []
    

    #disable_tqdm = dist.get_rank() != 0
    data_loader_with_tqdm = tqdm(test_dataloader, "test dataloading")

    print(f"### Iniciando extraçao de features no dataset {dataset_name} ...")
    with torch.no_grad():
        for inputs, labels in data_loader_with_tqdm:
            inputs = inputs.to(device)
            y_pred, embeddings = model(inputs, return_feats=True)
            embeddings = embeddings.detach().cpu()
            embeddings = embeddings.mean(dim=(2,3)) # GAP
            y_pred = y_pred.view(-1).unsqueeze(1)
            features = np.concatenate((features, embeddings.numpy()), axis=0)
            pred_list.extend(y_pred.sigmoid().detach().cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    #features = torch.cat(features)
    labels_list, pred_list = np.array(labels_list), np.array(pred_list)
    pred_list = np.where(pred_list < 0.5, 0, 1)
    
    labels_list[labels_list > 1] = 1
    
    #labels_list = torch.cat(labels_list).numpy()

    return pred_list, features, labels_list