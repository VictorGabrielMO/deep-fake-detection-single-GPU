import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import numpy as np
from PIL import Image
import os
import clip
from tqdm import tqdm
import timm
import argparse
import random
from sklearn.metrics import accuracy_score, precision_score
import json
import torchvision.models as vis_models

from dataset import *
#from augment import ImageAugmentor
from mask import *
from utils import *

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for your script")

    parser.add_argument(
        '--model_name',
        default='RN50',
        type=str,
        choices=[
            'RN18', 'RN34', 'RN50', 'RN50_mod', 'clip_rn50', 'clip_vitl14',
        ],
        help='Type of model to use; includes ResNet variants'
        )
    parser.add_argument(
        '--clip_ft', 
        action='store_true', 
        help='For loading a finetuned clip model'
        )
    parser.add_argument(
        '--mask_type', 
        default='spectral', 
        choices=[
            'patch', 
            'spectral',
            'pixel', 
            'nomask'], 
        help='Type of mask generator'
        )
    parser.add_argument(
        '--band', 
        default='all',
        type=str,
        choices=[
            'all', 'low', 'mid', 'high',]
        )
    parser.add_argument(
        '--pretrained', 
        action='store_true', 
        help='For pretraining'
        )
    parser.add_argument(
        '--ratio', 
        type=int, 
        default=50,
        help='Ratio of mask to apply'
        )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64, 
        help='Batch Size'
        )
    parser.add_argument(
        '--data_type', 
        default="Wang_CVPR20", 
        type=str, 
        choices=['Wang_CVPR20', 'Ojha_CVPR23','ArtiFact'], 
        help="Dataset Type"
        )
    parser.add_argument(
        '--other_model', 
        action='store_true', 
        help='if the model is from my own code'
        )
    parser.add_argument('--local_rank', type=int, help='Local rank for distributed training')

    args = parser.parse_args()


    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f'cuda:{args.local_rank}')
    torch.cuda.set_device(device)

    dist.init_process_group(backend='nccl')

    model_name = args.model_name.lower()
    finetune = 'ft' if args.pretrained else ''
    band = '' if args.band == 'all' else args.band

    if args.mask_type != 'nomask':
        ratio = args.ratio
        checkpoint_path = f'checkpoints/mask_{ratio}/{model_name}{finetune}_{band}{args.mask_type}mask.pth'
        #checkpoint_path = f"checkpoints/mask_{ratio}/rn50ft_spectralmask(0.5).pth"
    else:
        ratio = 0
        checkpoint_path = f'checkpoints/mask_{ratio}/{model_name}{finetune}.pth'

    # Define the path to the results file
    results_path = f'results/{args.data_type.lower()}'
    os.makedirs(results_path, exist_ok=True)
    filename = f'{model_name}{finetune}_{band}{args.mask_type}mask{ratio}.txt'

    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Device: {args.local_rank}")
    print(f"Dataset Type: {args.data_type}")
    print(f"Model type: {args.model_name}")
    print(f"Ratio of mask: {ratio}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Mask Type: {args.mask_type}")
    print(f"Checkpoint Type: {checkpoint_path}")
    print(f"Results saved to: {results_path}/{filename}")
    print("-" * 30, "\n")

    if args.data_type == 'Wang_CVPR20':
        datasets = {
            'ProGAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/progan',
            'CycleGAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/cyclegan',
            'BigGAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/biggan',
            'StyleGAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/stylegan',
            'GauGAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/gaugan',
            'StarGAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/stargan',
            'DeepFake': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/deepfake',
            'SITD': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/seeingdark',
            'SAN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/san',
            'CRN': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/crn',
            'IMLE': '/home/users/chandler_doloriel/scratch/Datasets/Wang_CVPR2020/testing/imle',
        }
    # elif args.data_type == 'GenImage':
    #     datasets = {
    #         'VQDM': '/home/users/chandler_doloriel/scratch/Datasets/GenImage/imagenet_vqdm/imagenet_vqdm/val',
    #         'Glide': '/home/users/chandler_doloriel/scratch/Datasets/GenImage/imagenet_glide/imagenet_glide/val',
    #     }
    elif args.data_type == 'Ojha_CVPR23':
        datasets = {
            'Guided': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/guided',
            'LDM_200': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/ldm_200',
            'LDM_200_cfg': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/ldm_200_cfg',
            'LDM_100': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/ldm_100',
            'Glide_100_27': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/glide_100_27',
            'Glide_50_27': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/glide_50_27',
            'Glide_100_10': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/glide_100_10',
            'DALL-E': '/home/users/chandler_doloriel/scratch/Datasets/Ojha_CVPR2023/dalle',       
        }
    elif args.data_type == 'ArtiFact':
        real_datasets = {
            "afhq": "/storage/datasets/gabriela.barreto/artifact/afhq",
            "celebAHQ": "/storage/datasets/gabriela.barreto/artifact/celebahq",
            "coco": "/storage/datasets/gabriela.barreto/artifact/coco",
            "ffhq": "/storage/datasets/gabriela.barreto/artifact/ffhq",
            "imagenet": "/storage/datasets/gabriela.barreto/artifact/imagenet",
            "landscape": "/storage/datasets/gabriela.barreto/artifact/landscape",
            "lsun": "/storage/datasets/gabriela.barreto/artifact/lsun",
            "metfaces": "/storage/datasets/gabriela.barreto/artifact/metfaces",
            "cycle_gan": "/storage/datasets/gabriela.barreto/artifact/cycle_gan"
        }



            
        fake_datasets = {
            "big_gan": "/storage/datasets/gabriela.barreto/artifact/big_gan",
            "cips": "/storage/datasets/gabriela.barreto/artifact/cips",
            "ddpm": "/storage/datasets/gabriela.barreto/artifact/ddpm",
            "denoising_diffusion_gan": "/storage/datasets/gabriela.barreto/artifact/denoising_diffusion_gan",
            "diffusion_gan": "/storage/datasets/gabriela.barreto/artifact/diffusion_gan",
            "face_synthetics": "/storage/datasets/gabriela.barreto/artifact/face_synthetics",
            "gansformer": "/storage/datasets/gabriela.barreto/artifact/gansformer",
            "gau_gan": "/storage/datasets/gabriela.barreto/artifact/gau_gan",
            "generative_inpainting": "/storage/datasets/gabriela.barreto/artifact/generative_inpainting",
            "glide": "/storage/datasets/gabriela.barreto/artifact/glide",
            "lama": "/storage/datasets/gabriela.barreto/artifact/lama",
            "latent_diffusion": "/storage/datasets/gabriela.barreto/artifact/latent_diffusion",
            "mat": "/storage/datasets/gabriela.barreto/artifact/mat",
            "palette": "/storage/datasets/gabriela.barreto/artifact/palette",
            "pro_gan": "/storage/datasets/gabriela.barreto/artifact/pro_gan",
            "projected_gan": "/storage/datasets/gabriela.barreto/artifact/projected_gan",
            "sfhq": "/storage/datasets/gabriela.barreto/artifact/sfhq",
            "stable_diffusion": "/storage/datasets/gabriela.barreto/artifact/stable_diffusion",
            "star_gan": "/storage/datasets/gabriela.barreto/artifact/star_gan",
            "stylegan1": "/storage/datasets/gabriela.barreto/artifact/stylegan1",
            "stylegan2": "/storage/datasets/gabriela.barreto/artifact/stylegan2",
            "stylegan3": "/storage/datasets/gabriela.barreto/artifact/stylegan3",
            "taming_transformer": "/storage/datasets/gabriela.barreto/artifact/taming_transformer",
            "vq_diffusion": "/storage/datasets/gabriela.barreto/artifact/vq_diffusion"
        }

    else:
        raise ValueError("wrong dataset type")

    datasets_metrics = {}
    all_y_true = []
    all_y_pred = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for dataset_name, dataset_path in real_datasets.items():
        if dist.get_rank() == 0:
            print(f"\nEvaluating {dataset_name}")

        y_true, y_pred = evaluate_model(
            args.model_name,
            args.data_type,
            args.mask_type,
            ratio/100,
            dataset_path,
            args.batch_size,
            checkpoint_path,
            device,
            args,
            label=0
        )
        y_pred = np.where(y_pred < 0.5, 0, 1)
        
        all_y_true = np.append(all_y_true, y_true)
        all_y_pred = np.append(all_y_pred, y_pred)

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        samples = len(y_pred)
        tn_ = np.int(np.sum(y_pred < 0.5))
        fp_ = np.int(np.sum(y_pred >= 0.5))
        metrics = {
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'y_true': samples,
            'tn': tn_,
            'fp': fp_
        }

        tn += tn_
        fp += fp_
        datasets_metrics[dataset_name] = metrics
        

    for dataset_name, dataset_path in fake_datasets.items():
        if dist.get_rank() == 0:
            print(f"\nEvaluating {dataset_name}")

        y_true, y_pred = evaluate_model(
            args.model_name,
            args.data_type,
            args.mask_type,
            ratio/100,
            dataset_path,
            args.batch_size,
            checkpoint_path,
            device,
            args,
            label=1
        )
        y_pred = np.where(y_pred < 0.5, 0, 1)
        
        all_y_true = np.append(all_y_true, y_true)
        all_y_pred = np.append(all_y_pred, y_pred)

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        samples = len(y_pred)
        tp_ = np.int(np.sum(y_pred >= 0.5))
        fn_ = np.int(np.sum(y_pred < 0.5))
        metrics = {
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'y_true': samples,
            'tp': tp_,
            'fn': fn_
        }
        tp += tp_
        fn += fn_
        datasets_metrics[dataset_name] = metrics
         
    acc = accuracy_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred)
    recall = recall_score(all_y_true, all_y_pred)
    
    metrics = {
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    datasets_metrics['all_datasets'] = metrics

    with open('results/metrics.json','w') as file:
        json.dump(datasets_metrics, file, indent=4)