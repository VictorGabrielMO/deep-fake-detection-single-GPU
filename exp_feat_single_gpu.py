import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
#from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import numpy as np
from PIL import Image
import os
#import clip
from tqdm import tqdm
#import timm
import argparse
import random
from sklearn.metrics import accuracy_score, precision_score
import json
import torchvision.models as vis_models
from pathlib import Path

from dataset import *
#from augment import ImageAugmentor
from mask import *
#from utils import *
from extract_single_gpu import *

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
    #parser.add_argument("--local_rank", type=int)
    #parser.add_argument("--local-rank","--local_rank", type=int)
    #parser.add_argument("--local-rank", "--local_rank")
    parser.add_argument('--checkpoint_path', type=Path, help='Path for the weights')
    parser.add_argument('--save_dir', type=Path, help='Save dir for the results')

    args = parser.parse_args()


    # Set the device to CUDA
    #device = torch.device("cuda")
    #print(f"Using device: {device}")
    
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #local_rank = int(os.environ["LOCAL_RANK"])
    #print("aqui")
    #print(local_rank)

    
    #device = torch.device(f'cuda:{local_rank}')
    #torch.cuda.set_device(local_rank)
    if not torch.cuda.is_available():
        raise Exception("CUDA not available")
    
    device = torch.device("cuda")
    
    #dist.init_process_group(backend='nccl')

    model_name = args.model_name.lower()
    finetune = 'ft' if args.pretrained else ''
    band = '' if args.band == 'all' else args.band
    
    ratio = args.ratio

    #CHANGED
    checkpoint_path = "/storage/datasets/gabriela.barreto/rn50_modft_spectralmask.pth"

    # Define the path to the results file
    results_dir = args.save_dir
    #os.makedirs(results_dir,exist_ok=True)
    #fake_results_dir = os.path.join(results_dir, "/fake")
    #os.makedirs(fake_results_dir, exist_ok=True)
    #real_results_dir = os.path.join(results_dir, "/real")
    #os.makedirs(real_results_dir, exist_ok=True)

    # Pretty print the arguments
    print("\nSelected Configuration:")
    print("-" * 30)
    print(f"Dataset Type: {args.data_type}")
    print(f"Model type: {args.model_name}")
    print(f"Ratio of mask: {ratio}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Mask Type: {args.mask_type}")
    print(f"Checkpoint Type: {checkpoint_path}")
    print(f"Results saved to: {results_dir}")
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


    
    
    all_fake_features = np.empty((0, 2048))
    all_fake_preds = np.empty((0, 1))
    
    all_real_features = np.empty((0, 2048))
    all_real_preds = np.empty((0, 1))
    for dataset_name, dataset_path in real_datasets.items():
        
        prediction_real, features_real, _ = extract_evaluation_features(
            args.model_name,
            args.data_type,
            args.mask_type,
            ratio/100,
            dataset_path,
            dataset_name,
            args.batch_size,
            checkpoint_path,
            device,
            args,
            label=0
        )

        #np.savez(os.path.join(results_dir,f"real_{dataset_name}.npz"), y_hat=prediction_real ,feat=features_real )
        all_real_features = np.concatenate((all_real_features, features_real), axis=0)
        all_real_preds = np.concatenate((all_real_preds, prediction_real), axis=0)
        
    for dataset_name, dataset_path in fake_datasets.items():
        prediction_fake, features_fake, _ = extract_evaluation_features(
            args.model_name,
            args.data_type,
            args.mask_type,
            ratio/100,
            dataset_path,
            dataset_name,
            args.batch_size,
            checkpoint_path,
            device,
            args,
            label=0
        )

        #np.savez(os.path.join(results_dir,f"fake_{dataset_name}.npz"), y_hat = prediction_fake, feat=features_fake )
        all_fake_features = np.concatenate((all_fake_features, features_fake), axis=0)
        all_fake_preds = np.concatenate((all_fake_preds, prediction_fake), axis=0)
    
    np.savez( os.path.join(results_dir,"experiment_results.npz"), fake_feats= all_fake_features, fake_preds = all_fake_preds,
             real_feats = all_real_features, real_preds = all_real_preds)
    