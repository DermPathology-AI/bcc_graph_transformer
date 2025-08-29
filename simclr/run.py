from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import os, glob
import pandas as pd
import argparse
import torch

def generate_csv(args):
    if args.WSI_address:
        WSI_address = args.WSI_address
    else:
        wsi_address = '..'
    
    if args.level=='high' and args.multiscale==1:
        path_temp = os.path.join(wsi_address, 'WSI', args.dataset, 'pyramid', '*', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp) 
        print(path_temp)
    if args.level=='low' and args.multiscale==1:
        path_temp = os.path.join(wsi_address, 'WSI', args.dataset, 'pyramid', '*', '*', '*.jpeg')
        print(path_temp)
        patch_path = glob.glob(path_temp) 
    if args.multiscale==0:
        path_temp = os.path.join(wsi_address, 'WSI', args.dataset, 'single', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp)  

    df = pd.DataFrame(patch_path)
    df.to_csv('all_patches.csv', index=False)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='low', help='Magnification level to compute embedder (low/high)')
    parser.add_argument('--multiscale', type=int, default=0, help='Whether the patches are cropped from multiscale (0/1-no/yes)')
    parser.add_argument('--dataset', type=str, default='BCC', help='Dataset folder name')
    parser.add_argument('--WSI_address', type=str, default='..', help='path the dataset is located')
    
    args = parser.parse_args()
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    gpu_ids = eval(config['gpu_ids'])
    print([gpu_ids])
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)   
    print('GPU devices:', torch.cuda.device_count())
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])   
    generate_csv(args)
    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
