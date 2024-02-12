import argparse
import importlib
import logging
from models.model import load_pretrained_model
from data.dataset import  TemporalDataset, SpatialDataset
from utils import save_sample_images, evaluate_performance
from train import train_model
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

def main(config_module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print(config_module)

    # data type(au, expr, va)에 따라서 ground truth 파일 경로 얻기

    if config_module.data_name == 'au':
        data_path = 'data/AU_Detection_Challenge'        
    elif config_module.data_name == 'expr':
        data_path = 'data/EXPR_Detection_Challenge'
    elif config_module.data_name == 'va':
        data_path = 'data/VA_Detection_Challenge'

    if config_module.data_type == 'spatial':
    
        dataset_train = SpatialDataset(data_path, config_module.data_name, 'train')
        dataset_val = SpatialDataset(data_path, config_module.data_name, 'val')

    dataloader_train = DataLoader(dataset_train, batch_size=config_module.batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=config_module.batch_size, shuffle=False)

    for epoch in range(config_module.epochs):
        for inputs, labels in dataloader_train:
            #inputs, labels = inputs.to(device), labels.to(device)
            print("inputs :",  inputs,"labels : ", labels)
            performance = evaluate_performance(labels, labels, config_module.data_name)
            print(f"Performance: {performance}")
            break
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Learning Model Configuration')
    parser.add_argument('--config', type=str, required=True, help='Config module name to use')
    #parser.add_argument('--model_name', type=str, required=True, help='Model module name to use')
    parser.add_argument('--data_type', type=str, required=True, help='Model module name to use')
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    #config_module.model_name = args.model_name
    config_module.data_type = args.data_type
    
    main(config_module)

