import argparse
import importlib
from data.dataset import  TemporalDataset, SpatialDataset, MultimodalDataset, SequenceData
from utils import save_sample_images, evaluate_performance, fix_seed, update_config
from train import train_model, train_function, evaluate_function
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.model import load_model
from extract import *
import logging

def main(config_module):
    device = torch.device(f"cuda:{config_module.device}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = load_model(config_module)
    model.to(device)
    if config_module.mode == 'train':
        # data type(au, expr, va)에 따라서 ground truth 파일 경로 얻기
        feat_path = "../dataset/feature"
        if config_module.data_name == 'au':
            data_path = '../dataset/6th ABAW Annotations/AU_Detection_Challenge'        
        elif config_module.data_name == 'expr':
            data_path = '../dataset/6th ABAW Annotations/EXPR_Recognition_Challenge'
        elif config_module.data_name == 'va':
            data_path = '../dataset/6th ABAW Annotations/VA_Estimation_Challenge'

        if config_module.data_type == 'spatial':
        
            dataset_train = SpatialDataset(data_path, config_module.data_name, 'train')
            dataset_val = SpatialDataset(data_path, config_module.data_name, 'val')
        
        elif config_module.data_type == 'multimodal':
        
            #dataset_train = MultimodalDataset(data_path, config_module.data_name, 'train')
            #dataset_val = MultimodalDataset(data_path, config_module.data_name, 'val')
            
            dataset_train = SequenceData(feat_path, data_path, 100, config_module.data_name, 'train')
            dataset_val = SequenceData(feat_path, data_path, 100, config_module.data_name, 'val')
            
        dataloader_train = DataLoader(dataset_train, batch_size=config_module.batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=config_module.batch_size, shuffle=False)
        
        model = load_model(config_module)
        model.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=config_module.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=config_module.epochs)
        
        log_path = f"output/{config_module.model_name}_{config_module.data_name}_{config_module.data_type}"
        os.makedirs(log_path, exist_ok=True)

        logging.basicConfig(filename=f"{log_path}/training_log.log", level=logging.INFO)
    
        for epoch in range(config_module.epochs):
            model, train_loss = train_function(model, dataloader_train, criterion, optimizer, device, config_module.num_classes)
            
            print(f'Epoch {epoch+1}/{config_module.epochs}, Train Loss: {train_loss:.4f}')
            logging.info(f'Epoch {epoch}/{config_module.epochs - 1}, Train Loss: {train_loss:.4f}')
            
            scheduler.step()
            performance, val_loss = evaluate_function(model, dataloader_val, criterion, device, config_module.num_classes)
            
            print(f'Epoch {epoch+1}/{config_module.epochs}, Validation Loss: {val_loss:.4f}, Average F1-Score: {performance:.4f}')
            logging.info(f'Epoch {epoch}/{config_module.epochs - 1}, Validation Loss: {val_loss:.4f}, Average F1-Score: {performance:.4f}')
            
            torch.save(model.state_dict(), os.path.join(log_path, f"model_{epoch}.pth"))

    elif config_module.mode == 'inference':
        pass
    
    elif config_module.mode == 'extract':
        extract_feature(config_module.data_type, device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Learning Model Configuration')
    parser.add_argument('--config', type=str, required=True, help='Config module name to use')
    parser.add_argument('--data_type', type=str, required=True, help='Model module name to use')
    parser.add_argument('--mode', type=str, required=True, help='Mode')
    parser.add_argument('--device', type=int, required=True, help='Mode')
    args = parser.parse_args()

    config_module = update_config(args)
    fix_seed()
    main(config_module)

