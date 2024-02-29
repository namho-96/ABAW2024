import argparse
import importlib
from data.dataset import TemporalDataset, SpatialDataset, MultimodalDataset, SequenceData, SequenceData_2
from utils import save_sample_images, evaluate_performance, fix_seed, update_config, save_checkpoint, load_checkpoint, setup_log, log_and_checkpoint
from train import train_model, train_function, evaluate_function
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.model import load_model
from extract import *
import logging
import wandb
logging.getLogger().setLevel(logging.INFO)


def setup_dataset(config):
    feat_path = config.feat_path
    data_path = {
        'au': 'AU_Detection_Challenge',
        'expr': 'EXPR_Recognition_Challenge',
        'va': 'VA_Estimation_Challenge'
    }
    data_path = os.path.join(config.label_path, data_path.get(config.data_name, ''))

    if config.data_type == 'spatial':
        dataset_train = SpatialDataset(data_path, config.data_name, 'train')
        dataset_val = SpatialDataset(data_path, config.data_name, 'val')
    elif config.data_type == 'multimodal':
        dataset_train = SequenceData(feat_path, data_path, config.sq_len, config.data_name, 'train')
        dataset_val = SequenceData(feat_path, data_path, config.sq_len, config.data_name, 'val')
    else:
        raise ValueError(f'Wrong Data type : {config.data_type}')

    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, drop_last=True)

    return dataloader_train, dataloader_val


def setup_training(config):
    # Set device
    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    model = load_model(config)
    model.to(device)

    # Optimizer
    if config.optimizer == "adamw":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, betas=(0.9, 0.999), weight_decay=0.05)
    elif config.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, betas=(0.9, 0.999))
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # Loss function
    if config.data_name == 'va':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    return device, model, optimizer, scheduler, criterion


def main(config):
    # 학습 설정
    device, model, optimizer, scheduler, criterion = setup_training(config)

    # 이어서 학습할 경우 체크포인트 로드
    start_epoch = 0
    if config.resume:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, config)

    if config.mode == 'train':
        log_path = setup_log(config)    # wandb & log 설정

        dataloader_train, dataloader_val = setup_dataset(config)    # 데이터 로드

        best_performance = float('-inf')
        for epoch in range(start_epoch, config.epochs):
            # Train
            _, train_loss = train_function(model, dataloader_train, criterion, optimizer, device, config)
            # Validate
            performance, val_loss = evaluate_function(model, dataloader_val, criterion, device, config)
            # 로깅 및 체크포인트 저장
            best_performance = log_and_checkpoint(epoch, model, optimizer, train_loss, val_loss, performance, scheduler, log_path, best_performance, config)
            scheduler.step()

    elif config.mode == 'inference':
        pass
    elif config.mode == 'extract':
        extract_feature(config.data_type, device)
    else:
        assert ValueError(f"Unknown mode: {config.mode}")





def main_old(config_module):
    device = torch.device(f"cuda:{config_module.device}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = load_model(config_module)
    model.to(device)
    if config_module.mode == 'train':
        setup_log(config_module)   # Initialize wandb only in training mode
        # data type(au, expr, va)에 따라서 ground truth 파일 경로 얻기
        feat_path = config_module.feat_path     # "C:/Users/hms/Desktop/Code/0.Datasets/Aff-Wild2/2024"

        if config_module.data_name == 'au':
            data_path = '../dataset/6th ABAW Annotations/AU_Detection_Challenge'        
        elif config_module.data_name == 'expr':
            data_path = '../dataset/6th ABAW Annotations/EXPR_Recognition_Challenge'
        elif config_module.data_name == 'va':
            data_path = config_module.label_path    # 'C:/Users/hms/Desktop/Code/0.Datasets/Aff-Wild2/6th ABAW Annotations/VA_Estimation_Challenge'

        if config_module.data_type == 'spatial':
            dataset_train = SpatialDataset(data_path, config_module.data_name, 'train')
            dataset_val = SpatialDataset(data_path, config_module.data_name, 'val')
        elif config_module.data_type == 'multimodal':
            #dataset_train = MultimodalDataset(data_path, config_module.data_name, 'train')
            #dataset_val = MultimodalDataset(data_path, config_module.data_name, 'val')
            dataset_train = SequenceData(feat_path, data_path, config_module.sq_len, config_module.data_name, 'train')
            dataset_val = SequenceData(feat_path, data_path, config_module.sq_len, config_module.data_name, 'val')
            # dataset_train = SequenceData_2(feat_path, os.path.join(data_path, 'integrated_train_labels.json'), 100, config_module.data_name, 'train')
            # dataset_val = SequenceData_2(feat_path, os.path.join(data_path, 'integrated_validation_labels.json'), 100, config_module.data_name, 'val')
            
        dataloader_train = DataLoader(dataset_train, batch_size=config_module.batch_size, shuffle=True, num_workers=config_module.num_workers, drop_last=True)
        dataloader_val = DataLoader(dataset_val, batch_size=config_module.batch_size, shuffle=False, num_workers=config_module.num_workers, drop_last=True)
        
        model = load_model(config_module)
        print(model)
        model.to(device)

        if config_module.optimizer == "adamw":
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config_module.lr, betas=(0.9, 0.999), weight_decay=0.05)
        elif config_module.optimizer == "adam":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config_module.lr, betas=(0.9, 0.999))
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config_module.lr, momentum=config_module.momentum, weight_decay=config_module)

        if config_module.data_name == 'va':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        scheduler = CosineAnnealingLR(optimizer, T_max=config_module.epochs)
        
        log_path = f"output/{config_module.model_name}_{config_module.data_name}_{config_module.data_type}"
        os.makedirs(log_path, exist_ok=True)

        logging.basicConfig(filename=f"{log_path}/training_log.log", level=logging.INFO)

        best_performance = 0.0

        for epoch in range(config_module.epochs):
            model, train_loss = train_function(model, dataloader_train, criterion, optimizer, device, config_module)
            
            print(f'Epoch {epoch+1}/{config_module.epochs}, Train Loss: {train_loss:.4f}')
            logging.info(f'Epoch {epoch}/{config_module.epochs - 1}, Train Loss: {train_loss:.4f}')

            scheduler.step()
            performance, val_loss = evaluate_function(model, dataloader_val, criterion, device, config_module)
            
            print(f'Epoch {epoch+1}/{config_module.epochs}, Validation Loss: {val_loss:.4f}, Average CCC: {performance:.4f}')
            logging.info(f'Epoch {epoch}/{config_module.epochs - 1}, Validation Loss: {val_loss:.4f}, Average CCC: {performance:.4f}')

            wandb.log({"Epoch": epoch, "Train Loss": train_loss, "Validation Loss": val_loss, "Performance": performance, 'lr': scheduler.get_last_lr()[0]})

            if performance > best_performance:      # Save Best Model.pt
                best_val_loss = val_loss            # Update best validation loss
                best_performance = performance      # Update best F1-Score
                best_epoch = epoch                  # Update best epoch

                torch.save(model.state_dict(), os.path.join(log_path, "[epoch_{}]best_model_loss_{:.3f}_f1_{:.3f}_.pth".format(best_epoch, best_val_loss, best_performance)))

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(log_path, 'checkpoint.pth.tar'))

            torch.save(model.state_dict(), os.path.join(log_path, f"model_{epoch}.pth"))

    elif config_module.mode == 'inference':
        pass
    
    elif config_module.mode == 'extract':
        extract_feature(config_module.data_type, device)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Learning Model Configuration')
    parser.add_argument('--config', type=str, required=True, help='Config module name to use')
    args = parser.parse_args()

    config_module = update_config(args)

    fix_seed()
    # main_ old(config_module)
    main(config_module)
