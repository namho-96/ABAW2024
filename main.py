import argparse
import importlib
from data.dataset import TemporalDataset, SpatialDataset, MultimodalDataset, SequenceData, SequenceData_2
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
import wandb


def setup_wandb():
    if not wandb.api.api_key:
        wandb.login()
    wandb.init(project='ABAW2024', name='ABAW2024_va')
    wandb.run.save()


def main(config_module):
    device = torch.device(f"cuda:{config_module.device}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = load_model(config_module)
    model.to(device)
    if config_module.mode == 'train':
        setup_wandb()   # Initialize wandb only in training mode
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
            
        dataloader_train = DataLoader(dataset_train, batch_size=config_module.batch_size, shuffle=True, num_workers=config_module.num_workers)
        dataloader_val = DataLoader(dataset_val, batch_size=config_module.batch_size, shuffle=False, num_workers=config_module.num_workers)
        
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

        # criterion = nn.MSELoss()
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

            torch.save(model.state_dict(), os.path.join(log_path, f"model_{epoch}.pth"))

    elif config_module.mode == 'inference':
        pass
    
    elif config_module.mode == 'extract':
        extract_feature(config_module.data_type, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Learning Model Configuration')
    # # Environments
    # parser.add_argument('--feat_path', default='./datasets/features', type=str, help='feature .npy path')
    # parser.add_argument('--annot_path', default='./datasets/6th ABAW Annotations/VA_Estimation_Challenge', type=str, help='annotation file path')
    # parser.add_argument('--device', default=0, type=int, help='Select GPU device')
    # parser.add_argument('--num_workers', default=4, type=int, help='num_workers')
    #
    # # Data
    # parser.add_argument('--data_name', type=str, required=True, help='au/expr/va')
    # parser.add_argument('--data_type', type=str, required=True, help='spatial/multimodal')
    #
    # # Model
    # parser.add_argument('--num_classes', default='base', type=str, help='au/expr - 8, va - 2')
    # parser.add_argument('--num_head', default=4, type=int, help='transformer head number')
    # parser.add_argument('--num_features', default=768, type=int, help='feature dimension')
    #
    # # Training
    # parser.add_argument('--mode',       type=str, required=True, help='train/inference/extract')
    # parser.add_argument('--batch_size', default=768, type=int, help='training batch size')
    # parser.add_argument('--epoch', default=100, type=int, help='training epoch')
    # parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

    parser.add_argument('--config', type=str, required=True, help='Config module name to use')
    args = parser.parse_args()

    config_module = update_config(args)

    fix_seed()
    main(config_module)

