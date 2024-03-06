from prediction import predict_function
from data.dataset import TemporalDataset, SpatialDataset, MultimodalDataset, SequenceData, SequenceData_2

from utils.utils import fix_seed, load_checkpoint, setup_log, log_and_checkpoint

from train import train_function, evaluate_function
from torch.utils.data import DataLoader

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.model import load_model
from config_abaw_au import get_args
from extract import *
import logging
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
        predict_function(config)                # 최종 txt 파일 생성 함수
    elif config.mode == 'extract':
        extract_feature(config.data_type, device)
    else:
        assert ValueError(f"Unknown mode: {config.mode}")


if __name__ == "__main__":
    configs = get_args()
    fix_seed()
    main(configs)
