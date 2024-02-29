import torch
import torch.nn as nn 
import numpy as np
import logging
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.model import load_model
from sklearn.metrics import f1_score
from custom_metrics import *

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
        criterion = nn.MSELoss()
    elif config.data_name == 'au':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    return device, model, optimizer, scheduler, criterion


# 학습 함수 정의
def train_function(model, dataloader, criterion, optimizer, device, config):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Initializing")

    for vid, aud, labels in progress_bar:
        optimizer.zero_grad()
        vid, aud, labels = vid.to(device), aud.to(device), labels.to(device)
        outputs = model(vid, aud)

        if config.data_name == 'va':
            loss, ccc_loss = compute_VA_loss(outputs[0], outputs[1], labels)
        else:
            outputs = outputs.reshape(-1, config.num_classes)
            labels = labels.reshape(-1, config.num_classes)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        average_loss = running_loss / (progress_bar.n + 1)          # progress_bar.n은 현재까지 처리된 배치의 수입니다.
        progress_bar.set_description(f"Loss: {loss.item():.4f}, Avg_Loss: {average_loss:.4f}")

    train_loss = running_loss / len(dataloader)
    return model, train_loss


# 평가 함수 정의
def evaluate_function(model, dataloader, criterion, device, config):
    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Initializing")
    
    if config.data_name == 'va':
        prediction_valence = []
        prediction_arousal = []
        gt_valence = []
        gt_arousal = []
    else:
        if config.data_name == 'au':    
            m = nn.Sigmoid()
        prediction = []
        gt = []

    with torch.no_grad():
        for vid, aud, labels in progress_bar:
            vid, aud, labels = vid.to(device), aud.to(device), labels.to(device)
            outputs = model(vid, aud)

            if config.data_name == 'va':
                loss, ccc_loss = compute_VA_loss(outputs[0], outputs[1], labels)
                progress_bar.set_description(f"Loss: {loss.item():.4f}")
                prediction_valence.extend(outputs[0][:, :, 0].cpu().numpy())
                prediction_arousal.extend(outputs[1][:, :, 0].cpu().numpy())
                gt_valence.extend(labels[:, :, 0].cpu().numpy())
                gt_arousal.extend(labels[:, :, 1].cpu().numpy())
            else:
                outputs = outputs.reshape(-1, config.num_classes)
                labels = labels.reshape(-1, config.num_classes)
                loss = criterion(outputs, labels)
                
                if config.data_name == 'au':
                    predicted = m(outputs)
                    predicted = predicted > 0.5
                elif config.data_name == 'expr':
                    _, predicted = outputs.max(1)
                    
                prediction.extend(predicted.cpu().numpy())
                gt.extend(labels.cpu().numpy())   
            
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            running_loss += loss.item()

    test_loss = running_loss / len(dataloader)
    if config.data_name == 'va':
        avg_performance = 0.5 * (CCC_score(prediction_valence, gt_valence) + CCC_score(prediction_arousal, gt_arousal))
    else:    
        f1s = f1_score(gt, prediction, average=None, zero_division=1)
        avg_performance = f1_score(gt, prediction, average='macro', zero_division=1)
        
    return test_loss, avg_performance