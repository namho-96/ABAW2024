import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchviz import make_dot
from sklearn.metrics import f1_score
from models.model import load_model
from utils.loss import VA_loss, CCC_loss
from utils.metric import CCC_np, CCC_torch


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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    return device, model, optimizer, scheduler, criterion
    

# 학습 함수 정의
def train_function(model, dataloader, criterion, optimizer, device, config):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Initializing")

    for vid, aud, labels in progress_bar:
        optimizer.zero_grad()
        vid, aud, labels = vid.to(device), aud.to(device), labels.to(device)

        if config.mixup:
            vid, aud, labels = mixup_function(vid, aud, labels, config.data_name)

        outputs = model(vid, aud)
        if config.data_name == 'va':
            loss, ccc_loss, ccc_avg, _ = VA_loss(outputs[0], outputs[1], labels)
            if config.vis:
                make_dot(outputs[0].mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("model_arch", format="png")
                config.vis = False
        else:
            outputs = outputs.reshape(-1, config.num_classes)
            labels = labels.reshape(-1, config.num_classes)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        average_loss = running_loss / (progress_bar.n + 1)          # progress_bar.n은 현재까지 처리된 배치의 수입니다.
        if config.data_name == 'va':
            progress_bar.set_description(f"Batch_Loss: {loss.item():.4f}, Avg_Loss: {average_loss:.4f}, CCC_Loss: {ccc_loss:.4f}")
        else:
            progress_bar.set_description(f"Batch_Loss: {loss.item():.4f}, Avg_Loss: {average_loss:.4f}")
    train_loss = running_loss / len(dataloader)
    return model, train_loss


# 평가 함수 정의
@torch.no_grad()
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

    for vid, aud, labels in progress_bar:
        vid, aud, labels = vid.to(device), aud.to(device), labels.to(device)
        outputs = model(vid, aud)

        if config.data_name == 'va':
            loss, ccc_loss, ccc_avg, _ = VA_loss(outputs[0], outputs[1], labels)
            prediction_valence.extend(outputs[0][:, :, 0].cpu().numpy())
            prediction_arousal.extend(outputs[1][:, :, 0].cpu().numpy())
            gt_valence.extend(labels[:, :, 0].cpu().numpy())
            gt_arousal.extend(labels[:, :, 1].cpu().numpy())
        else:
            outputs = outputs.reshape(-1, config.num_classes)
            labels = labels.reshape(-1, config.num_classes) if config.data_name == "au" else labels.reshape(-1)
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
        avg_performance = 0.5 * (CCC_np(prediction_valence, gt_valence) + CCC_np(prediction_arousal, gt_arousal))
    else:
        avg_performance = f1_score(gt, prediction, average='macro', zero_division=1)

    return avg_performance, test_loss


def one_hot(x, num_classes, on_value=1., off_value=0.):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value)
    return y1 * lam + y2 * (1. - lam)


def mixup_function(vid, aud, labels, task):
    lam = float(torch.distributions.beta.Beta(0.8, 0.8).sample())
    if lam == 1.:
        return vid, aud, labels
    vid_flipped = vid.flip(0).mul_(1. - lam)
    vid.mul_(lam).add_(vid_flipped)
    aud_flipped = aud.flip(0).mul_(1. - lam)
    aud.mul_(lam).add_(aud_flipped)
    if task == "expr":
        labels = mixup_target(labels, 8)
    else:
        labels = labels * lam + labels.flip(0) * (1. - lam)
    return vid, aud, labels
