import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from tqdm import tqdm
from torchviz import make_dot
from sklearn.metrics import f1_score
from models.model import load_model
from utils.loss import VA_loss, CCC_loss
from utils.metric import CCC_np, CCC_torch
from utils.scheduler import CosineAnnealingWarmUpRestarts


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = load_model(self.args)
        self.device, self.optimizer, self.criterion = self.setup_training(self.model)
        self.scheduler = self.setup_scheduler()
        self.model.to(self.device)
        self.state_dict = {'args': self.args}
        if self.args.resume:
            self.load_checkpoint()

    def setup_scheduler(self):
        # Scheduler
        if self.args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.freq_epochs,
                                                                   eta_min=self.args.min_lr)
        elif self.args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        elif self.args.scheduler == 'warmup':
            scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, T_0=self.args.epochs, T_mult=1,
                                                      eta_max=self.args.eta_max, T_up=5, gamma=1.0)

        return scheduler

    def setup_training(self, model):
        # Set device
        device = torch.device(f"cuda:{self.args.device}" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Optimizer
        if self.args.optimizer == "adamw":
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                                    betas=(0.9, 0.999), weight_decay=0.05)
        elif self.args.optimizer == "adam":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                                   betas=(0.9, 0.999))
        elif self.args.optimizer == "sgd":
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                                  momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        # Loss function
        if self.args.task == 'va':
            criterion = nn.MSELoss()
        elif self.args.task == 'au':
            criterion = nn.BCELoss()
        elif self.args.task == 'expr':
            criterion = nn.CrossEntropyLoss()

        return device, optimizer, criterion

    def load_checkpoint(self):
        """체크포인트 로드 함수"""
        filename = self.args.resume
        if os.path.isfile(filename):
            logging.info(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        else:
            logging.info(f"No checkpoint found at '{filename}'")

    def train(self, dataloader):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc="Initializing")

        for vid, aud, labels in progress_bar:
            self.optimizer.zero_grad()
            vid, aud, labels = vid.to(self.device), aud.to(self.device), labels.to(self.device)

            if self.args.mixup:
                vid, aud, labels = mixup_function(vid, aud, labels, self.args.task)

            outputs = self.model(vid, aud)
            if self.args.task == 'va':
                loss, ccc_loss = VA_loss(outputs[0], outputs[1], labels)
                if self.args.vis:
                    make_dot(outputs[0].mean(), params=dict(self.model.named_parameters()), show_attrs=True, show_saved=True).render("model_arch", format="png")
                    self.args.vis = False

            elif self.args.task == 'au':
                outputs = outputs.reshape(-1, self.args.num_classes)
                labels = labels.reshape(-1, self.args.num_classes)
                loss = self.criterion(outputs.to(torch.float32), labels.to(torch.float32))

            elif self.args.task == 'expr':
                outputs = outputs.reshape(-1, self.args.num_classes)
                if not self.args.mixup:
                    labels = labels.reshape(-1)
                loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            average_loss = running_loss / (progress_bar.n + 1)  # progress_bar.n은 현재까지 처리된 배치의 수입니다.            
            if self.args.task == 'va':
                progress_bar.set_description(f"Batch_Loss: {loss.item():.4f}, Avg_Loss: {average_loss:.4f}, CCC_Loss: {ccc_loss:.4f}")
            else:
                progress_bar.set_description(f"Batch_Loss: {loss.item():.4f}, Avg_Loss: {average_loss:.4f}")

        train_loss = running_loss / len(dataloader)
        self.scheduler.step()
        self.state_dict.update({'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler, 'train_loss': train_loss, 'lr' : self.scheduler})
        return self.state_dict

    def train_all(self, train_loader, val_loader):
        self.model.train()
        running_loss = 0.0
        combine = itertools.chain(train_loader, val_loader)
        progress_bar = tqdm(combine, desc="Initializing")

        for vid, aud, labels in progress_bar:
            self.optimizer.zero_grad()
            vid, aud, labels = vid.to(self.device), aud.to(self.device), labels.to(self.device)

            if self.args.mixup:
                vid, aud, labels = mixup_function(vid, aud, labels, self.args.task)

            outputs = self.model(vid, aud)
            if self.args.task == 'va':
                loss, ccc_loss = VA_loss(outputs[0], outputs[1], labels)
                if self.args.vis:
                    make_dot(outputs[0].mean(), params=dict(self.model.named_parameters()), show_attrs=True, show_saved=True).render("model_arch", format="png")
                    self.args.vis = False

            elif self.args.task == 'au':
                outputs = outputs.reshape(-1, self.args.num_classes)
                labels = labels.reshape(-1, self.args.num_classes)
                loss = self.criterion(outputs.to(torch.float32), labels.to(torch.float32))

            elif self.args.task == 'expr':
                outputs = outputs.reshape(-1, self.args.num_classes)
                if not self.args.mixup:
                    labels = labels.reshape(-1)
                loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            average_loss = running_loss / (progress_bar.n + 1)  # progress_bar.n은 현재까지 처리된 배치의 수입니다.
            progress_bar.set_description(f"Batch_Loss: {loss.item():.4f}, Avg_Loss: {average_loss:.4f}")

        train_loss = running_loss / (len(train_loader) + len(val_loader))
        self.scheduler.step()
        self.state_dict.update({'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler, 'train_loss': train_loss})
        return self.state_dict

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc="Initializing")

        if self.args.task == 'va':
            pv, pa, gv, ga = [], [], [], []
        else:
            m = nn.Sigmoid() if self.args.task == 'au' else None
            prediction, gt = [], []

        for vid, aud, labels in progress_bar:
            vid, aud, labels = vid.to(self.device), aud.to(self.device), labels.to(self.device)
            outputs = self.model(vid, aud)

            if self.args.task == 'va':
                loss, ccc_loss = VA_loss(outputs[0], outputs[1], labels)
                pv.extend(outputs[0][:, :, 0].cpu().numpy())
                pa.extend(outputs[1][:, :, 0].cpu().numpy())
                gv.extend(labels[:, :, 0].cpu().numpy())
                ga.extend(labels[:, :, 1].cpu().numpy())

            elif self.args.task == 'au':
                outputs = outputs.reshape(-1, self.args.num_classes)
                labels = labels.reshape(-1, self.args.num_classes)
                loss = self.criterion(outputs.to(torch.float32), labels.to(torch.float32))

                predicted = outputs > 0.5

                prediction.extend(predicted.cpu().numpy())
                gt.extend(labels.cpu().numpy())

            elif self.args.task == 'expr':
                outputs = outputs.reshape(-1, self.args.num_classes)
                labels = labels.reshape(-1)
                loss = self.criterion(outputs, labels)

                _, predicted = outputs.max(1)

                prediction.extend(predicted.cpu().numpy())
                gt.extend(labels.cpu().numpy())

            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            running_loss += loss.item()


        test_loss = running_loss / len(dataloader)
        if self.args.task == 'va':
            avg_performance = 0.5 * (CCC_np(pv, gv) + CCC_np(pa, ga))
        else:
            avg_performance = f1_score(gt, prediction, average='macro', zero_division=1)

        self.state_dict.update({'performance': avg_performance, 'eval_loss': test_loss})
        return self.state_dict


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

    if task == "expr":
        labels = mixup_target(labels, 8, lam)
    else:
        labels = labels * lam + labels.flip(0) * (1. - lam)

    if lam == 1.:
        return vid, aud, labels

    vid_flipped = vid.flip(0).mul_(1. - lam)
    vid.mul_(lam).add_(vid_flipped)
    aud_flipped = aud.flip(0).mul_(1. - lam)
    aud.mul_(lam).add_(aud_flipped)
    return vid, aud, labels

