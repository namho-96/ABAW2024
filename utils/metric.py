import numpy as np
import torch
from sklearn.metrics import f1_score


def evaluate_performance(y_true, y_pred, data_name):
    if data_name == 'va':
        # CCC 계산
        CCC_arousal = CCC_torch(y_true[:, 0], y_pred[:, 0])
        CCC_valence = CCC_torch(y_true[:, 1], y_pred[:, 1])
        performance = 0.5 * (CCC_arousal.item() + CCC_valence.item())
    else:
        # F1 점수 계산을 위해 y_pred를 이진 레이블로 변환
        y_pred_binary = y_pred > 0.5
        if data_name == 'au':
            # 다중 레이블 분류 문제에 대한 처리
            performance = f1_score(y_true.cpu().numpy(), y_pred_binary.cpu().numpy(), average='macro', zero_division=1)
        else:
            # 다중 클래스 분류 또는 단일 레이블 분류 문제에 대한 처리
            # 'expr'와 같은 경우
            performance = f1_score(y_true.cpu().numpy().argmax(axis=1), y_pred_binary.cpu().numpy().argmax(axis=1), average='macro', zero_division=1)
    return performance


def CCC_torch(y_true, y_pred):
    vx = y_true - torch.mean(y_true)
    vy = y_pred - torch.mean(y_pred)
    rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    var_true = torch.var(y_true)
    var_pred = torch.var(y_pred)
    std_true = torch.sqrt(var_true)
    std_pred = torch.sqrt(var_pred)
    ccc = (2. * rho * std_true * std_pred) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc


def CCC_np(x, y):
    x = np.array(x)
    y = np.array(y)
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

