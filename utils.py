from torchvision.transforms import Normalize, Compose, ToPILImage
from PIL import Image
import torch
import os
from sklearn.metrics import f1_score
import numpy as np
import importlib
import random

def fix_seed(seed: int = 42):
    """
    PyTorch 및 NumPy 라이브러리의 시드를 고정하는 함수.
    
    Args:
        seed (int): 고정할 시드 값. 기본값은 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def update_config(args):   
    config_module = importlib.import_module(args.config)
    config_module.data_type = args.data_type
    config_module.mode = args.mode
    config_module.device = args.device
    return config_module
    
def denormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    denormalize = Compose([
        Normalize(mean=[0., 0., 0.], std=[1./s for s in std]),
        Normalize(mean=[-m for m in mean], std=[1., 1., 1.]),
        ToPILImage()
    ])
    return denormalize(tensor)

def save_sample_images(dataloader, save_path, file_name, nrow=8):
    images, _ = next(iter(dataloader))
    # Denormalize and convert to PIL images
    pil_images = [denormalize(img) for img in images]

    # 격자의 한 칸에 들어갈 이미지 크기
    img_width, img_height = pil_images[0].size
    # 전체 격자 이미지의 크기 계산
    grid_width = img_width * nrow
    grid_height = img_height * (len(pil_images) // nrow)

    # 전체 격자 이미지 생성
    grid_img = Image.new('RGB', (grid_width, grid_height))

    # 개별 이미지들을 격자에 붙여넣기
    for i, img in enumerate(pil_images):
        grid_x = (i % nrow) * img_width
        grid_y = (i // nrow) * img_height
        grid_img.paste(img, (grid_x, grid_y))

    # 저장 경로 확인 및 생성
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 이미지 저장
    grid_img.save(os.path.join(save_path, file_name))

def CCC(y_true, y_pred):
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


def evaluate_performance(y_true, y_pred, data_name):
    if data_name == 'va':
        # CCC 계산
        CCC_arousal = CCC(y_true[:, 0], y_pred[:, 0])
        CCC_valence = CCC(y_true[:, 1], y_pred[:, 1])
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