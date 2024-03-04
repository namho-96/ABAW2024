from torchvision.transforms import Normalize, Compose, ToPILImage
from PIL import Image
import torch
import os
from sklearn.metrics import f1_score
import numpy as np
import importlib
import random
import wandb
from datetime import datetime
import logging


LOGGING_NAME = 'ABAW2024'
# LOGGER = setup_log(LOGGING_NAME)

def setup_log(config):
    if not wandb.api.api_key:
        wandb.login()
    wandb.init(project='ABAW2024', name=config.train_name)
    wandb.run.save()

    # 로깅 설정
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("output", config.data_name, current_time)
    os.makedirs(log_path, exist_ok=True)

    # 학습로그 저장
    log_filename = os.path.join(log_path, "training_log.log")
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f"log path: {log_path}")

    return log_path


def log_and_checkpoint(epoch, model, optimizer, train_loss, val_loss, performance, scheduler, log_path, best_performance, config):
    if not hasattr(log_and_checkpoint, "config_saved"):
        # config 정보를 텍스트 파일로 저장합니다.
        config_path = os.path.join(log_path, "config.txt")
        with open(config_path, "w") as f:
            for key, value in config_to_dict(config).items():
                f.write(f"{key}: {value}\n")
        log_and_checkpoint.config_saved = True

    # 로그 및 체크포인트 저장
    logging.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Performance: {performance:.4f}')
    wandb.log({"Epoch": epoch, "Train Loss": train_loss, "Validation Loss": val_loss, "Performance": performance, "Learning Rate": scheduler.get_last_lr()[0]})

    # 현재 성능이 이전 최고 성능보다 좋을 경우 모델 저장
    if performance > best_performance:
        logging.info(f"New best model with performance: {performance:.4f}, saving model...")
        best_performance = performance  # 최고 성능 갱신
        best_model_path = os.path.join(log_path, f"best_model_epoch_{epoch}_performance_{performance:.4f}.pth")
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'performance': performance,
            'config': config_to_dict(config)
        }, best_model_path)

    return best_performance


def config_to_dict(config_module):
    """
    config module을 dictionary로 변경
    추후 모델 저장시 state 저장할 때 사용
    """
    config_dict = {}
    for attribute in dir(config_module):
        if not attribute.startswith("__"):
            value = getattr(config_module, attribute)
            config_dict[attribute] = value
    return config_dict


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_checkpoint(model, optimizer, config):
    """체크포인트 로드 함수"""
    start_epoch = 0
    filename = config.resume_path
    if os.path.isfile(filename):
        logging.info(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    else:
        logging.info(f"No checkpoint found at '{filename}'")
    return model, optimizer, start_epoch


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
    # config_module.data_type = args.data_type
    # config_module.mode = args.mode
    # config_module.device = args.device
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


def CCC_score(x, y):
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


def CCC_loss(x, y):
    y = y.view(-1)
    x = x.view(-1)
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2)))+1e-8)
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
    return 1-ccc, ccc



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