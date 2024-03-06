import os
import wandb
import logging
import random
import torch
import numpy as np
from datetime import datetime
from torchvision.transforms import Normalize, Compose, ToPILImage
from PIL import Image
import importlib


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