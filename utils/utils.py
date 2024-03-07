import os
import wandb
import logging
import random
import torch
import numpy as np
from datetime import datetime


def setup_log(config):
    if not wandb.api.api_key:
        wandb.login()
    wandb.init(project='ABAW2024', name=config.train_name, config=config_to_dict(config))
    wandb.run.save()

    # 로깅 설정
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("output", config.task, current_time)
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


def log_and_checkpoint(epoch, state_dict, log_path, best_performance):
    if not hasattr(log_and_checkpoint, "config_saved"):
        # config 정보를 텍스트 파일로 저장합니다.
        config_path = os.path.join(log_path, "config.txt")
        with open(config_path, "w") as f:
            for key, value in state_dict['args'].__dict__.items():
                f.write(f"{key}: {value}\n")
        log_and_checkpoint.config_saved = True

    # 현재 성능이 이전 최고 성능보다 좋을 경우 모델 저장
    if state_dict['performance'] > best_performance:
        logging.info(f"New best model with performance: {state_dict['performance']:.4f}, saving model...")
        best_performance = state_dict['performance']  # 최고 성능 갱신
        best_model_path = os.path.join(log_path, f"best_model_epoch_{epoch}_performance_{state_dict['performance']:.4f}.pth")
        torch.save({
            'epoch': epoch + 1,
            'state_dict': state_dict['model'].state_dict(),
            'optimizer': state_dict['optimizer'].state_dict(),
            'performance': state_dict['performance'],
            'config': state_dict['args'].__dict__
        }, best_model_path)

    # 로그 및 체크포인트 저장
    logging.info(f'Epoch {epoch}: Train Loss: {state_dict["train_loss"]:.4f}, Valid Loss: {state_dict["eval_loss"]:.4f}, '
                 f'Performance: {state_dict["performance"]:.4f}, Best: {best_performance:.4f}')

    wandb.log({"Epoch": epoch,
               "Train Loss": state_dict["train_loss"],
               "Validation Loss": state_dict["eval_loss"],
               "Performance": state_dict["performance"],
               "Best Performance": best_performance,
               "Learning Rate": state_dict['scheduler'].get_last_lr()[0]})

    return best_performance



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