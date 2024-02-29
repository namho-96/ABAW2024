import argparse
from utils import *
from train import *
from prediction import predict_function
from data.dataset import setup_dataset
import logging

logging.getLogger().setLevel(logging.INFO)

def main(config):

    if config.mode == 'train':
        # 학습 설정
        device, model, optimizer, scheduler, criterion = setup_training(config)

        # 이어서 학습할 경우 체크포인트 로드
        start_epoch = 0
        if config.resume:
            model, optimizer, start_epoch = load_checkpoint(model, optimizer, config)
            
        log_path = setup_log(config)    # wandb & log 설정

        dataloader_train, dataloader_val = setup_dataset(config)    # 데이터 로드

        best_performance = float('-inf')
        for epoch in range(start_epoch, config.epochs):
            # Train
            model, train_loss = train_function(model, dataloader_train, criterion, optimizer, device, config)
            # Validate
            val_loss, performance = evaluate_function(model, dataloader_val, criterion, device, config)
            # 로깅 및 체크포인트 저장
            best_performance = log_and_checkpoint(epoch, model, optimizer, train_loss, val_loss, performance, scheduler, log_path, best_performance, config)
            scheduler.step()

    elif config.mode == 'predict':
        predict_function(config) # 최종 txt 파일 생성 함수
        
    else:
        assert ValueError(f"Unknown mode: {config.mode}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Learning Model Configuration')
    parser.add_argument('--config', type=str, required=True, help='Config module name to use')
    args = parser.parse_args()

    config_module = update_config(args)

    fix_seed()
    # main_ old(config_module)
    main(config_module)
