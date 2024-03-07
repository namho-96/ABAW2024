import logging
from config_abaw_au import get_args
from prediction import predict_function
from data.dataset import setup_dataset
from utils.utils import fix_seed, setup_log, log_and_checkpoint
from train import train_function, evaluate_function, setup_training, Trainer
logging.getLogger().setLevel(logging.INFO)

def main(config):

    if config.mode == 'train':
        start_epoch = 0
        trainer = Trainer(config)

        log_path = setup_log(config)  # wandb & log 설정
        dataloader_train, dataloader_val = setup_dataset(config)  # 데이터 로드

        best_performance = float('-inf')
        for epoch in range(start_epoch, config.epochs):
            _ = trainer.train(dataloader_train)
            results_dict = trainer.evaluate(dataloader_val)
            best_performance = log_and_checkpoint(epoch, results_dict, log_path, best_performance)

        #
        # # 학습 설정
        # device, model, optimizer, scheduler, criterion = setup_training(config)
        #
        # # 이어서 학습할 경우 체크포인트 로드
        # start_epoch = 0
        # if config.resume:
        #     model, optimizer, start_epoch = load_checkpoint(model, optimizer, config)
        #
        # log_path = setup_log(config)    # wandb & log 설정
        #
        # dataloader_train, dataloader_val = setup_dataset(config)    # 데이터 로드
        #
        # best_performance = float('-inf')
        # for epoch in range(start_epoch, config.epochs):
        #     # Train
        #     _, train_loss = train_function(model, dataloader_train, criterion, optimizer, device, config)
        #     # Validate
        #     performance, val_loss = evaluate_function(model, dataloader_val, criterion, device, config)
        #     # 로깅 및 체크포인트 저장
        #     best_performance = log_and_checkpoint(epoch, model, optimizer, train_loss, val_loss, performance, scheduler, log_path, best_performance, config)
        #     scheduler.step()

    elif config.mode == 'predict':
        predict_function(config)                # 최종 txt 파일 생성 함수

    else:
        assert ValueError(f"Unknown mode: {config.mode}")


if __name__ == "__main__":
    configs = get_args()
    fix_seed()
    main(configs)