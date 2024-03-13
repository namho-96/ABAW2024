import logging
from config_abaw_au import get_args
from prediction import predict_function
from data.dataset import setup_dataset
from utils.utils import fix_seed, setup_log, log_and_checkpoint
from train import Trainer
logging.getLogger().setLevel(logging.INFO)


def main(config):

    if config.mode == 'train':
        start_epoch = 0
        trainer = Trainer(config)

        log_path = setup_log(config)  # wandb & log 설정
        dataloader_train, dataloader_val = setup_dataset(config)  # 데이터 로드

        best_performance = float('-inf')
        for epoch in range(start_epoch, config.epochs):
            if config.train_all:
                _ = trainer.train_all(dataloader_train, dataloader_val)
            else:
                _ = trainer.train(dataloader_train)
            results_dict = trainer.evaluate(dataloader_val)     # Results_dict 내에 train/eval_loss, 성능, 모델, args 전부 포함
            best_performance = log_and_checkpoint(epoch, results_dict, log_path, best_performance)

    elif config.mode == 'predict':
        predict_function(config)                # 최종 txt 파일 생성 함수

    else:
        assert ValueError(f"Unknown mode: {config.mode}")


if __name__ == "__main__":
    configs = get_args()
    fix_seed()
    main(configs)

