# inference.py
import argparse
import importlib
import torch
import torch.nn.functional as F
from models.model import load_pretrained_model
from torch.utils.data import DataLoader
from data.dataset import RandomFrameDataset, AudioDataset, TemporalDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_model(model_path,config_module):
    # 모델 구조 로드 및 초기화
    model = load_pretrained_model(config_module)
    # 모델 상태 불러오기
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def inference(model, dataloader, config_module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    softmax_scores_file = f'output/{config_module.model_name}_{config_module.data_name}_{config_module.data_type}/softmax_scores.txt'
    with torch.no_grad(), open(softmax_scores_file, 'w') as file:
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Softmax 점수 계산
            softmax_scores = F.softmax(outputs, dim=1)
            for idx, softmax_score in enumerate(softmax_scores):
                # Softmax 점수를 쉼표로 구분된 문자열로 변환
                score_str = ','.join([f'{score:.4f}' for score in softmax_score.cpu().numpy()])
                file.write(score_str + '\n')

    # 정확도와 F1 점수 계산
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

    # 혼동 행렬 계산
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 혼동 행렬 시각화 및 저장
    plt.figure(figsize=(30, 30))  # 그림 크기 증가
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', annot_kws={"size": 8},  # annot_kws로 글꼴 크기 조정
            xticklabels=range(config_module.num_classes), yticklabels=range(config_module.num_classes))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(f'output/{config_module.model_name}_{config_module.data_name}_{config_module.data_type}/confusion_matrix.jpg')
    plt.close()


def main(config_module):
    model_path = f"output/{config_module.model_name}_{config_module.data_name}_{config_module.data_type}/{config_module.model_name}_{config_module.data_name}_{config_module.data_type}.pth"
    model = load_model(model_path,config_module)

    if config_module.data_name == 'ucf':
        data_path_train = 'data/ucf_train.txt'
        data_path_test = 'data/ucf_test.txt'
    elif config_module.data_name == 'kinetics':
        data_path_train = 'data/kinetics_train.txt'
        data_path_test = 'data/kinetics_test.txt'
        

    if config_module.data_type == 'spatial':
        dataset = RandomFrameDataset(f'{data_path_test}', 'test')
    elif config_module.data_type == 'audio':
        dataset = AudioDataset(f'{data_path_test}', 'test')
    elif config_module.data_type == 'temporal':
        dataset = TemporalDataset(f'{data_path_test}', 'test')
    dataloader = DataLoader(dataset, batch_size=config_module.batch_size, shuffle=False)
    # 추론 수행
    inference(model, dataloader,config_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Learning Model Configuration')
    parser.add_argument('--config', type=str, required=True, help='Config module name to use')
    parser.add_argument('--model_name', type=str, required=True, help='Model module name to use')
    parser.add_argument('--data_type', type=str, required=True, help='Model module name to use')
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    config_module.model_name = args.model_name
    config_module.data_type = args.data_type
    
    main(config_module)

