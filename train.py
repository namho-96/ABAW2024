import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import os
from utils import evaluate_performance
from sklearn.metrics import classification_report
from tqdm import tqdm
def train_model(model, dataloader, criterion, optimizer, config_module, device, num_epochs=100):

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    losses = []
    log_path = f"output/{config_module.model_name}_{config_module.data_name}_{config_module.data_type}"
    os.makedirs(log_path, exist_ok=True)

    logging.basicConfig(filename=f"{log_path}/training_log.log", level=logging.INFO)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터를 디바이스로 이동

            optimizer.zero_grad()

            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 역전파 및 최적화
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 에포크가 끝날 때마다 학습률 갱신
        scheduler.step()

        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        logging.info(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

        # 매 epoch마다 손실 그래프 저장
        plt.figure()
        plt.plot(range(1, epoch + 2), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(f"{log_path}/loss_epoch.png")
        plt.close()

    return model

# 학습 함수 정의
def train_function(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    
    for vid, aud, labels in tqdm(dataloader):
        #inputs[0], inputs[1], labels = inputs[0].to(device), inputs[1].to(device), labels.to(device)
        
        vid, aud, labels = vid.to(device), aud.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(vid, aud)
        outputs = outputs.reshape(-1, num_classes)
        labels = labels.reshape(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(dataloader)
    return model, train_loss

# 평가 함수 정의
def evaluate_function(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    perdiction = []
    gt = []
    with torch.no_grad():
        for vid, aud, labels in tqdm(dataloader):
            #inputs, labels = inputs.to(device), labels.to(device)
            vid, aud, labels = vid.to(device), aud.to(device), labels.to(device)
            outputs = model(vid, aud)
            outputs = outputs.reshape(-1, num_classes)
            labels = labels.reshape(-1)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)

            perdiction.extend(predicted.cpu().numpy())
            gt.extend(labels.cpu().numpy())
            
            running_loss += loss.item()

    test_loss = running_loss / len(dataloader)        
    classification_rep = classification_report(gt, perdiction, output_dict=True, zero_division=1)
    performance = 0
    for class_name, metrics in classification_rep.items():
        if class_name.isdigit():
            performance += metrics["f1-score"]
    performance /= 8
    
    return performance, test_loss