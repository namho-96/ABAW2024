import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import os

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

