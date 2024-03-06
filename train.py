from torchviz import make_dot
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score

from utils.loss import VA_loss, CCC_loss
from utils.metric import CCC_np, CCC_torch


# 학습 함수 정의
def train_function(model, dataloader, criterion, optimizer, device, config):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Initializing")

    for vid, aud, labels in progress_bar:
        #inputs[0], inputs[1], labels = inputs[0].to(device), inputs[1].to(device), labels.to(device)
        optimizer.zero_grad()
        vid, aud, labels = vid.to(device), aud.to(device), labels.to(device)

        if config.mixup:
            vid, aud, labels = mixup_function(vid, aud, labels)

        outputs = model(vid, aud)
        if config.data_name == 'va':
            loss, ccc_loss, ccc_avg, _ = VA_loss(outputs[0], outputs[1], labels)
            if config.vis:
                make_dot(outputs[0].mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("model_arch", format="png")
                config.vis = False
        else:
            outputs = outputs.reshape(-1, config.num_classes).type(torch.float32)
            labels = labels.reshape(-1, config.num_classes).type(torch.float32)  # shape 일치
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        average_loss = running_loss / (progress_bar.n + 1)          # progress_bar.n은 현재까지 처리된 배치의 수입니다.
        progress_bar.set_description(f"Batch_Loss: {loss.item():.4f}, Avg_Loss: {average_loss:.4f}, CCC_Loss: {ccc_loss:.4f}")

    train_loss = running_loss / len(dataloader)
    return model, train_loss


# 평가 함수 정의
def evaluate_function(model, dataloader, criterion, device, config):
    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Initializing")

    if config.data_name == 'va':
        prediction_valence = []
        prediction_arousal = []
        gt_valence = []
        gt_arousal = []
    else:
        if config.data_name == 'au':
            m = nn.Sigmoid()
        prediction = []
        gt = []

    with torch.no_grad():
        for vid, aud, labels in progress_bar:
            vid, aud, labels = vid.to(device), aud.to(device), labels.to(device)
            outputs = model(vid, aud)

            if config.data_name == 'va':
                loss, ccc_loss, ccc_avg, _ = VA_loss(outputs[0], outputs[1], labels)
                progress_bar.set_description(f"Loss: {loss.item():.4f}")
                prediction_valence.extend(outputs[0][:, :, 0].cpu().numpy())
                prediction_arousal.extend(outputs[1][:, :, 0].cpu().numpy())
                gt_valence.extend(labels[:, :, 0].cpu().numpy())
                gt_arousal.extend(labels[:, :, 1].cpu().numpy())
            else:
                outputs = outputs.reshape(-1, config.num_classes)
                labels = labels.reshape(-1, config.num_classes)
                loss = criterion(outputs, labels)

                if config.data_name == 'au':
                    predicted = m(outputs)
                    predicted = predicted > 0.5
                elif config.data_name == 'expr':
                    _, predicted = outputs.max(1)

                prediction.extend(predicted.cpu().numpy())
                gt.extend(labels.cpu().numpy())

            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            running_loss += loss.item()

    test_loss = running_loss / len(dataloader)
    if config.data_name == 'va':
        avg_performance = 0.5 * (CCC_np(prediction_valence, gt_valence) + CCC_np(prediction_arousal, gt_arousal))
    else:
        f1s = f1_score(gt, prediction, average=None, zero_division=1)
        avg_performance = f1_score(gt, prediction, average='macro', zero_division=1)

    return test_loss, avg_performance


def mixup_function(vid, aud, labels):
    lam = float(torch.distributions.beta.Beta(0.8, 0.8).sample())
    if lam == 1.:
        return vid, aud, labels
    vid_flipped = vid.flip(0).mul_(1. - lam)
    vid.mul_(lam).add_(vid_flipped)
    aud_flipped = aud.flip(0).mul_(1. - lam)
    aud.mul_(lam).add_(aud_flipped)
    labels = labels * lam + labels.flip(0) * (1. - lam)
    return vid, aud, labels


"""
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
"""
