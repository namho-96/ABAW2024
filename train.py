from torchviz import make_dot
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import os
from utils import evaluate_performance
from sklearn.metrics import classification_report
from tqdm import tqdm
from utils import CCC_loss, CCC_score


def compute_VA_loss(Vout, Aout, label, criterion):
    Vout = torch.clamp(Vout, -1, 1)     # [bs, sq, 1]
    Aout = torch.clamp(Aout, -1, 1)     # [bs, sq, 1]
    bz, seq, _ = Vout.shape
    label = label.view(bz * seq, -1)        # [bs, sq, 2]
    Vout = Vout.view(bz * seq, -1)
    Aout = Aout.view(bz * seq, -1)

    ccc_valence_loss, ccc_valence = CCC_loss(Vout[:, 0], label[:, 0])
    ccc_arousal_loss, ccc_arousal = CCC_loss(Aout[:, 0], label[:, 1])

    # logging.info(f"ccc_valence_loss:{ccc_valence_loss:.4}, ccc_arousal_loss:{ccc_arousal_loss:.4}")

    ccc_loss = ccc_valence_loss + ccc_arousal_loss
    ccc_avg = 0.5 * (ccc_valence + ccc_arousal)
    # ccc_loss = CCC_loss(Vout[:, 0], label[:, 0]) + CCC_loss(Aout[:, 0], label[:, 1])       # 0 - arousal / 1 - valence

    mse_loss = criterion(Vout[:, 0], label[:, 0]) + criterion(Aout[:, 0], label[:, 1])

    loss = ccc_loss
    return loss, mse_loss, ccc_loss, ccc_avg, [ccc_valence, ccc_arousal]


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
            loss, mse_loss, ccc_loss, ccc_avg, _ = compute_VA_loss(outputs[0], outputs[1], labels, criterion)
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

    total_performance = []
    prediction_valence = []
    prediction_arousal = []
    gt_valence = []
    gt_arousal = []

    with torch.no_grad():
        for vid, aud, labels in progress_bar:
            #inputs, labels = inputs.to(device), labels.to(device)
            vid, aud, labels = vid.to(device), aud.to(device), labels.to(device)
            outputs = model(vid, aud)

            if config.data_name == 'va':
                loss, mse_loss, ccc_loss, ccc_avg, va = compute_VA_loss(outputs[0], outputs[1], labels, criterion)
                # total_performance.append(sum(va))
                # avg_performance = sum(total_performance) / len(total_performance)

                prediction_valence.extend(outputs[0][:, :, 0].cpu().numpy())            # [bs, sq]
                prediction_arousal.extend(outputs[1][:, :, 0].cpu().numpy())            # [bs, sq]
                gt_valence.extend(labels[:, :, 0].cpu().numpy())                        # [bs, sq]
                gt_arousal.extend(labels[:, :, 1].cpu().numpy())

                # progress_bar.set_description(f"Batch_Loss: {loss.item():.4f}, Avg CCC: {avg_performance:.4f}, "
                #                              f"Valence CCC: {va[0]:.4f}, Arousal CCC: {va[1]:.4f}")
            else:
                outputs = outputs.reshape(-1, config.num_classes)
                labels = labels.reshape(-1, config.num_classes)
                loss = criterion(outputs, labels)
                progress_bar.set_description(f"Loss: {loss.item():.4f}")

            running_loss += loss.item()
            # CCC_arousal = CCC(labels[:, 0], outputs[0])
            # CCC_valence = CCC(labels[:, 1], outputs[1])
            # performance = 0.5 * (CCC_arousal.item() + CCC_valence.item())

    test_loss = running_loss / len(dataloader)        
    # classification_rep = classification_report(gt, perdiction, output_dict=True, zero_division=1)
    # avg_performance = sum(total_performance) / len(total_performance)
    avg_performance = 0.5 * (CCC_score(prediction_valence, gt_valence) + CCC_score(prediction_arousal, gt_arousal))

    return avg_performance, test_loss


