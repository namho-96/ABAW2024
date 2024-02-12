import os
import random
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 학습 데이터 전처리
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 테스트 데이터 전처리
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 학습 데이터 전처리 (1채널 이미지용)
train_transform_op = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # 1채널 이미지용 정규화 (평균과 표준편차를 하나의 값으로 설정)
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 테스트 데이터 전처리 (1채널 이미지용)
test_transform_op = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # 1채널 이미지용 정규화
    transforms.Normalize(mean=[0.5], std=[0.5])
])




class TemporalDataset(Dataset):
    def __init__(self, txt_file, mode='train', transform=None):
        assert mode in ['train', 'test'], "Mode must be 'train' or 'test'"

        self.mode = mode
        self.data = []
        with open(txt_file, 'r') as file:
            for line in file:
                path, label = line.strip().split()
                self.data.append((os.path.join('/media/jun/jun4/2024', path), int(label)))
        
        self.transform = train_transform_op if mode == 'train' else test_transform_op

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        flow_x_path = os.path.join(video_path, 'flow_x')
        flow_y_path = os.path.join(video_path, 'flow_y')

        flow_x_files = sorted([f for f in os.listdir(flow_x_path) if f.endswith('.jpg')])
        flow_y_files = sorted([f for f in os.listdir(flow_y_path) if f.endswith('.jpg')])

        # 각 방향에서 10개의 이미지 선택 (부족하면 반복)
        num_frames_per_direction = 10
        flow_x_files = self.select_frames(flow_x_files, num_frames_per_direction)
        flow_y_files = self.select_frames(flow_y_files, num_frames_per_direction)

        # 이미지 로드 및 변환
        flow_images = []
        for file in flow_x_files + flow_y_files:
            image_path = os.path.join(flow_x_path if file in flow_x_files else flow_y_path, file)
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            flow_images.append(image)

        # 스택으로 결합
        flow_stack = torch.stack(flow_images)
        flow_stack = torch.squeeze(flow_stack, 1)
        return flow_stack, label

    def select_frames(self, files, target_length):
        if len(files) >= target_length:
            # 전체 파일 수에서 균등하게 선택
            indices = np.linspace(0, len(files) - 1, target_length, dtype=int)
            return [files[i] for i in indices]
        else:
            # 부족한 경우 반복
            return (files * target_length)[:target_length]

        return spectrogram_image, label

class SpatialDataset(Dataset):
    def __init__(self, dataset_dir, data_name, mode='train', transform=None):
        assert mode in ['train','val', 'test'], "Mode must be 'train' or 'test'"
        assert data_name in ['au', 'va', 'expr'], "data_name must be 'au', 'va', or 'expr'"
        
        self.mode = mode
        self.data_name = data_name
        self.data = []
        self.transform = transform
        self.base_path = os.getcwd()  # 비디오 프레임이 저장된 기본 경로
        if self.mode == 'train':
            dataset_dir = os.path.join(dataset_dir, 'Train_Set')
        elif self.mode == 'val':
            dataset_dir = os.path.join(dataset_dir, 'Validation_Set')
        elif self.mode == 'test':
            dataset_dir = os.path.join(dataset_dir, 'Test_Set')
        # 주어진 디렉토리 내의 모든 txt 파일을 순회
        for txt_file in sorted(os.listdir(dataset_dir)):
            if txt_file.endswith('.txt'):
                video_id = txt_file.replace('.txt', '')
                with open(os.path.join(dataset_dir, txt_file), 'r') as file:
                    next(file)  # 첫 번째 줄 건너뛰기
                    frame_number = 1  # 프레임 번호 초기화
                    for line in file:
                        label = line.strip().split(',')
                        
                        # 레이블 타입에 따라 적절하게 변환 및 -1 체크
                        if '-1' not in label:  # -1이 포함되지 않은 경우만 처리
                            if self.data_name == 'au':
                                label = [int(x) for x in label]
                            elif self.data_name == 'va':
                                label = [float(x) for x in label]
                            elif self.data_name == 'expr':
                                label = int(label[0])
                            
                            frame_path = os.path.join(self.base_path, video_id, f"{frame_number}.jpg")
                            self.data.append((frame_path, label))
                        frame_number += 1  # 프레임 번호 업데이트

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_path, label = self.data[idx]
        """
        image = Image.open(frame_path)

        if self.transform:
            image = self.transform(image)
        """
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return frame_path, label_tensor

