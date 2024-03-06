import os
import random
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import json
import h5py


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

def setup_dataset(config):
    data_path = {
        'au': 'AU_Detection_Challenge',
        'expr': 'EXPR_Recognition_Challenge',
        'va': 'VA_Estimation_Challenge'
    }

    feat_path = config.feat_path
    data_path = os.path.join(config.label_path, data_path.get(config.data_name, ''))

    if config.data_type == 'multimodal':
        dataset_train = SequenceData(feat_path, data_path, config.sq_len, config.data_name, 'train')
        dataset_val = SequenceData(feat_path, data_path, config.sq_len, config.data_name, 'val')
    else:
        raise ValueError(f'Wrong Data type : {config.data_type}')

    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, drop_last=True)

    return dataloader_train, dataloader_val



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



class MultimodalDataset(Dataset):
    def __init__(self, dataset_dir, data_name, mode='train', transform=None):
        assert mode in ['train','val', 'test'], "Mode must be 'train' or 'test'"
        assert data_name in ['au', 'va', 'expr'], "data_name must be 'au', 'va', or 'expr'"

        self.mode = mode
        self.data_name = data_name
        self.data = []
        self.transform = transform
        self.base_path = "../dataset/feature"  # 비디오 프레임이 저장된 기본 경로
        if self.mode == 'train':
            dataset_dir = os.path.join(dataset_dir, 'Train_Set')
        elif self.mode == 'val':
            dataset_dir = os.path.join(dataset_dir, 'Validation_Set')
        elif self.mode == 'test':
            dataset_dir = os.path.join(dataset_dir, 'Test_Set')
        # 주어진 디렉토리 내의 모든 txt 파일을 순회
        for txt_file in sorted(os.listdir(dataset_dir)):
            if txt_file.endswith('.txt'):
                image_id = txt_file.replace('.txt', '')
                video_id = image_id.replace("_left", "").replace("_right", "")
                with open(os.path.join(dataset_dir, txt_file), 'r') as file:
                    next(file)  # 첫 번째 줄 건너뛰기
                    frame_number = 1  # 프레임 번호 초기화

                    for line in file:
                        spatial_feature_path = os.path.join(self.base_path, "spatial", image_id, f"{frame_number:05d}.npy")
                        audio_feature_path = os.path.join(self.base_path, "audio", video_id, f"{frame_number:05d}.npy")
                        label = line.strip().split(',')

                        # 레이블 타입에 따라 적절하게 변환 및 -1 체크
                        if '-1' not in label and os.path.isfile(spatial_feature_path):  # -1이 포함되지 않은 경우, feature 파일이 있는 경우 처리
                            if self.data_name == 'au':
                                label = [int(x) for x in label]
                            elif self.data_name == 'va':
                                label = [float(x) for x in label]
                            elif self.data_name == 'expr':
                                label = int(label[0])
                            self.data.append((spatial_feature_path, audio_feature_path, label))
                        frame_number += 1  # 프레임 번호 업데이트

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spatial_feature_path, audio_feature_path, label = self.data[idx]

        spatial_feature = torch.from_numpy(np.load(spatial_feature_path))
        audio_feature = torch.from_numpy(np.load(audio_feature_path))

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return [spatial_feature, audio_feature], label_tensor



class SequenceData(Dataset):
    def __init__(self,
                 feat_root,
                 label_root,
                 seq_len,
                 task,
                 mode,
                 pad_mode='repeat_last'):
        """SequenceData

        Args:
            feat_root (str): feat root path
            label_root (str): label root path
            feat_dict (dict): feat dict in which key is `feat_name`, value is `feat_dim`
            seq_len (int): sequence length
            task (str): `va`, `expr` or `au`
            pad_mode (str): pad mode, here just implemented `repeat_last` (default: 'repeat_last')
        """
        if mode == 'train':
            label_root = os.path.join(label_root, 'Train_Set')
        elif mode == 'val':
            label_root = os.path.join(label_root, 'Validation_Set')
        elif mode == 'test':
            label_root = os.path.join(label_root, 'Test_Set')

        self.feat_root = feat_root
        self.label_root = label_root
        self.seq_len = seq_len
        self.task = task
        self.pad_mode = pad_mode
        self.feat_map = dict()
        self.sequence_list, self.label_list = self.make_sequence()

    def get_txt_contents(self, path):
        """get txt annotation contents, and return a dict which key is `frame_id`(aligned with 05d), value is `annotation`.
        In task `va`:, content is like {'00001': [0.1, 0.2], ...}
        In task `expr`:, content is like {'00001': 1, ...}
        In task `au`:, content is like {'00001': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], ...}

        Args:
            path (str): txt file path

        Returns:
            content (dict)
        """
        with open(path, 'r') as f:
            content = dict()
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                if self.task == 'va':
                    value_list = [float(value)
                                  for value in line.strip('\n').split(',')]
                    content[f'{i :05d}'] = value_list
                elif self.task == 'expr':
                    value_list = int(line.strip('\n'))
                    content[f'{i :05d}'] = value_list
                elif self.task == 'au':
                    value_list = [float(value)
                                  for value in line.strip('\n').split(',')]
                    content[f'{i :05d}'] = value_list

        return content

    def __filter_invalid_annotations(self, label_dict, video_name):
        """filter invalid annotation like `-5` in va, `-1` in expr and au.
        In some case, annotations are given by organizer, but images are not provided (in task `va` and `expr`), so ... just ignore that

        Args:
            label_dict (dict): label dict generate by :method `get_txt_contents`
            video_name (str): video name string without ext

        Returns:
            returned_label_dict (dict): dict organized like `label_dict`, but is filtered.
        """
        returned_label_dict = label_dict.copy()
        if self.task == 'va':
            for seq_id in label_dict.keys():
                if (-5 in label_dict[seq_id]):
                    returned_label_dict.pop(seq_id)
        elif self.task == 'expr':
            for seq_id in label_dict.keys():
                if (-1 == label_dict[seq_id]):
                    returned_label_dict.pop(seq_id)
        elif self.task == 'au':
            for seq_id in label_dict.keys():
                if -1 in label_dict[seq_id]:
                    returned_label_dict.pop(seq_id)

        return returned_label_dict

    def get_video_list(self):
        """get video list

        Returns:
            video_list (list): video list in `task`
        """
        video_list = [x.split('.')[0]
                      for x in sorted(os.listdir(self.label_root))]
        return video_list

    def make_sequence_id_list(self, label_dict, video_name):
        """make sequence id list and sequence label list, it's upon to :attribute `pad_mode`

        Args:
            label_dict (dict): label dict generate by :method `get_txt_contents`
            video_name (str): video name string without ext

        Returns:
            sequence_id_list (list): 2-d sequence id list like [[1, 2, 3, ..., 127, 127], [...], ...]
            sequence_label_list (list): sequence label list
                                        in task `va` like: [[[0.0, 0.1], [0.0, 0.1], ...], [...], ...]
                                        in task `expr` like: [[0, 4, ...], [...], ...]
                                        in task `au` like: [[[0, 0, 1, 0, ...], ...], [...], ...]

        """
        label_dict = self.__filter_invalid_annotations(label_dict, video_name)
        sequence_id_list = list(label_dict.keys())
        sequence_label_list = list()
        if self.pad_mode == 'repeat_last':
            sequence_id_list = [sequence_id_list[i: i + self.seq_len]
                                for i in range(0, len(sequence_id_list), self.seq_len)]

            for seq in sequence_id_list:
                for i in range(len(seq)):
                    seq[i] = video_name + '/' + str(seq[i])

            for i in range(len(sequence_id_list)):
                if len(sequence_id_list[i]) < self.seq_len:
                    pad_list = sequence_id_list[i]
                    while (len(pad_list) < self.seq_len):
                        pad_list.append(pad_list[-1])
                    sequence_id_list[i] = pad_list

            for sequence_id in sequence_id_list:
                sequence_label_list.append(
                    [label_dict[k.split('/')[-1]] for k in sequence_id])

            return sequence_id_list, sequence_label_list

    def make_sequence(self):
        """make sequence

        Returns:
            seq_id_list (list): explained in :method `make_sequence_id_list`
            seq_label_list (list): explained in :method `make_sequence_id_list`
        """
        seq_id_list = list()
        seq_label_list = list()
        video_list = self.get_video_list()
        for video in video_list:
            txt_path = os.path.join(self.label_root, video + '.txt')
            label_dict = self.get_txt_contents(txt_path)
            crt_id_list, crt_label_list = self.make_sequence_id_list(
                label_dict, video)
            seq_id_list += crt_id_list
            seq_label_list += crt_label_list

        return seq_id_list, seq_label_list

    def open_h5(self):
        for feat_name in ["audio", "spatial"]:
            self.feat_map[feat_name] = h5py.File(
                os.path.join(self.feat_root, feat_name + '_features.h5'), 'r')

    def close_h5(self):
        for feat_name in ["audio", "spatial"]:
            self.feat_map[feat_name].close()

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx):
        self.open_h5()
        seq_list = self.sequence_list[idx]
        lb_list = self.label_list[idx]
        assert len(seq_list) == self.seq_len == len(lb_list)

        aud_feat = list()
        seq_feat = list()
        seq_label = list()

        for seq_name, label in zip(seq_list, lb_list):
            aud_name = seq_name.replace("_left", "").replace("_right", "")
            aud_feature = np.asarray(self.feat_map["audio"][aud_name.split("/")[0]][aud_name.split("/")[1]])
            seq_feature = np.asarray(self.feat_map["spatial"][seq_name.split("/")[0]][seq_name.split("/")[1]])
            aud_feature_normalized = (aud_feature - aud_feature.mean()) / aud_feature.std()
            seq_feature_normalized = (seq_feature - seq_feature.mean()) / seq_feature.std()

            aud_feat.append(aud_feature_normalized)
            seq_feat.append(seq_feature_normalized)
            seq_label.append(label)

        aud_feat = np.asarray(aud_feat)
        seq_feat = np.asarray(seq_feat)
        seq_label = np.asarray(seq_label)

        self.close_h5()
        return seq_feat, aud_feat, seq_label


class SequenceData_2(Dataset):
    def __init__(self, feat_root, integrated_label_file, seq_len, task, mode, pad_mode='repeat_last'):
        # Load integrated label file
        with open(integrated_label_file, 'r') as f:
            self.label_data = json.load(f)

        self.feat_root = feat_root
        self.seq_len = seq_len
        self.task = task
        self.pad_mode = pad_mode
        # Generate sequence list based on mode
        self.sequence_list = self._generate_sequence_list(mode)

    def _generate_sequence_list(self, mode):
        sequence_list = []
        # Filter sequences based on mode (train, val, test)
        sequence_list = []
        for video_name, annotations in self.label_data.items():
            for frame_id, label in annotations:
                sequence_list.append((video_name, frame_id, label))
        return sequence_list

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx):

        seq_name, frame_id, label = self.sequence_list[idx]
        aud_name = seq_name.replace("_left", "").replace("_right", "")
        # Assuming feature paths are structured as feat_root/task/video_name/frame_id.npy

        # Load audio feature
        aud_feat_path = os.path.join(self.feat_root, "audio", aud_name, f"{frame_id}.npy")
        aud_feature = torch.from_numpy(np.load(aud_feat_path)).float().to("cuda:0")
        aud_feature_normalized = (aud_feature - aud_feature.mean()) / aud_feature.std()

        # Load sequence feature
        seq_feat_path = os.path.join(self.feat_root, "spatial", seq_name, f"{frame_id}.npy")
        seq_feature = torch.from_numpy(np.load(seq_feat_path)).float().to("cuda:0")
        seq_feature_normalized = (seq_feature - seq_feature.mean()) / seq_feature.std()

        # Convert label to tensor
        if isinstance(label, list):  # For tasks with multiple labels per frame
            label = torch.tensor(label).float()
        else:  # For tasks with single label per frame
            label = torch.tensor([label]).float()

        return aud_feature_normalized, seq_feature_normalized, label



def integrate_and_save_labels(label_root, mode, task, output_file):
    integrated_labels = {}
    mode_path = os.path.join(label_root, mode)
    video_list = [x.split('.')[0] for x in sorted(os.listdir(mode_path)) if x.endswith('.txt')]
    for video in video_list:
        txt_path = os.path.join(mode_path, video + '.txt')
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            video_labels = []
            for i, line in enumerate(lines):
                if i == 0: continue  # Skip header
                frame_id = f'{i:05d}'
                # Assuming 'task' variable is defined outside this function. Adjust accordingly.
                if task == 'va' or task == 'au':
                    values = [float(value) for value in line.strip('\n').split(',')]
                elif task == 'expr':
                    values = int(line.strip('\n'))
                video_labels.append((frame_id, values))
            integrated_labels[video] = video_labels
    with open(output_file, 'w') as f:
        json.dump(integrated_labels, f, indent=1)


if __name__ == '__main__':
    task = 'va'
    label_root = 'C:/Users/hms/Desktop/Code/dataset/6th ABAW Annotations/VA_Estimation_Challenge'
    output_file = 'C:/Users/hms/Desktop/Code/dataset/6th ABAW Annotations/VA_Estimation_Challenge/{}_integrated_labels.json'.format(task)

    # Train 데이터에 대한 통합 파일 저장
    train_output_file = os.path.join(label_root, 'integrated_train_labels.json')
    integrate_and_save_labels(label_root, 'Train_Set', task, train_output_file)

    # Validation 데이터에 대한 통합 파일 저장
    validation_output_file = os.path.join(label_root, 'integrated_validation_labels.json')
    integrate_and_save_labels(label_root, 'Validation_Set', task, validation_output_file)


