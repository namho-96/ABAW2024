import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, ViTImageProcessor, ViTModel
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
from utils import createDirectory
from PIL import Image
from tqdm import tqdm
import natsort as nt

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def extract_feature(data_type, device=torch.device("cpu")):
    if data_type == "audio":
        model_path = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        model = Wav2Vec2ForSpeechClassification.from_pretrained(model_path).to(device)
        sampling_rate = feature_extractor.sampling_rate
        data_path = "../dataset/audio"
        save_base_path = "../dataset/feature/audio"
        createDirectory(save_base_path)
        video_lists = os.listdir(data_path)
        for video_name in tqdm(video_lists):
            full_video_path = os.path.join(data_path, video_name)
            save_video_path = os.path.join(save_base_path, video_name)
            createDirectory(save_video_path)
            audio_lists = nt.natsorted(os.listdir(full_video_path))
                        
            inputs = []
            save_feature_paths = []            
            for idx, audio_name in enumerate(audio_lists):                
                base_name = os.path.splitext(audio_name)[0]                
                full_audio_name = os.path.join(full_video_path, audio_name)                
                save_feature_path = os.path.join(save_video_path, base_name)
                
                input = speech_file_to_array_fn(full_audio_name, sampling_rate)
                
                inputs.append(input)
                save_feature_paths.append(save_feature_path)
                if idx % 1024 == 1023 or idx == len(audio_lists) - 1:
                    inputs = feature_extractor(inputs, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
                    inputs = {key: inputs[key].to(device) for key in inputs}
                    
                    # 모델을 사용하여 이미지의 feature를 추출합니다.
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # 추출된 feature를 numpy 배열로 변환합니다.
                    features = outputs.last_hidden_states.cpu().numpy()
                    
                    for si_path, feature in zip(save_feature_paths, features):
                        # numpy 배열을 npy 파일로 저장합니다.
                        np.save(si_path + ".npy", feature)
                        
                    inputs = []
                    save_feature_paths = []                    
            
    elif data_type == 'spatial': # Hugging Face의 모델과 토크나이저를 로드합니다.
        model_name = "trpakov/vit-face-expression"
        feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTModel.from_pretrained(model_name)
        model.to(device)
        base_path = "../dataset/image"
        save_base_path = "../dataset/feature/spatial"
        createDirectory(save_base_path)
        
        folder_list = os.listdir(base_path)
        
        for folder_name in folder_list:
            full_folder_name = os.path.join(base_path, folder_name)
            video_lists = os.listdir(full_folder_name)
            
            for video_name in tqdm(video_lists):
                full_video_name = os.path.join(full_folder_name, video_name)
                image_lists = nt.natsorted(os.listdir(full_video_name))
                save_video_path = os.path.join(save_base_path, video_name)
                createDirectory(save_video_path)
                inputs = []
                save_image_paths = []
                
                for idx, image_name in enumerate(image_lists):
                    if image_name == ".DS_Store":
                        continue            
                    base_name = os.path.splitext(image_name)[0]
                    save_image_path = os.path.join(save_video_path, base_name)
                    
                    full_image_path = os.path.join(full_video_name, image_name)
                    input = Image.open(full_image_path)

                    # 이미지를 모델에 입력할 수 있는 형태로 전처리합니다.
                    inputs.append(input)
                    save_image_paths.append(save_image_path)
                    if idx % 1024 == 1023 or idx == len(image_lists) - 1:
                        inputs = feature_extractor(images=inputs, return_tensors="pt").to(device)
                        
                        # 모델을 사용하여 이미지의 feature를 추출합니다.
                        with torch.no_grad():
                            outputs = model(**inputs)
                        
                        # 추출된 feature를 numpy 배열로 변환합니다.
                        features = outputs.pooler_output.cpu().numpy()
                        
                        for si_path, feature in zip(save_image_paths, features):
                            # numpy 배열을 npy 파일로 저장합니다.
                            np.save(si_path + ".npy", feature)
                            
                        inputs = []
                        save_image_paths = []
    elif data_type == 'temporal':
        pass
    
    
    print(f"Extract {data_type} feature finish!")
    
    
    
class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        #if not return_dict:
        #    output = (logits,) + outputs[2:]
        #    return ((loss,) + output) if loss is not None else output
        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            last_hidden_states=hidden_states
        )
        
@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_states: torch.FloatTensor = None