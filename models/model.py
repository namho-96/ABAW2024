import os
import timm
import torch.nn as nn
from transformers import AutoModel, AutoConfig, Wav2Vec2FeatureExtractor
from .custom_transformer import BaseModel, VAmodel, BaseModel2, BaseModel3, DeepMixAttention


def load_model(config_module):
    # num_classes 지정
    
    if config_module.mode == 'train': # For train model(transformer, visual extractor)
        if config_module.data_type == 'spatial':    # Fine-tune
            model = load_visual_model(config_module.visual_model, train=True)
        elif config_module.data_type == 'temporal':
            pass
        elif config_module.data_type == 'multimodal':
            model = load_transformer_model(config_module)
        return model
        
    elif config_module.mode == 'predict':     # For inference
        transformer_model = load_transformer_model(config_module)
        return transformer_model
        
    elif config_module.mode == 'extract':       # For feature extract
        if config_module.data_type == 'audio':
            model = load_audio_model(config_module.audio_model)
        elif config_module.data_type == 'spatial':
            model = load_visual_model(config_module.visual_model, train=False)
        elif config_module.data_type == 'temporal':
            pass
            #model = load_temporal_model(config_module.visual_model, train=False)
        return model        
        
def load_audio_model(model_path):
    if not os.path.isfile(model_path): # 모델 파일이 없으면 다운로드
        model_path = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
    model = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    return model
    
def load_visual_model(config_module, model_name, train=False):
    if model_name == 'swin':
        # Swin Transformer 모델 로드
        model = AutoModel.from_pretrained(model_name)
    else:
        # timm 라이브러리를 사용한 다른 모델 로드
        model = timm.create_model(model_name, pretrained=train)

    first_conv_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv_layer = module
            break
    if first_conv_layer is not None:
        new_first_conv = nn.Conv2d(20, first_conv_layer.out_channels, 
                                   kernel_size=first_conv_layer.kernel_size, 
                                   stride=first_conv_layer.stride, 
                                   padding=first_conv_layer.padding, 
                                   bias=first_conv_layer.bias)
        new_first_conv.weight.data = first_conv_layer.weight.data.mean(dim=1, keepdim=True).expand_as(new_first_conv.weight) / 20.0
        if 'features' in dir(model):
            model.features[0] = new_first_conv
        else:
            model.conv1 = new_first_conv

    # 모델의 마지막 층을 num_classes에 맞게 조정
    if 'fc' in dir(model):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, config_module.num_classes)
    elif 'classifier' in dir(model):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, config_module.num_classes)
        
    return model
    
def load_transformer_model(config_module):
    if config_module.model_name == 'base':
        model = BaseModel(config_module.num_features, config_module.num_head, config_module.num_classes)
    elif config_module.model_name == 'va':
        model = VAmodel(config_module)
    elif config_module.model_name == 'base2':
        model = BaseModel2(config_module.num_features, config_module.num_head, config_module.num_classes)
    elif config_module.model_name == 'base3':
        model = BaseModel3(config_module.num_features, config_module.num_head, config_module.num_classes)
    elif config_module.model_name == 'dma':
        model = DeepMixAttention(config_module)
    else:
        raise Exception("Wrong config_module.model_name")
    return model
