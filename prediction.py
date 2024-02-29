import os
import h5py
import torch
import torch.nn as nn
import natsort as nt
import numpy as np
from tqdm import tqdm
from models.model import load_model

def select_data_phase(data_name, phase):
    data_path = {
        'au': 'AU_Detection_Challenge',
        'expr': 'EXPR_Recognition_Challenge',
        'va': 'VA_Estimation_Challenge'
    }
    phase_lists = {
        'val'  : 'Validation_Set',
        'test' : 'Test_Set'
    }
    
    return data_path[data_name], phase_lists[phase]

@torch.no_grad()
def predict_function(config):    
    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    model = load_model(config)
    checkpoint = torch.load(config.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    task_name, phase_name = select_data_phase(config.data_name, config.phase)
    data_path = os.path.join(config.label_path, task_name, phase_name)
    
    audio_features = h5py.File(os.path.join(config.feat_path, 'audio_features.h5'), 'r')
    spatial_features = h5py.File(os.path.join(config.feat_path, 'spatial_features.h5'), 'r')
    
    output_path = f"output/prediction/{config.data_name}/{config.phase}/{config.tag}"
    os.makedirs(output_path, exist_ok=True)
    
    if config.data_name == 'au':    
        m = nn.Sigmoid()
    
    txt_lists = nt.natsorted(os.listdir(data_path))

    for txt_name in tqdm(txt_lists):
        full_txt_name = os.path.join(data_path, txt_name)
        full_output_name = os.path.join(output_path, txt_name)
        with open(full_txt_name, 'r') as f:
            lines = f.readlines()
            
        start_index = 1
        end_index = len(lines)
        
        vid_name = os.path.splitext(txt_name)[0]
        aud_name = vid_name.replace("_left", "").replace("_right", "")
        
        aud_inputs = []
        spatial_inputs = []
        
        with open(full_output_name, "a") as f:
        
            if config.data_name == 'va':
                f.write('valence,arousal\n')
            elif config.data_name == 'au':
                f.write('AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26\n')                
            elif config.data_name == 'expr':
                f.write('Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n')                
                
            for i in range(start_index, end_index):            
                filename = f"{i:05d}"
                
                aud_inputs.append(torch.as_tensor(audio_features[aud_name][filename]).unsqueeze(0))
                spatial_inputs.append(torch.as_tensor(spatial_features[vid_name][filename]).unsqueeze(0))

                if len(aud_inputs) == 100 or i == end_index - 1:
                    vid = torch.cat(spatial_inputs).unsqueeze(0).to(device)
                    aud = torch.cat(aud_inputs).unsqueeze(0).to(device)
                    outputs = model(vid, aud)
                    
                    if config.data_name == 'va':
                        predicted = np.hstack(outputs[0].cpu().numpy(), outputs[1].cpu().numpy())   
                        np.savetxt(f, predicted, delimiter=',', fmt='%f')
                    else:
                        predicted = outputs.reshape(-1, config.num_classes)
                        
                        if config.data_name == 'au':
                            predicted = m(predicted)
                            predicted = predicted > 0.5
                            
                        predicted = predicted.cpu().numpy()
                        np.savetxt(f, predicted, delimiter=',', fmt='%d')                    
                    
                    aud_inputs = []
                    spatial_inputs = []
                    
    audio_features.close()
    spatial_features.close()