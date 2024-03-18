import os
import h5py
import torch
import torch.nn as nn
import natsort as nt
import numpy as np
from tqdm import tqdm
from models.model import load_model


def select_data_phase(task, phase):
    if phase == 'val':
        data_path = {
            'au': 'AU_Detection_Challenge',
            'expr': 'EXPR_Recognition_Challenge',
            'va': 'VA_Estimation_Challenge'
        }

        return data_path[task], 'Validation_Set'

    elif phase == 'test':
        data_path = {
            'au': 'CVPR_6th_ABAW_AU_test_set_sample.txt',
            'expr': 'CVPR_6th_ABAW_Expr_test_set_sample.txt',
            'va': 'CVPR_6th_ABAW_VA_test_set_sample.txt'
        }

        name_path = {
            'au': 'Action_Unit_Detection_Challenge_test_set_release.txt',
            'expr': 'Expression_Recognition_Challenge_test_set_release.txt',
            'va': 'Valence_Arousal_Estimation_Challenge_test_set_release.txt'
        }
        return data_path[task], name_path[task]


def predict_function(config):
    if config.phase == "val":
        run_validation(config)
    elif config.phase == "test":
        run_test(config)


@torch.no_grad()
def run_test(config):
    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    model = load_model(config)
    checkpoint = torch.load(config.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    format_txt, name_txt = select_data_phase(config.task, config.phase)

    format_path = os.path.join(config.test_path, "prediction_files_format", format_txt)
    name_path = os.path.join(config.test_path, "names_of_videos_in_each_test_set", name_txt)

    with open(format_path, "r") as f:
        lines = f.readlines()

    shit_dict = {}

    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        video_name, jpg_name = line.strip().replace(",", "").split("/")
        if video_name not in shit_dict:
            shit_dict[video_name] = 1
        else:
            shit_dict[video_name] += 1

    with open(name_path, "r") as f:
        video_list = f.readlines()

    audio_features = h5py.File(os.path.join(config.feat_path, 'audio_features_test.h5'), 'r')
    # spatial_features = h5py.File(os.path.join(config.feat_path, f'spatial_features_{config.task}_{config.prefix}.h5'), 'r')
    spatial_features = h5py.File(os.path.join(config.feat_path, f'spatial_features_expr_tiny_test.h5'), 'r')

    output_path = f"output/prediction/{config.task}/{config.phase}/{config.tag}"
    os.makedirs(output_path, exist_ok=True)

    prediction_path = os.path.join(output_path, "predictions.txt")

    with open(prediction_path, "w") as f:

        if config.task == 'va':
            f.write('image_location,valence,arousal\n')
        elif config.task == 'au':
            f.write('image_location,AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26\n')
        elif config.task == 'expr':
            f.write('image_location,Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n')

        for vid_name in tqdm(video_list):
            vid_name = vid_name.strip()
            start_index = 1
            end_index = shit_dict[vid_name] + 1
            aud_name = vid_name.replace("_left", "").replace("_right", "")

            full_file_path = []
            aud_inputs = []
            spatial_inputs = []

            for i in range(start_index, end_index):
                filename = f"{i:05d}"

                aud_feature = np.asarray(audio_features[aud_name][filename])
                seq_feature = np.asarray(spatial_features[vid_name][filename])
                full_file_path.append(f"{vid_name}/{filename}.jpg")

                aud_feature_normalized = (aud_feature - aud_feature.mean()) / aud_feature.std()
                seq_feature_normalized = (seq_feature - seq_feature.mean()) / seq_feature.std()

                aud_inputs.append(torch.from_numpy(aud_feature_normalized).unsqueeze(0))
                spatial_inputs.append(torch.from_numpy(seq_feature_normalized).unsqueeze(0))

                if len(aud_inputs) == 100 or i == end_index - 1:
                    vid = torch.cat(spatial_inputs).unsqueeze(0).to(device)
                    aud = torch.cat(aud_inputs).unsqueeze(0).to(device)
                    outputs = model(vid, aud)

                    if config.task == 'va':
                        predicted = np.hstack(
                            [np.expand_dims(np.array(full_file_path), axis=1), outputs[0].reshape(-1, 1).cpu().numpy(),
                             outputs[1].reshape(-1, 1).cpu().numpy()])
                        np.savetxt(f, predicted, fmt='%s,%s,%s')
                    else:
                        predicted = outputs.reshape(-1, config.num_classes)

                        if not config.ensemble:
                            if config.task == 'au':
                                predicted = predicted > 0.5
                            elif config.task == 'expr':
                                _, predicted = predicted.max(1)

                        if config.task == 'au':
                            predicted = predicted.cpu().numpy() if config.ensemble else predicted.cpu().numpy().astype(
                                int)
                            predicted = np.hstack([np.expand_dims(np.array(full_file_path), axis=1), predicted])
                            np.savetxt(f, predicted, fmt='%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s')
                        elif config.task == 'expr':
                            predicted = predicted.cpu().unsqueeze(1).numpy()
                            predicted = np.hstack([np.expand_dims(np.array(full_file_path), axis=1), predicted])
                            if config.ensemble:
                                np.savetxt(f, predicted, fmt='%s,%s,%s,%s,%s,%s,%s,%s,%s')
                            else:
                                np.savetxt(f, predicted, fmt='%s,%s')

                    full_file_path = []
                    aud_inputs = []
                    spatial_inputs = []

    audio_features.close()
    spatial_features.close()


@torch.no_grad()
def run_validation(config):
    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    model = load_model(config)
    checkpoint = torch.load(config.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    task_name, phase_name = select_data_phase(config.task, config.phase)
    if config.fold:
        data_path = os.path.join(config.label_path, task_name, f"fold_{config.fold}", phase_name)
    else:
        data_path = os.path.join(config.label_path, task_name, phase_name)

    audio_features = h5py.File(os.path.join(config.feat_path, 'audio_features_test.h5'), 'r')
    spatial_features = h5py.File(os.path.join(config.feat_path, f'spatial_features_{config.task}_{config.prefix}.h5'),
                                 'r')

    output_path = f"output/prediction/{config.task}/{config.phase}/{config.tag}"
    os.makedirs(output_path, exist_ok=True)

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

        with open(full_output_name, "w") as f:

            if config.task == 'va':
                f.write('valence,arousal\n')
            elif config.task == 'au':
                f.write('AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26\n')
            elif config.task == 'expr':
                f.write('Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n')

            for i in range(start_index, end_index):
                filename = f"{i:05d}"
                aud_feature = np.asarray(audio_features[aud_name][filename])
                seq_feature = np.asarray(spatial_features[vid_name][filename])

                aud_feature_normalized = (aud_feature - aud_feature.mean()) / aud_feature.std()
                seq_feature_normalized = (seq_feature - seq_feature.mean()) / seq_feature.std()

                aud_inputs.append(torch.from_numpy(aud_feature_normalized).unsqueeze(0))
                spatial_inputs.append(torch.from_numpy(seq_feature_normalized).unsqueeze(0))

                if len(aud_inputs) == 100 or i == end_index - 1:
                    vid = torch.cat(spatial_inputs).unsqueeze(0).to(device)
                    aud = torch.cat(aud_inputs).unsqueeze(0).to(device)
                    outputs = model(vid, aud)

                    if config.task == 'va':
                        predicted = np.hstack(
                            [outputs[0].reshape(-1, 1).cpu().numpy(), outputs[1].reshape(-1, 1).cpu().numpy()])
                        np.savetxt(f, predicted, delimiter=',', fmt='%f')
                    else:
                        predicted = outputs.reshape(-1, config.num_classes)

                        if not config.ensemble:
                            if config.task == 'au':
                                predicted = predicted > 0.5
                            elif config.task == 'expr':
                                _, predicted = predicted.max(1)

                        predicted = predicted.cpu().numpy()
                        if config.ensemble:
                            np.savetxt(f, predicted, delimiter=',')
                        else:
                            np.savetxt(f, predicted, delimiter=',', fmt='%d')

                    aud_inputs = []
                    spatial_inputs = []

    audio_features.close()
    spatial_features.close()
