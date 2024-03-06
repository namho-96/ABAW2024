import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.metric import *
from sklearn.metrics import f1_score

au_names = {
    0 : "AU1",
    1 : "AU2",
    2 : "AU4",
    3 : "AU6",
    4 : "AU7",
    5 : "AU10",
    6 : "AU12",
    7 : "AU15",
    8 : "AU23",
    9 : "AU24",
    10 : "AU25",
    11 : "AU26"
}

expression_names = {
    0 : "Neutral",
    1 : "Anger",
    2 : "Disgust",
    3 : "Fear",
    4 : "Happiness",
    5 : "Sadness",
    6 : "Surprise",
    7 : "Other"
}

def evaluate_function(args):
    txt_lists = os.listdir(args.pred_path)
    
    if args.task == 'va':
        prediction_valence = []
        prediction_arousal = []
        gt_valence = []
        gt_arousal = []
    else:    
        prediction = []
        gt = []    
    
    for txt_name in tqdm(txt_lists):
        pred_txt_path = os.path.join(args.pred_path, txt_name)
        label_txt_path = os.path.join(args.label_path, txt_name)
        
        pred_df = pd.read_csv(pred_txt_path, sep=',').to_numpy()
        label_df = pd.read_csv(label_txt_path, sep=',').to_numpy()
        
        mask = ~np.all(label_df == -5, axis=1) if args.task == 'va' else ~np.all(label_df == -1, axis=1)
        
        label_df = label_df[mask]
        pred_df = pred_df[mask]
        
        if args.task == 'va':
            prediction_valence.extend(pred_df[:, 0])
            prediction_arousal.extend(pred_df[:, 1])
            gt_valence.extend(label_df[:, 0])
            gt_arousal.extend(label_df[:, 1])
        else:
            prediction.extend(pred_df)
            gt.extend(label_df)
            
    if args.task == 'va':
        valence_score = CCC_np(prediction_valence, gt_valence)
        arousal_score = CCC_np(prediction_arousal, gt_arousal)
        avg_performance = 0.5 * (valence_score + arousal_score)
        print(f"*Average Score : {avg_performance}")
        print(f"*Valence Score : {valence_score}")
        print(f"*Arousal Score : {arousal_score}")
    else:    
        f1s = f1_score(gt, prediction, average=None, zero_division=1)
        avg_performance = f1_score(gt, prediction, average='macro', zero_division=1)
        print(f"* Average Performance : {avg_performance}")
        
        if args.task == 'au':
            for i in range(12):
                print(f"* Performance {au_names[i]} : {f1s[i]}")                
        elif args.task == 'expr':
            for i in range(8):
                print(f"* Performance {expression_names[i]} : {f1s[i]}")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Learning Model Configuration')
    parser.add_argument('--pred-path', type=str, required=True, help='Prediction txt path')
    parser.add_argument('--label-path', type=str, required=True, help='Ground Truth txt path')
    parser.add_argument('--task', type=str, required=True, help='Task name')
    args = parser.parse_args()
    
    evaluate_function(args)
