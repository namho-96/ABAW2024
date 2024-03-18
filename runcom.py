import os

"""
Train Code
"""
# AU

# os.system('python main.py --device 0 --task au --prefix tiny_test --num_features 768 --model_name dma3 --num_classes 12 --train_name au_final_test_fold1_auf --lr 0.001 --min_lr 0.0 --batch_size 32 --num_workers 8 \
#    --model_arch self mix forward neck head --dropout 0.3 --droppath 0.4 --mixup --label_path \"../dataset/Fold Annotations\" --fold 1')

# EXPR

# os.system('python main.py --device 0 --task expr --prefix tiny_test --num_features 768 --model_name dma3 --num_classes 8 --train_name expr_final_test_fold5_exprf --lr 0.001 --min_lr 0.0 --batch_size 32 --num_workers 8 \
#    --model_arch self mix forward neck head --dropout 0.3 --droppath 0.4 --mixup --label_path \"../dataset/Fold Annotations\" --fold 5')

# VA

# os.system('python main.py --device 0 --task va --prefix tiny --num_features 768 --model_name dma3 --num_classes 1 --train_name va_test --lr 0. --batch_size 32 --num_workers 8 \
#    --model_arch self mix forward neck head --dropout 0.2 --droppath 0.4 --mixup --scheduler warmup --epochs 30 --eta_max 0.001')


"""
Prediction Code
"""
# AU
# os.system("python main.py --device 0 --task au --prefix tiny_test --mode predict --num_features 768 --model_name dma3 --num_classes 12 --model_arch self mix forward neck head --dropout 0.3 --droppath 0.4 --phase val --tag fold1_check --fold 1 --label_path \"../dataset/Fold Annotations\" --model_path /mnt/kbsrnd1/ABAW/ABAW2024/output/au/fold_1/best_model_epoch_42_loss_0.1667.pth")

# EXPR
# os.system("python main.py --device 0 --task expr --mode predict --model_name dma3 --num_classes 8 --model_arch self mix forward neck head --dropout 0.3 --droppath 0.4 --phase test --tag test --model_path /mnt/kbsrnd1/ABAW/ABAW2024/output/expr/2024-03-13_13-50-55/best_model_epoch_28_performance_0.6663.pth")

# VA
# os.system("python main.py --device 0 --task va --prefix tiny --mode predict --model_name dma3 --num_classes 2 --model_arch self mix forward neck head --dropout 0.1 --droppath 0.2 --phase test --tag test --model_path /mnt/kbsrnd1/ABAW/ABAW2024/output/va/test1/best_model_epoch_64_performance_0.5356.pth")

"""
Evaluation Code
"""
# AU
# os.system("python evaluate.py --pred-path output/prediction/au/val/fold1_check --label-path \"../dataset/Fold Annotations/AU_Detection_Challenge/fold_1/Validation_Set\" --task au")

# EXPR
# os.system("python evaluate.py --pred-path output/prediction/expr/val/test --label-path \"../dataset/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set\" --task expr")

# VA
# os.system("python evaluate.py --pred-path output/prediction/va/val/test --label-path \"../dataset/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set\" --task va")