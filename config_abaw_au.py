
# Environments
feat_path = '../dataset/feature'
label_path = '../dataset/6th ABAW Annotations'
device = 0
num_workers = 4


# Data
data_name = 'va'
data_type = 'multimodal'


# Model
model_name = 'va'
num_head = 4
sq_len = 100
num_features = 768
num_classes = 2


# Training
mode = 'train'
batch_size = 256
epochs = 100
lr = 0.01
optimizer = 'adamw'  # adam, adamw, sgd
momentum = 0.9
weight_decay = 0.00001
hidden_size = [256, 128, 64]
resume = False
resume_path = "C:/Users/hms/Desktop/Code/ABAW2024/output/va/2024-02-21_16-04-06/best_model_epoch_0_performance_1.0000.pth"



