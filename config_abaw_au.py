
# Environments
feat_path = '../dataset/feature'
label_path = '../dataset/6th ABAW Annotations'
device = 0
num_workers = 0


# Data
data_name = 'va'
data_type = 'multimodal'


# Model
model_name = 'dma'
num_head = 8            # 8개
sq_len = 100
num_features = 768
num_classes = 2


# Training
mode = 'train'
batch_size = 256
epochs = 100
lr = 0.001               # 0.001
optimizer = 'adamw'  # adam, adamw, sgd
momentum = 0.9
weight_decay = 0.00001
hidden_size = [256, 128, 64]
dropout = 0.1
resume = False
resume_path = "./output/va/ccc_lr_0.001_head_8_new_model_swin_Feature/best_model_epoch_8_performance_0.3937.pth"

# Model architecture
model_arch = ["self", "mix", "forward", "forward", "forward", "neck", data_name]
vis = True          # save model architecture



