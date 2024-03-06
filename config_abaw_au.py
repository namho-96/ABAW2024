import argparse


def get_args():
    parser = argparse.ArgumentParser(description="ABAW2024 Configuration")

    # Environments
    parser.add_argument('--feat_path', type=str, default='../dataset/feature', help='Path to features directory')
    parser.add_argument('--label_path', type=str, default='../dataset/6th ABAW Annotations', help='Path to labels directory')
    parser.add_argument('--device', type=int, default=0, help='Device to run the model on')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')

    # Data
    parser.add_argument('--data_name', type=str, default='va', help='Name of the data')
    parser.add_argument('--data_type', type=str, default='multimodal', help='Type of the data')

    # Model
    parser.add_argument('--model_name', type=str, default='dma', help='Name of the model')
    parser.add_argument('--num_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--sq_len', type=int, default=100, help='Sequence length')
    parser.add_argument('--num_features', type=int, default=768, help='Number of input features')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')

    # Training
    parser.add_argument('--train_name', type=str, default='mixup_residual_mix_mix_forward x 1', help='Name of the training for Wandb')
    parser.add_argument('--mode', type=str, default='train', help='Mode of the training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer to use')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay for optimizer')
    parser.add_argument('--hidden_size', nargs='+', type=int, default=[256, 128, 64], help='Hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--droppath', type=float, default=0.2, help='DropPath rate')
    parser.add_argument('--mixup', default=True, help='Use mixup')
    parser.add_argument('--resume', type=str, default=None, help="Path to resume model. If not specified or 'none', training starts from scratch.")

    # Model architecture
    parser.add_argument('--model_arch', nargs='+', type=str, default=["self", "mix", "mix", "forward", "neck", "va"], help='Model architecture')
    parser.add_argument('--vis', default=False, help='Save model architecture visualization')

    args = parser.parse_args()
    return args

