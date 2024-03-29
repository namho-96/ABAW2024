import os
from .abaw import SequenceData
from torch.utils.data import DataLoader


def setup_dataset(config):
    data_path = {
        'au': 'AU_Detection_Challenge',
        'expr': 'EXPR_Recognition_Challenge',
        'va': 'VA_Estimation_Challenge'
    }

    if config.fold:
        data_path = os.path.join(config.label_path, data_path.get(config.task, ''), f'fold_{config.fold}')
    else:
        data_path = os.path.join(config.label_path, data_path.get(config.task, ''))

    if config.data_type == 'multimodal':
        dataset_train = SequenceData(config.feat_path, data_path, config.sq_len, config.task, 'train', config.prefix)
        dataset_val = SequenceData(config.feat_path, data_path, config.sq_len, config.task, 'val', config.prefix)
    else:
        raise ValueError(f'Wrong Data type : {config.data_type}')

    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers, drop_last=False)

    return dataloader_train, dataloader_val

