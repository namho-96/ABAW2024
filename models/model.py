from .custom_transformer import BaseModel, VAmodel, BaseModel2, BaseModel3, DeepMixAttention


def load_model(config_module):
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


