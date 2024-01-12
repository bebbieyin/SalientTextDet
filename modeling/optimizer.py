import torch

AVAILABLE_OPTIMIZERS = ["adam","adamw","sgd"]

def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """

    opt_lower = config['TRAIN']['OPTIMIZER']['NAME'].lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                              momentum=float(config['TRAIN']['OPTIMIZER']['MOMENTUM']), 
                              nesterov=True,
                              lr=float(config['TRAIN']['BASE_LR']), 
                              weight_decay=float(config['TRAIN']['WEIGHT_DECAY']))
    elif opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(params = model.parameters(), 
                                eps=float(config['TRAIN']['OPTIMIZER']['EPS']), 
                                lr=float(config['TRAIN']['BASE_LR']), 
                                weight_decay=float(config['TRAIN']['WEIGHT_DECAY']))

    elif opt_lower == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=float(config['TRAIN']['BASE_LR']))

    else:
        raise ValueError(f"Optimizer '{opt_lower}' does not match any of the available choices: {', '.join(AVAILABLE_OPTIMIZERS)}")

    return optimizer

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
