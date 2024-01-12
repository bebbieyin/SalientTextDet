import torch
from modeling.data.data_ecom import ecommercedata
from modeling.data.data_icdar15 import icdar15data

AVAILABLE_DATASETS = ["ICDAR15", "ECOMMERCE"]

def build_ecom_dataloader(data_path, num_data, is_training, batch_size=4, num_workers=8, pin_memory=True, augmentation=None,
                          include_saliency=True):
    dataset = ecommercedata(data_path=data_path, img_nums=num_data, is_train=is_training, include_saliency=include_saliency,augment=augmentation)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        #collate_fn = collate_fn
    )
    return dataloader

def build_icdar15_dataloader(data_path, num_data, batch_size=4, num_workers=8, pin_memory=True, is_training=True, augmentation=None,
                             ignore_donotcare=True):
    dataset = icdar15data(data_path, num_data, is_training, augmentation,ignore_donotcare)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return data_loader

def build_dataloader(config, augmentation):
    
    dataset_name = config['INPUT']['DATASET']

    if dataset_name=="ICDAR15":
        train_data = build_icdar15_dataloader(config['INPUT']['DATA_PATH'], config['INPUT']['TRAIN_NUM_DATA'],
                                        config['TRAIN']['BATCH_SIZE'],config['TRAIN']['NUM_WORKERS'], 
                                        config['TRAIN']['PIN_MEMORY'],is_training=True, augmentation=augmentation,
                                        ignore_donotcare=config['INPUT']['IGNORE_DONT_CARE'])
        
        test_data = build_icdar15_dataloader(config['INPUT']['DATA_PATH'], config['INPUT']['TEST_NUM_DATA'],
                                                config['TRAIN']['BATCH_SIZE'],config['TRAIN']['NUM_WORKERS'], 
                                                config['TRAIN']['PIN_MEMORY'],is_training=False, augmentation=augmentation,
                                                ignore_donotcare=config['INPUT']['IGNORE_DONT_CARE'])
    elif dataset_name=="ECOMMERCE":
        train_data = build_ecom_dataloader(data_path=config['INPUT']['DATA_PATH'], num_data=config['INPUT']['TRAIN_NUM_DATA']+config['INPUT']['TEST_NUM_DATA'],
                                            batch_size=config['TRAIN']['BATCH_SIZE'],num_workers=config['TRAIN']['NUM_WORKERS'], 
                                            pin_memory=config['TRAIN']['PIN_MEMORY'],include_saliency=config['INPUT']['SALIENCY'],
                                            is_training=True,augmentation=augmentation)
    
        test_data =  build_ecom_dataloader(data_path=config['INPUT']['DATA_PATH'], num_data=config['INPUT']['TRAIN_NUM_DATA']+config['INPUT']['TEST_NUM_DATA'],
                                            batch_size=config['TRAIN']['BATCH_SIZE'],num_workers=config['TRAIN']['NUM_WORKERS'], 
                                            pin_memory=config['TRAIN']['PIN_MEMORY'],include_saliency=config['INPUT']['SALIENCY'],
                                            is_training=False,augmentation=augmentation)
    else:
        raise ValueError(f"Dataset '{dataset_name}' does not match any of the available choices: {', '.join(AVAILABLE_DATASETS)}")

    
    return train_data, test_data
