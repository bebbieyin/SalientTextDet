from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config['TRAIN']['EPOCHS'] * n_iter_per_epoch)
    warmup_steps = int(config['TRAIN']['WARMUP_EPOCHS'] * n_iter_per_epoch)
    decay_steps = int(config['TRAIN']['LR_SCHEDULER']['DECAY_EPOCHS'] * n_iter_per_epoch)

    lr_scheduler = None
    if config['TRAIN']['LR_SCHEDULER']['NAME'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=float(config['TRAIN']['MIN_LR']),
            warmup_lr_init=float(config['TRAIN']['WARMUP_LR']),
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

    elif config['TRAIN']['LR_SCHEDULER']['NAME'] == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=float(config['TRAIN']['LR_SCHEDULER']['DECAY_RATE']),
            warmup_lr_init=float(config['TRAIN']['WARMUP_LR']),
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler