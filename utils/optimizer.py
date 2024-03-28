import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
def build_optimizer(config, model):
    model = model.module if hasattr(model, 'module') else model

    optimizer = optim.AdamW(model.parameters(), lr=config.TRAIN.LR,
                            weight_decay=config.TRAIN.WEIGHT_DECAY,
                            betas=(0.9, 0.98), eps=1e-8, )
    return optimizer

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=config.TRAIN.LR / 100,
        warmup_lr_init=0,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    return lr_scheduler