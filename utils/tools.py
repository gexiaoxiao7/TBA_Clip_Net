import numpy
import torch.distributed as dist
import torch
import clip
import os


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def epoch_saving(config, epoch, model, max_accuracy, optimizer, lr_scheduler, working_dir, is_best):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(working_dir, f'ckpt_epoch_{epoch}.pth')
    print(f"{save_path} saving......")
    torch.save(save_state, save_path)
    print(f"{save_path} saved !!!")
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        print(f"{best_path} saved !!!")




