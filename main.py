import model.TClip as tbaclip
import clip
import torch
import os
import argparse
from utils.config import get_config
from dataSets.build import build_dataloader
from utils.tools import AverageMeter

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/zero_shot/eval/hmdb/tba_clip_hmdb51_base.yaml')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--num_frames', type=int)
    parser.add_argument('--if_teacher', type=bool)
    parser.add_argument('--instructionFT', type=bool)
    parser.add_argument('--trans_frames', type=bool)
    args = parser.parse_args()
    config = get_config(args)
    return args, config

def main(config):

    train_data, val_data, train_loader, val_loader = build_dataloader(config)
    class_names = [class_name for i, class_name in val_data.classes]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tbaclip.returnCLIP(config,class_names,device)
    acc1 = validate(val_loader, model,config)
    print(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")

@torch.no_grad()
def validate(val_loader,model,config):
    b = val_loader.batch_size
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            # print('filename:',batch_data['filename'])
            images = batch_data['data']
            label_id = batch_data['label']
            # print(label_id[0])
            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            image_input = []
            for image in images:
                image = image.cpu().numpy()
                # output = model(image)
                # similarity = output
                # tot_similarity += similarity
                image_input.append(image)
            image_input = [item for sublist in image_input for item in sublist]
            if config.TRAINER.TRANS_FRAMES:
                tot_similarity = model(image_input)
            else:
                for image in image_input:
                    similarity = model(image)
                    tot_similarity += similarity
            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)
            acc1, acc5 = 0, 0
            for i in range(b):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1
            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            if idx % 200 == 0:
                print( f'Test: [{idx}/{len(val_loader)}]\t'
                        f'Acc@1: {acc1_meter.avg:.3f}\t')
        print(f'Acc@1: {acc1_meter.avg:.3f}\t'
              f'Acc@5: {acc5_meter.avg:.3f}\t')
        # Check if the file is empty
        if os.stat(config.OUT_PUT).st_size == 0:
            with open(config.OUT_PUT, 'a') as f:
                # Write the column names
                f.write('Model,Num_Frames,Acc1,Acc5,Dataset\n')
        with open(config.OUT_PUT, 'a') as f:
            f.write(f'{config.MODEL.ARCH},{config.NUM_FRAMES},{acc1_meter.avg:.3f},{acc5_meter.avg:.3f},{config.DATA.DATASET}\n')
        return acc1_meter.avg

if __name__ == '__main__':
    args, config = parse_option()
    main(config)