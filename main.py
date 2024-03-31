import model.TClip as tbaclip
import clip
import torch
import os
import argparse
import torch.nn.functional as F
import time
from tqdm import tqdm
from utils.config import get_config
from dataSets.build import build_dataloader
from utils.tools import AverageMeter, clip_classifier, build_cache_model, pre_load_features, split_dataset
from utils.tools import cls_acc, search_hp
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/zero_shot/eval/hmdb/tba_clip_hmdb51_base.yaml')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--num_frames', type=int)
    parser.add_argument('--shots', type=int)
    parser.add_argument('--temporal_pooling', type=str)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--if_teacher', type=int)
    parser.add_argument('--output', type=str)
    parser.add_argument('--load_cache', type=int)
    parser.add_argument('--load_pre_feat', type=int)
    parser.add_argument('--zs', type=int)
    parser.add_argument('--cache_size', type=int)
    args = parser.parse_args()
    config = get_config(args)
    return args, config



def run_tip_adapter(config, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    print("\n-------- Searching hyperparameters on the val set. --------")
    # Zero-shot CLIP
    clip_logits = (100. * val_features @ clip_weights.T).softmax(dim=-1)
    acc1, acc5 = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy1: {:.2f}. , accuracy5: {:.2f} ****\n".format(acc1,acc5))

    # Tip-Adapter
    beta, alpha = config.TIP_ADAPTER.INIT_BETA, config.TIP_ADAPTER.INIT_ALPHA

    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.to(affinity.device)

    tip_logits = clip_logits + cache_logits * alpha
    acc1,acc5 = cls_acc(tip_logits, val_labels)

    print("**** Tip-Adapter's val accuracy1: {:.2f}. accuracy5: {:.2f}****\n".format(acc1,acc5))


    # Search Hyperparameters
    best_beta, best_alpha = search_hp(config, cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = (100. * test_features @ clip_weights.T).softmax(dim=-1)
    acc1, acc5 = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy1: {:.2f}. accuracy5:{:.2f}****\n".format(acc1,acc5))
    with open(config.OUTPUT, 'a') as f:
        f.write(
            f'Zero-shot Clip,{config.MODEL.ARCH},{config.DATA.IF_TEACHER},{config.DATA.NUM_FRAMES},{acc1:.3f},{acc5:.3f},{config.DATA.DATASET},'
            f'0,{config.PREFIX},0,{config.TEMPORAL_POOLING}\n')
    # Tip-Adapter
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values.to(affinity.device)

    tip_logits = clip_logits + cache_logits * best_alpha
    acc1, acc5 = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy1: {:.2f}. accuracy5: {:.2f} ****\n".format(acc1,acc5))
    with open(config.OUTPUT, 'a') as f:
        f.write(
            f'Tip-Adapter,{config.MODEL.ARCH},{config.DATA.IF_TEACHER},{config.DATA.NUM_FRAMES},{acc1:.3f},{acc5:.3f},{config.DATA.DATASET},'
            f'0 ,{config.PREFIX},{config.DATA.CACHE_SIZE},{config.TEMPORAL_POOLING}\n')

def run_tip_adapter_F(config, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,
                      clip_model, train_loader_F):
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.TRAIN.LR, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.TRAIN.EPOCHS * len(train_loader_F))

    beta, alpha = config.TIP_ADAPTER.INIT_BETA, config.TIP_ADAPTER.INIT_ALPHA
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(config.TRAIN.EPOCHS):
        # Train
        b = config.TRAIN.BATCH_SIZE
        adapter.train()
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, config.TRAIN.EPOCHS))
        acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
        for idx, batch_data in enumerate(tqdm(train_loader_F)):
            images = batch_data['data']
            label_id = batch_data['label']
            with torch.no_grad():
                image_input = []
                for image in images:
                    image = image.cpu().numpy()
                    image_input.append(image)
                image_input = [item for sublist in image_input for item in sublist]
                clip_logits, image_features, _ = clip_model(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.to(affinity.device)
            tip_logits = clip_logits + cache_logits * alpha
            value_1, indices_1 = tip_logits.topk(1, dim=-1)
            value_5, indices_5 = tip_logits.topk(5, dim=-1)
            acc1, acc5 = 0, 0
            for i in range(b):  # batch_size
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5:
                    acc5 += 1
            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            #修改成label_smooth的损失函数
            criterion = LabelSmoothingCrossEntropy()
            label_id = label_id.to(tip_logits.device)
            #把tip_logits的维度从[1,1,51]转到[1,51]
            tip_logits = tip_logits.squeeze(1)
            loss = criterion(tip_logits, label_id)


            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc@1: {:.4f}, Loss: {:.4f}'.format(current_lr, acc1_meter.avg, sum(loss_list) / len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.to(affinity.device)
        clip_logits = 100. * test_features @ clip_weights.T
        tip_logits =  clip_logits + cache_logits * alpha
        acc1, acc5 = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc1))
        if acc1 > best_acc:
            best_acc = acc1
            best_epoch = train_idx
            torch.save(adapter.weight, config.TIP_ADAPTER.CACHE_DIR + "/best_F_" + str(config.DATA.SHOTS) + "shots.pt")

    adapter.weight = torch.load(config.TIP_ADAPTER.CACHE_DIR + "/best_F_" + str(config.DATA.SHOTS) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(config, cache_keys, cache_values, val_features, val_labels, clip_weights,
                                      adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")

    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values.to(affinity.device)

    tip_logits = clip_logits + cache_logits * best_alpha
    acc1,acc5 = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter-F's test accuracy1: {:.2f}. , accuracy5: {:.2f}.****\n".format(max(best_acc, acc1),acc5))
    with open(config.OUTPUT, 'a') as f:
        f.write(
            f'Tip-Adapter-F,{config.MODEL.ARCH},{config.DATA.IF_TEACHER},{config.DATA.NUM_FRAMES},{acc1:.3f},{acc5:.3f},{config.DATA.DATASET},'
            f'{config.DATA.SHOTS} ,{config.PREFIX},{config.DATA.CACHE_SIZE},{config.TEMPORAL_POOLING}\n')


def main(config):
    cache_dir = './caches/' +  config.DATA.DATASET + '/'
    os.makedirs(cache_dir, exist_ok=True)
    print(config, "\n")
    config.defrost()  # Unfreeze the config
    config.TIP_ADAPTER.CACHE_DIR = cache_dir
    config.freeze()  # Freeze the config again
    #zero-shot
    if config.TRAIN.IF_TEST == 1:
        _,_,test_data,_,_,test_loader = build_dataloader(config)
        class_names = [class_name for i, class_name in test_data.classes]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = tbaclip.returnCLIP(config,class_names,device)
        acc1 = validate(test_loader, model, config)
        print(f"Accuracy of the network on the {len(test_data)} test videos: {acc1:.1f}%")
    else:
        if not os.path.exists(config.OUTPUT):
            with open(config.OUTPUT, 'w') as f:
                pass
        # Check if the file is empty
        if os.stat(config.OUTPUT).st_size == 0:
            with open(config.OUTPUT, 'a') as f:
                # Write the column names
                f.write('Model,Arch,If_teacher,Num_Frames,Acc1,Acc5,Dataset,Shots,prefix,cache_size,TEMPORAL_POOLING\n')
        train_cache_data, val_data, test_data,train_data_F, train_load_cache, val_loader, test_loader, train_load_F= build_dataloader(config)
        class_names = [class_name for i, class_name in test_data.classes]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = tbaclip.returnCLIP(config, class_names, device)
        # USE adapter-clip
        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = clip_classifier(class_names, model, config,device)
        # Construct the cache model by few-shot training set
        print("\nConstructing cache model by few-shot visual features and labels.")
        cache_keys, cache_values = build_cache_model(config, model, train_load_cache)
        # Pre-load val features
        print("\nLoading visual features and labels from val set.")
        val_features, val_labels = pre_load_features(config, "val", model, val_loader)
        # Pre-load test features
        print("\nLoading visual features and labels from test set.")
        test_features, test_labels = pre_load_features(config, "test", model, test_loader)
        # ------------------------------------------ Tip-Adapter ------------------------------------------
        if config.TRAIN.ZS == 1:
            run_tip_adapter(config, cache_keys, cache_values, val_features, val_labels, test_features, test_labels,
                            clip_weights)
        else:
        # ------------------------------------------ Tip-Adapter-F ------------------------------------------
            run_tip_adapter_F(config, cache_keys, cache_values, val_features, val_labels, test_features, test_labels,
                              clip_weights, model, train_load_F)

@torch.no_grad()
def validate(val_loader,model,config):
    b = val_loader.batch_size
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            images = batch_data['data']
            label_id = batch_data['label']
            image_input = []
            for image in images:
                image = image.cpu().numpy()
                image_input.append(image)
            image_input = [item for sublist in image_input for item in sublist]
            tot_similarity, _, _ = model(image_input)
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
        if not os.path.exists(config.OUTPUT):
            with open(config.OUTPUT, 'w') as f:
                pass
        # Check if the file is empty
        if os.stat(config.OUTPUT).st_size == 0:
            with open(config.OUTPUT, 'a') as f:
                # Write the column names
                f.write('Model,If_teacher,Num_Frames,Acc1,Acc5,Dataset\n')
        with open(config.OUTPUT, 'a') as f:
            f.write(f'{config.MODEL.ARCH},{config.DATA.IF_TEACHER},{config.DATA.NUM_FRAMES},{acc1_meter.avg:.3f},{acc5_meter.avg:.3f},{config.DATA.DATASET}\n')
        return acc1_meter.avg

if __name__ == '__main__':
    args, config = parse_option()
    main(config)