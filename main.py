import numpy as np

import model.TClip as tbaclip
import torch
import os
import argparse

import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model.transformer import FSATransformerEncoder
from utils.config import get_config
from dataSets.build import build_dataloader
from utils.tools import AverageMeter, clip_classifier, build_cache_model, pre_load_features, split_dataset, \
    attention_Fuc, promptlearner_Fuc, classes, visual
from utils.tools import search_hp
import torch.nn as nn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy
import torch.backends.cudnn as cudnn
import random
import shutil
from utils.logger import create_logger

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/zero_shot/eval/hmdb/tba_clip_hmdb51_base.yaml')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--num_frames', type=int)
    parser.add_argument('--shots', type=int)
    parser.add_argument('--temporal_pooling', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--if_teacher', type=int)
    parser.add_argument('--output', type=str)
    parser.add_argument('--load_cache', type=int)
    parser.add_argument('--load_attention', type=int)
    parser.add_argument('--load_pre_feat', type=int)
    parser.add_argument('--load_lp', type=int)
    parser.add_argument('--zs', type=int)
    parser.add_argument('--cache_size', type=int)
    parser.add_argument('--lp', type=int)
    parser.add_argument('--only_label', type=int)
    parser.add_argument('--label_smooth', type=int)
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    config = get_config(args)
    return args, config



def run_tip_adapter(config, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    test_features = test_features
    val_features = val_features
    print("\n-------- Searching hyperparameters on the val set. --------")
    # Zero-shot CLIP
    clip_logits = (100. * val_features @ clip_weights.T).softmax(dim=-1)
    acc1, acc3, acc5, auc, f1 = validate(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy1: {:.2f}. accuracy3: {:.2f} accuracy5: {:.2f} ****\n".format(acc1,acc3,acc5))

    # Tip-Adapter
    beta, alpha = config.TIP_ADAPTER.INIT_BETA, config.TIP_ADAPTER.INIT_ALPHA

    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.to(affinity.device)

    tip_logits = clip_logits + cache_logits * alpha
    acc1,acc3,acc5, auc, f1 = validate(tip_logits, val_labels)

    print("**** Tip-Adapter's val accuracy1: {:.2f}. accuracy3: {:.2f} accuracy5: {:.2f}, auc: {:.2f}, f1: {:.2f}****\n".format(acc1,acc3,acc5,auc,f1))


    # Search Hyperparameters
    best_beta, best_alpha = search_hp(config, cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = (100. * test_features @ clip_weights.T).softmax(dim=-1)
    acc1, acc3 ,acc5, auc, f1 = validate(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy1: {:.2f}. accuracy3:{:.2f} accuracy5:{:.2f} auc:{:.2f} f1:{:.2f}****\n".format(acc1,acc3,acc5,auc,f1))
    with open(config.OUTPUT, 'a') as f:
        f.write(
            f'Zero-shot Clip,{config.MODEL.ARCH},{config.DATA.IF_TEACHER},{config.DATA.NUM_FRAMES},{acc1:.3f},{acc3:.3f},{acc5:.3f},{auc:.3f},{f1:.3f},{config.DATA.DATASET},'
            f'0,{str(config.TEXT_PROMPT.N_CTX_PRE) + " " + str(config.TEXT_PROMPT.N_CTX_POST) },0,{config.TEMPORAL_POOLING}\n')
    # Tip-Adapter
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values.to(affinity.device)

    tip_logits = clip_logits + cache_logits * best_alpha
    acc1, acc3 ,acc5, auc, f1 = validate(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy1: {:.2f}. accuracy3: {:.2f} accuracy5: {:.2f} auc: {:.2f} f1:{:.2f}****\n".format(acc1,acc3,acc5,auc,f1))
    with open(config.OUTPUT, 'a') as f:
        f.write(
            f'Tip-Adapter,{config.MODEL.ARCH},{config.DATA.IF_TEACHER},{config.DATA.NUM_FRAMES},{acc1:.3f},{acc3:.3f},{acc5:.3f},{auc:.3f},{f1:.3f},{config.DATA.DATASET},'
            f'0 ,{str(config.TEXT_PROMPT.N_CTX_PRE) + " " + str(config.TEXT_PROMPT.N_CTX_POST) },{config.DATA.CACHE_SIZE},{config.TEMPORAL_POOLING}\n')

def run_tip_adapter_F(config, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,
                      clip_model, train_loader_F,attention_net,prompt_learner,attention_test_feature,attention_val_feature):

    if config.TEMPORAL_POOLING == 'attention':
        attention_net.load_state_dict(
            torch.load(config.TIP_ADAPTER.CACHE_DIR + "/" + str(config.DATA.SHOTS) + "attention_model.pth"))

    if config.TRAIN.LP == 1 :
        prompt_learner.load_state_dict(
            torch.load(config.TIP_ADAPTER.CACHE_DIR + "/" + str(config.DATA.SHOTS) + "prompt_learner.pth"))

    test_features = test_features if config.TEMPORAL_POOLING != 'attention' else attention_Fuc(attention_net,
                                                                                               attention_test_feature)
    val_features = val_features if config.TEMPORAL_POOLING != 'attention' else attention_Fuc(attention_net,attention_val_feature)
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
                clip_logits, image_features, _,attention_format = clip_model(image_input)
                if config.TEMPORAL_POOLING == 'attention':
                    t = []
                    t.append(attention_format)
                    attention_format = t
                    image_features = attention_Fuc(attention_net,attention_format)
                else:
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                if config.TRAIN.LP == 1:
                    clip_logits = promptlearner_Fuc(prompt_learner, image_features, clip_model)
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

            # 用交叉熵损失函数
            criterion = LabelSmoothingCrossEntropy() if config.TRAIN.LABEL_SMOOTH == 1 else nn.CrossEntropyLoss()
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
        clip_logits = 100. * test_features @ clip_weights.T if config.TRAIN.LP == 0 else promptlearner_Fuc(
            prompt_learner, test_features, clip_model)
        tip_logits =  clip_logits + cache_logits * alpha
        acc1, acc3 ,acc5, auc, f1 = validate(tip_logits, test_labels)
        print("**** Tip-Adapter-F's test accuracy: {:.2f}. auc:{:.2f}****\n".format(acc1,auc))
        if acc1 >= best_acc:
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
    acc1,acc3,acc5, auc, f1 = validate(tip_logits, test_labels, plot= True,config= config)
    print("**** Tip-Adapter-F's test accuracy1: {:.2f}. , accuracy3: {:.2f},accuracy5: {:.2f}. auc: {:.2f}, f1: {:.2f}****\n".format(max(best_acc, acc1),acc3,acc5, auc, f1))
    with open(config.OUTPUT, 'a') as f:
        f.write(
            f'Tip-Adapter-F,{config.MODEL.ARCH},{config.DATA.IF_TEACHER},{config.DATA.NUM_FRAMES},{acc1:.3f},{acc3:.3f},{acc5:.3f},{auc:.3f},{f1:.3f},{config.DATA.DATASET},'
            f'{config.DATA.SHOTS} ,{str(config.TEXT_PROMPT.N_CTX_PRE) + " " + str(config.TEXT_PROMPT.N_CTX_POST) },{config.DATA.CACHE_SIZE},{config.TEMPORAL_POOLING}\n')

def train_lp(clip_model,device,config,train_loader,class_names,attention_net):

    if config.TEMPORAL_POOLING == 'attention':
        attention_net.load_state_dict(
            torch.load(config.TIP_ADAPTER.CACHE_DIR + "/" + str(config.DATA.SHOTS) + "attention_model.pth"))

    prompt_learner = tbaclip.PromptLearner(config, class_names, clip_model.model, device).to(torch.half)
    optimizer = torch.optim.Adam(prompt_learner.parameters(), lr=config.TRAIN.LR, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.TRAIN.EPOCHS * len(train_loader))

    for train_idx in range(config.TRAIN.EPOCHS):
        # Train
        b = config.TRAIN.BATCH_SIZE
        prompt_learner.train()
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, config.TRAIN.EPOCHS))
        acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
        for idx, batch_data in enumerate(tqdm(train_loader)):
            images = batch_data['data']
            label_id = batch_data['label']
            with torch.no_grad():
                image_input = []
                for image in images:
                    image = image.cpu().numpy()
                    image_input.append(image)
                image_input = [item for sublist in image_input for item in sublist]
                _, image_features, _,attention_format = clip_model(image_input)
                if config.TEMPORAL_POOLING == 'attention':
                    t = []
                    t.append(attention_format)
                    attention_format = t
                    image_features = attention_Fuc(attention_net,attention_format)
                    # 增加一个维度
                    image_features = torch.unsqueeze(image_features, 0)
                else:
                    image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = []
            prompts = prompt_learner(image_features)
            for pts_i, imf_i in zip(prompts, image_features):
                text_features = clip_model.text_encoder(pts_i, prompt_learner.tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = (clip_model.logit_scale.exp() * imf_i @ text_features.t()).softmax(dim=-1)
                logits.append(l_i)
            logits = torch.stack(logits)
            value_1, indices_1 = logits.topk(1, dim=-1)
            value_5, indices_5 = logits.topk(5, dim=-1)
            acc1, acc5 = 0, 0
            for i in range(b):  # batch_size
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5:
                    acc5 += 1
            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            # 修改成label_smooth的损失函数
            criterion = LabelSmoothingCrossEntropy() if config.TRAIN.LABEL_SMOOTH == 1 else nn.CrossEntropyLoss()
            label_id = label_id.to(logits.device)
            # 把tip_logits的维度从[1,1,51]转到[1,51]
            logits = logits.squeeze(1)
            loss = criterion(logits, label_id)

            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            prompt_learner.float()
            optimizer.step()
            prompt_learner.half()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc@1: {:.4f}, Loss: {:.4f}'.format(current_lr, acc1_meter.avg,
                                                               sum(loss_list) / len(loss_list)))

        torch.save(prompt_learner.state_dict(),
                   config.TIP_ADAPTER.CACHE_DIR + "/" + str(config.DATA.SHOTS) + "prompt_learner.pth")

def train_attention(clip_model,device,config,train_loader,clip_weights):
    attention_net = FSATransformerEncoder(dim=clip_model.model.visual.output_dim, depth=6,
                                      heads=1, dim_head=64,
                                      mlp_dim=clip_model.model.visual.output_dim * 4, nt=config.DATA.NUM_FRAMES,
                                      nh=1, nw=1,
                                      dropout=0.1).to(device).to(torch.half)
    # attention_net = FSATransformerEncoder(dim=clip_model.visual.output_dim, depth=6,
    #                                   heads=1, dim_head=64,
    #                                   mlp_dim=clip_model.visual.output_dim * 4, nt=config.DATA.NUM_FRAMES,
    #                                   nh=1, nw=1,
    #                                   dropout=0.1).to(device).to(torch.half)
    optimizer = torch.optim.Adam(attention_net.parameters(), lr=config.TRAIN.LR, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.TRAIN.EPOCHS * len(train_loader))



    for train_idx in range(config.TRAIN.EPOCHS):
        # Train
        b = config.TRAIN.BATCH_SIZE
        attention_net.train()
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, config.TRAIN.EPOCHS))
        acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
        for idx, batch_data in enumerate(tqdm(train_loader)):
            images = batch_data['data']
            label_id = batch_data['label']
            with torch.no_grad():
                image_input = []
                for image in images:
                    image = image.cpu().numpy()
                    image_input.append(image)
                image_input = [item for sublist in image_input for item in sublist]
                _,image_features,_,image_features_attention = clip_model(image_input)
            attention_weights = attention_net(image_features_attention)
            # video_feature = torch.sum(torch.bmm(attention_weights.transpose(1, 2), image_features), dim=1)

            weighted_features = torch.mul(attention_weights, image_features)
            video_feature = torch.mean(weighted_features, dim=1)

            video_features = torch.unsqueeze(video_feature, 0)
            norm = video_features.norm(dim=-1, keepdim=True)
            video_features = video_features / norm
            clip_logits = (100. * video_features @ clip_weights.T).softmax(dim=-1)
            value_1, indices_1 = clip_logits.topk(1, dim=-1)
            value_5, indices_5 = clip_logits.topk(5, dim=-1)
            acc1, acc5 = 0, 0
            for i in range(b):  # batch_size
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5:
                    acc5 += 1
            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            # 修改成label_smooth的损失函数
            criterion = LabelSmoothingCrossEntropy() if config.TRAIN.LABEL_SMOOTH == 1 else nn.CrossEntropyLoss()
            label_id = label_id.to(clip_logits.device)
            # 把tip_logits的维度从[1,1,51]转到[1,51]
            clip_logits = clip_logits.squeeze(1)
            loss = criterion(clip_logits, label_id)

            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc@1: {:.4f}, Loss: {:.4f}'.format(current_lr, acc1_meter.avg,
                                                               sum(loss_list) / len(loss_list)))

        torch.save(attention_net.state_dict(), config.TIP_ADAPTER.CACHE_DIR + "/" + str(config.DATA.SHOTS) +"attention_model.pth")

@torch.no_grad()
def validate(output, label, plot = False, config = None):
    acc1_meter, acc5_meter,acc3_meter = AverageMeter(), AverageMeter(), AverageMeter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = label.clone().detach().to(device)
    all_preds = []
    all_labels = []
    all_probs = []
    for idx, batch_data in enumerate(output):
        cur_label = label[idx]
        value1, indices_1 = similarity.topk(1, dim=-1)
        value3, indices_3 = similarity.topk(3, dim=-1)
        value5, indices_5 = similarity.topk(5, dim=-1)
        acc1, acc3 ,acc5 = 0, 0,0
        for i in range(1): # batch_size
            if indices_1[i] == cur_label:
                acc1 += 1
            if cur_label in indices_3:
                acc3 += 1
            if cur_label in indices_5:
                acc5 += 1
        acc1_meter.update(float(acc1) * 100,1)
        acc3_meter.update(float(acc3) * 100, 1)
        acc5_meter.update(float(acc5) * 100,1)
        all_preds.append(indices_1.cpu().numpy())
        all_labels.append(cur_label.cpu().numpy())
        probs = similarity.softmax(dim=-1).cpu().detach().numpy()
        if len(probs.shape) > 1:  # 如果probs有多个维度
            probs /= probs.sum(axis=1, keepdims=True)  # 归一化概率，使其和为1
        else:  # 如果probs只有一个维度
            probs /= probs.sum()  # 归一化概率，使其和为1
        if not np.isclose(probs.sum(), 1):
            probs = np.clip(probs, 0, 1)
            min_index = np.argmin(probs)
            sum = 0
            for i,num in enumerate(probs):
                if i != min_index:
                    sum += num
            probs[min_index] = 1 - sum
        all_probs.append(probs)
    # AUC and F1
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    # print("labels shape:",end="")
    # print(all_labels.shape)
    # print("probs shape:",end="")
    # print(all_probs.shape)
    # print("preds shape:",end="")
    # print(all_preds.shape)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    f1 = f1_score(all_labels, all_preds, average='macro')
    if plot:
        cls = classes(config)
        labels = [sublist[1] for sublist in cls]

        cm = confusion_matrix(np.array(all_labels), np.array(all_preds))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Convert to percentages

        fig, ax = plt.subplots(figsize=(10, 10))  # Increase figure size
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, shrink=0.7)  # Adjust the length of colorbar

        # Show all ticks
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        # ax.set_xticklabels(labels, rotation=45, fontsize='medium')  # Increase font size
        # ax.set_yticklabels(labels, fontsize='medium')  # Increase font size

        # Loop over data dimensions and create text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], '.2f'),  # Show 2 decimal places
                        ha='center', va='center', color='black')

        fig.tight_layout()  # Increase margin
        plt.savefig('confusion_matrix.png')

    return acc1_meter.avg, acc3_meter.avg, acc5_meter.avg, auc, f1

def main(config):
    cache_dir = './caches/' +  config.DATA.DATASET + '/'
    os.makedirs(cache_dir, exist_ok=True)
    config.defrost()  # Unfreeze the config
    config.TIP_ADAPTER.CACHE_DIR = cache_dir
    config.freeze()  # Freeze the config again
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = [class_name for i, class_name in classes(config)]
    model = tbaclip.returnCLIP(config, class_names, device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,
                                                      find_unused_parameters=False)
    #zero-shot
    if config.TRAIN.IF_TEST == 1:
        (_, _, test_data, _, _,
        _, _, test_loader, _, _) = build_dataloader(config, logger)
        acc1 = validate(test_loader, model, config)
        logger.info(f"Accuracy of the network on the {len(test_data)} test videos: {acc1:.1f}%")
    else:
        if not os.path.exists(config.OUTPUT):
            with open(config.OUTPUT, 'w') as f:
                pass
        # Check if the file is empty
        if os.stat(config.OUTPUT).st_size == 0:
            with open(config.OUTPUT, 'a') as f:
                # Write the column names
                f.write('Model,Arch,If_teacher,Num_Frames,Acc1,Acc3,Acc5,AUC,F1,Dataset,Shots,n_ctx,cache_size,TEMPORAL_POOLING\n')
        pre_time = int(time.time())
        (train_cache_data, val_data, test_data,train_data_F, train_data_a,
         train_load_cache, val_loader, test_loader, train_load_F, train_load_a)= build_dataloader(config, logger)
        logger.info(f"process Time cost: {int(time.time()) - pre_time} seconds.")
        pre_time_seconds = int(time.time())
        # USE adapter-clip
        logger.info("\nGetting textual features as CLIP's classifier.")
        clip_weights = clip_classifier(class_names, model, config,device)
        # Construct the cache model by few-shot training set
        print("\nConstructing cache model by few-shot visual features and labels.")
        cache_keys, cache_values = build_cache_model(config, model, train_load_cache)
        # Pre-load val features
        print("\nLoading visual features and labels from val set.")
        val_features, val_labels, attention_val_feature = pre_load_features(config, "val", model, val_loader)
        # Pre-load test features
        print("\nLoading visual features and labels from test set.")
        test_features, test_labels, attention_test_feature = pre_load_features(config, "test", model, test_loader)
        #load_attention
        attention_net = FSATransformerEncoder(dim=model.model.visual.output_dim, depth=6,
                                          heads=1, dim_head=64,
                                          mlp_dim=model.model.visual.output_dim * 4, nt=config.DATA.NUM_FRAMES,
                                          nh=1, nw=1,
                                          dropout=0.1).to(device).to(torch.half)
        print("\nTraining attention Net.")
        if config.MODEL.LOAD_ATTENTION == 0 and config.TEMPORAL_POOLING == 'attention' and config.TRAIN.ZS == 0:
            train_attention(model, device, config, train_load_F, clip_weights)
        # use prompt_learner
        print("\nTraining Prompt Learner.")
        prompt_learner = tbaclip.PromptLearner(config, class_names, model.model, device).to(torch.half)
        if config.MODEL.LOAD_LP == 0 and config.TRAIN.LP == 1 and config.TRAIN.ZS == 0 :
            train_lp(model, device, config, train_load_a, class_names,attention_net)
        # ------------------------------------------ Tip-Adapter ------------------------------------------
        if config.TRAIN.ZS == 1:
            run_tip_adapter(config, cache_keys, cache_values, val_features, val_labels, test_features, test_labels,clip_weights)
        else:
        # ------------------------------------------ Tip-Adapter-F ------------------------------------------
            run_tip_adapter_F(config, cache_keys, cache_values, val_features, val_labels, test_features, test_labels,
                              clip_weights, model, train_load_F,attention_net,prompt_learner,attention_test_feature,attention_val_feature)

        print(f"Training Time cost: {int(time.time()) - pre_time_seconds} seconds.")

if __name__ == '__main__':
    args, config = parse_option()
    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank) #for linux
    torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    # logger
    logger = create_logger(output_dir='train_output', dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(config)

    seed = dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True



    # save config
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)



    main(config)