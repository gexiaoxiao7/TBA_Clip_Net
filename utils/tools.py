import numpy
import torch.distributed as dist
import torch
import clip
from model.TClip import Prompts_build
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

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

def clip_classifier(classnames,clip_model,config,device):
    with torch.no_grad():
        prompts_learner = Prompts_build(classnames=classnames,config=config,device=device)
        prompts = prompts_learner()
        x = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)
        clip_weights = clip_model.model.encode_text(x)
        clip_weights /= clip_weights.norm(dim=-1, keepdim=True)
    return clip_weights

def build_cache_model(config, clip_model, train_loader_cache):
    if config.TIP_ADAPTER.LOAD_CACHE == 0:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(config.TIP_ADAPTER.AUGMENT_EPOCH):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, config.TIP_ADAPTER.AUGMENT_EPOCH))
                for idx, batch_data in enumerate(tqdm(train_loader_cache)):
                    images = batch_data['data']
                    label_id = batch_data['label']
                    image_input = []
                    for image in images:
                        image = image.cpu().numpy()
                        image_input.append(image)
                    image_input = [item for sublist in image_input for item in sublist]
                    _,image_features,_ = clip_model(image_input)
                    image_features = image_features.squeeze(0)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = label_id
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, config.TIP_ADAPTER.CACHE_DIR + '/keys_' + str(config.DATA.SHOTS) + "shots.pt")
        torch.save(cache_values, config.TIP_ADAPTER.CACHE_DIR + '/values_' + str(config.DATA.SHOTS) + "shots.pt")

    else:
        cache_keys = torch.load(config.TIP_ADAPTER.CACHE_DIR + '/keys_' + str(config.DATA.SHOTS) + "shots.pt")
        cache_values = torch.load(config.TIP_ADAPTER.CACHE_DIR + '/values_' + str(config.DATA.SHOTS) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(config, split, clip_model, loader):
    if config.TIP_ADAPTER.LOAD_PRE_FEAT == 0:
        features, labels = [], []

        with torch.no_grad():
            for idx, batch_data in enumerate(tqdm(loader)):
                images = batch_data['data']
                label_id = batch_data['label']
                image_input = []
                for image in images:
                    image = image.cpu().numpy()
                    image_input.append(image)
                image_input = [item for sublist in image_input for item in sublist]
                _,image_features,_ = clip_model(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                #把 image_features 从[1,1,512] 转换成[1,512]
                image_features = image_features.squeeze(0)
                features.append(image_features)
                labels.append(label_id)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, config.TIP_ADAPTER.CACHE_DIR + "/" + split + "_f.pt")
        torch.save(labels, config.TIP_ADAPTER.CACHE_DIR + "/" + split + "_l.pt")

    else:
        features = torch.load(config.TIP_ADAPTER.CACHE_DIR + "/" + split + "_f.pt")
        labels = torch.load(config.TIP_ADAPTER.CACHE_DIR + "/" + split + "_l.pt")
    return features, labels

def cls_acc(output, label):
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    # label = label.reshape(-1, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = torch.tensor(label).to(device)
    for idx, similarity in enumerate(output):
        cur_label = label[idx]
        value1, indices_1 = similarity.topk(1, dim=-1)
        value5, indices_5 = similarity.topk(5, dim=-1)
        acc1, acc5 = 0, 0
        for i in range(1): # batch_size
            if indices_1[i] == cur_label:
                acc1 += 1
            if cur_label in indices_5:
                acc5 += 1
        acc1_meter.update(float(acc1) * 100,1)
        acc5_meter.update(float(acc5) * 100,1)
    return acc1_meter.avg, acc5_meter.avg


def search_hp(config, cache_keys, cache_values, features, labels, clip_weights, adapter=None):
    if config.SEARCH_HP== 1:

        beta_list = [i * (config.SEARCH_SCALE[0] - 0.1) / config.SEARCH_STEP[0] + 0.1 for i in
                     range(config.SEARCH_STEP[0])]
        alpha_list = [i * (config.SEARCH_SCALE[1] - 0.1) / config.SEARCH_STEP[1] + 0.1 for i in
                      range(config.SEARCH_STEP[1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.to(affinity.device)
                clip_logits = 100. * features @ clip_weights.T
                tip_logits = clip_logits + cache_logits * alpha
                acc1, acc5 = cls_acc(tip_logits, labels)

                if acc1 > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc1))
                    best_acc = acc1
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha



def split_dataset(dataset, batch_size):
    # Step 1: Create a list of indices for each label
    label_to_indices = defaultdict(list)
    for idx, batch_data in enumerate(dataset):
        label = batch_data['label']
        label_to_indices[label].append(idx)

    # Step 2: Split the indices for each label and add them to the new index lists
    indices1, indices2 = [], []
    for indices in label_to_indices.values():
        mid = len(indices) // 2
        indices1.extend(indices[:mid])
        indices2.extend(indices[mid:])

    # Step 3: Create two Subset objects and two DataLoaders
    subset1 = Subset(dataset, indices1)
    subset2 = Subset(dataset, indices2)

    dataloader1 = DataLoader(subset1, batch_size=batch_size)
    dataloader2 = DataLoader(subset2, batch_size=batch_size)

    return dataloader1, dataloader2