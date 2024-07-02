import torch.distributed as dist
import torch
import clip
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import random
from sklearn.metrics import roc_auc_score, f1_score
import cv2
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib
import clip

from model.TClip import load_clip

matplotlib.use('Agg')
from model.tld import TeacherDetection
import matplotlib.pyplot as plt
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
        if config.TEXT_PROMPT.ONLY_LABEL == 0:
            prompts = ['The person was ' + x + '.' for x in classnames]
        else:
            prompts = classnames
        x = [clip.tokenize(prompt).to(device) for prompt in prompts]
        clip_weights = [clip_model.model.encode_text(i) for i in x]
        # x = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)
        clip_weights = torch.stack(clip_weights)
        clip_weights = clip_weights.squeeze(dim=1)
        clip_weights /= clip_weights.norm(dim=-1, keepdim=True)
    return clip_weights

def classes(config):
    classes_all = pd.read_csv(config.DATA.LABEL_LIST)
    return classes_all.values.tolist()

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
                    _,image_features,_,_  = clip_model(image_input)
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

        torch.save(cache_keys, config.TIP_ADAPTER.CACHE_DIR + '/keys_' + str(config.DATA.CACHE_SIZE) + "shots.pt")
        torch.save(cache_values, config.TIP_ADAPTER.CACHE_DIR + '/values_' + str(config.DATA.CACHE_SIZE) + "shots.pt")

    else:
        cache_keys = torch.load(config.TIP_ADAPTER.CACHE_DIR + '/keys_' + str(config.DATA.CACHE_SIZE) + "shots.pt")
        cache_values = torch.load(config.TIP_ADAPTER.CACHE_DIR + '/values_' + str(config.DATA.CACHE_SIZE) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(config, split, clip_model, loader):
    if config.TIP_ADAPTER.LOAD_PRE_FEAT == 0:
        features, labels ,attention_feature = [], [], []

        with torch.no_grad():
            for idx, batch_data in enumerate(tqdm(loader)):
                images = batch_data['data']
                label_id = batch_data['label']
                image_input = []
                for image in images:
                    image = image.cpu().numpy()
                    image_input.append(image)
                image_input = [item for sublist in image_input for item in sublist]
                _,image_features,_,attention_format_feature= clip_model(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                #把 image_features 从[1,1,512] 转换成[1,512]
                image_features = image_features.squeeze(0)
                features.append(image_features)
                labels.append(label_id)
                attention_feature.append(attention_format_feature)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, config.TIP_ADAPTER.CACHE_DIR + "/" + split + "_f.pt")
        torch.save(labels, config.TIP_ADAPTER.CACHE_DIR + "/" + split + "_l.pt")
        torch.save(attention_feature, config.TIP_ADAPTER.CACHE_DIR + "/" + split + "_a.pt")

    else:
        features = torch.load(config.TIP_ADAPTER.CACHE_DIR + "/" + split + "_f.pt")
        labels = torch.load(config.TIP_ADAPTER.CACHE_DIR + "/" + split + "_l.pt")
        attention_feature = torch.load(config.TIP_ADAPTER.CACHE_DIR + "/" + split + "_a.pt")
    return features, labels, attention_feature

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def cls_acc(output, label, plot = False, config = None):
    acc1_meter, acc5_meter,acc3_meter = AverageMeter(), AverageMeter(), AverageMeter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = label.clone().detach().to(device)
    all_preds = []
    all_labels = []
    all_probs = []
    for idx, similarity in enumerate(output):
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
    # print(all_labels)
    # print("probs shape:",end="")
    # print(all_probs)
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
                acc1, acc3 ,acc5, auc, f1 = cls_acc(tip_logits, labels, False)

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

    # Step 2: Shuffle and split the indices for each label and add them to the new index lists
    indices1, indices2 = [], []
    for indices in label_to_indices.values():
        random.shuffle(indices)  # Shuffle the indices
        mid = len(indices) // 2
        if len(indices) % 2 == 1:  # Check if the number of samples is odd
            indices1.extend(indices[:mid+1])  # If odd, subset1 gets one more sample
            indices2.extend(indices[mid+1:])  # subset2 gets one less sample
        else:
            indices1.extend(indices[:mid])
            indices2.extend(indices[mid:])

    # Step 3: Create two Subset objects and two DataLoaders
    subset1 = Subset(dataset, indices1)
    subset2 = Subset(dataset, indices2)

    dataloader1 = DataLoader(subset1, batch_size=batch_size)
    dataloader2 = DataLoader(subset2, batch_size=batch_size)

    return subset1, dataloader1, subset2, dataloader2

def attention_Fuc(attention_net, attention_feature):
    attention_net.eval()
    res = []
    for feature in attention_feature:
        # feature = torch.unsqueeze(feature, 0)
        attention_weights = attention_net(feature)
        # video_feature = torch.sum(torch.bmm(attention_weights.transpose(1, 2), feature), dim=1)

        weighted_features = torch.mul(attention_weights, feature)
        video_feature = torch.mean(weighted_features, dim=1)

        video_features = torch.unsqueeze(video_feature, 0)
        norm = video_features.norm(dim=-1, keepdim=True)
        video_features = video_features / norm
        res.append(video_features)
    # 移除多余的维度
    res = [x.squeeze() for x in res]
    # 堆叠张量
    res = torch.stack(res)
    return res

def promptlearner_Fuc(prompt_learner, image_feature, clip_model):
    prompt_learner.eval()
    logits = []
    prompts = prompt_learner(image_feature)
    for pts_i, imf_i in zip(prompts, image_feature):
        text_features = clip_model.text_encoder(pts_i, prompt_learner.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        l_i = (clip_model.logit_scale.exp() * imf_i @ text_features.t()).softmax(dim=-1)
        logits.append(l_i)
    logits = torch.stack(logits)
    return logits


def visulize_attention_ratio(img_path, attention_mask, ratio=0.5, cmap="jet"):
    # load the image
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention mask
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)


def prepare_frames(path,num_frames,device):
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return None
    video_capture = cv2.VideoCapture(path)
    model , preprocess = load_clip('ViT-L/14@336px',device)
    detector = TeacherDetection('Yolo-model/yolov8n.pt')
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_ids = np.linspace(0, total_frames - 2, num_frames)
    frame_ids = np.floor(frame_ids).astype(int)
    for i in range(total_frames+1) :
        ret, frame = video_capture.read()
        if not ret:
            break
        if i in frame_ids:
            # Resize the frame
            frame = cv2.resize(frame, (224*8, 224*8))
            frames.append(frame)
    while len(frames) < num_frames:
        frames.extend(frames[:num_frames - len(frames)])
    video_capture.release()
    for i in range(len(frames)):
        frames[i] = detector(frames[i])
    return frames
def visual(config, path ,logits):
    # Step 1: Prepare frames from the video
    logits = logits.flatten()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = prepare_frames(path, config.DATA.NUM_FRAMES,device)
    top1_classes = logits.argsort()[-1].item()  # Use .item() to get the value
    top1_probs = logits[top1_classes]
    # Step 2: Add text to each frame and save
    for i, frame in enumerate(frames):
        # Get top 1 class and its probability
        cls = classes(config)
        class_name = cls[top1_classes][1]
        prob = top1_probs
        # Add text to the frame
        text_prob = f'{prob:.2f}:'
        text_class = f'{class_name}'
        cv2.putText(frame, text_prob, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, text_class, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # Save the frame
        # 取path的最后一级路径名
        video_name = path.split('/')[-1].split('.')[0]
        cv2.imwrite(f'output/{video_name}_frame_{i}.jpg', frame)
    for i, frame in enumerate(frames):
        video_name = path.split('/')[-1].split('.')[0]
        cv2.imwrite(f'output/{video_name}_frame_orign_{i}.jpg', frame)