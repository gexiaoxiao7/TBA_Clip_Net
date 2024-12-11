import pandas as pd
from tqdm import tqdm

from model.tld import TeacherDetection
from model.transformer import FSATransformerEncoder
from utils.config import get_config
from utils.logger import create_logger
import argparse
import torch
import yaml
import os
import cv2
import model.TClip as tbaclip
import numpy as np
from PIL import Image
import torch.nn as nn
from utils.tools import classes, clip_classifier, attention_Fuc, promptlearner_Fuc, search_hp


def load_model(config,settings,class_names,model,device,logger):
    prompt_learner = tbaclip.PromptLearner(config, class_names, model.model, device, logger).to(torch.half)
    prompt_learner.load_state_dict(
        torch.load(config.TIP_ADAPTER.CACHE_DIR + "/" + str(settings['shot']) + "prompt_learner.pth"))

    attention_net = FSATransformerEncoder(dim=model.model.visual.output_dim, depth=6,
                                      heads=1, dim_head=64,
                                      mlp_dim=model.model.visual.output_dim * 4, nt=config.DATA.NUM_FRAMES,
                                      nh=1, nw=1,
                                      dropout=0.1).to(device).to(torch.half)
    cache_keys = torch.load(config.TIP_ADAPTER.CACHE_DIR + '/keys_' + str(settings['shot']) + "shots.pt")
    cache_values = torch.load(config.TIP_ADAPTER.CACHE_DIR + '/values_' + str(settings['shot']) + "shots.pt")
    attention_net.load_state_dict(
        torch.load(config.TIP_ADAPTER.CACHE_DIR + "/" + str(settings['shot']) + "attention_model.pth"))
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(model.dtype).cuda()
    adapter.weight = torch.load(config.TIP_ADAPTER.CACHE_DIR + "/best_F_" + str(settings['shot']) + "shots.pt")

    return prompt_learner, attention_net, cache_keys, cache_values, adapter

def prepare_frames(path, num_frames, detector, preprocess):

    video_capture = cv2.VideoCapture(path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_ids = np.linspace(0, total_frames - 2, num_frames)
    frame_ids = np.floor(frame_ids).astype(int)
    for i in range(total_frames + 1):
        ret, frame = video_capture.read()
        if not ret:
            break
        if i in frame_ids:
            frames.append(frame)

    while len(frames) < num_frames:
        frames.extend(frames[:num_frames - len(frames)])
    video_capture.release()
    for i in range(len(frames)):
        frames[i] = detector(frames[i])
    frames = [
        preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))).unsqueeze(0) for c in
        frames]
    return frames


@torch.no_grad()
def main(config, settings):
    cache_dir = './caches/' +  config.DATA.DATASET + '/'
    os.makedirs(cache_dir, exist_ok=True)
    config.defrost()  # Unfreeze the config
    config.TIP_ADAPTER.CACHE_DIR = cache_dir
    config.freeze()  # Freeze the config again

    # prepare model
    detector = TeacherDetection(config.MODEL.YOLO)
    class_names = [class_name for i, class_name in classes(config)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tbaclip.returnCLIP(config, class_names, device,logger)
    prompt_learner, attention_model, cache_keys, cache_values, tip_adapter = load_model(config,settings,class_names,model,device,logger)

    # preprocess
    video_infos = []
    for item in os.listdir(settings['input']):
        item_path = os.path.join(settings['input'], item)
        data = prepare_frames(item_path, config.DATA.NUM_FRAMES, detector, model.preprocess)
        video_infos.append((dict(filename=item, data=data)))

    from torch.utils.data import Dataset, DataLoader
    class VideoDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    video_dataset = VideoDataset(video_infos)
    video_loader = DataLoader(video_dataset, batch_size=settings['batch_size'], shuffle=False)

    # inference
    predict_label, file_name = [], []
    for idx, batch_info in enumerate(tqdm(video_loader)):
        images = batch_info['data']
        images = torch.stack(images)
        images = torch.transpose(images, 0, 1)
        _ , image_features, _, attention_format_feature = model(images)
        image_features = attention_Fuc(attention_model, attention_format_feature, image_features)
        clip_logits = promptlearner_Fuc(prompt_learner, image_features, model)
        tip_adapter.eval()
        affinity = tip_adapter(image_features)
        cache_logits = ((-1) * (settings['beta'] - settings['beta'] * affinity)).exp() @ cache_values.to(affinity.device)
        tip_logits = clip_logits + cache_logits * settings['alpha']

        _, indices_1 = tip_logits.topk(1, dim=-1)
        # 把indices_1 展平加入到predict_label中
        predict_label.extend(indices_1.flatten().tolist())
        file_name.extend(batch_info['filename'])
    print(predict_label)
    print(file_name)
    #visualize





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_set', '-infer_set', required=True, type=str, default='configs/inference.yaml')
    parser.add_argument('--config', '-cfg', required=True, type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--shot', type=int)
    opt = parser.parse_args()
    logger = create_logger(output_dir='train_output', dist_rank=0, name=f"inference")
    logger.info(opt)
    config = get_config(opt)

    with open(opt.infer_set) as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)
    if opt.input:
        settings['input'] = opt.input
    if opt.output:
        settings['output'] = opt.output
    if opt.shot:
        settings['shot'] = opt.shot

    main(config,settings)