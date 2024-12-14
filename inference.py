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

def prepare_frames(path, num_frames, detector, preprocess, frame_step=2):

    video_capture = cv2.VideoCapture(path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = list(range(num_frames, total_frames - 1, frame_step))
    frames = []
    all_frames = []
    for i in range(total_frames + 1):
        ret, frame = video_capture.read()
        if not ret:
            break
        all_frames.append(frame)

    for i in frame_ids:
        t = []
        for j in range(i-num_frames,i):
            t.append(all_frames[j])
        frames.append(t)

    video_capture.release()
    for i in range(len(frames)):
        for j in range(len(frames[i])):
            frames[i][j] = detector(frames[i][j])
            frames[i][j] = preprocess(Image.fromarray(cv2.cvtColor(frames[i][j], cv2.COLOR_BGR2RGB))).unsqueeze(0)
            frames[i][j] = frames[i][j].unsqueeze(1)
    return frames, all_frames, frame_ids


def visualize(all_frames, frame_ids, predict_text, file_names, sims ,output_dir='output'):
    for idx in range(len(all_frames)):
        height, width, layers = all_frames[idx][0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        output_path = os.path.join(output_dir, file_names[idx])
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        current_text_1 = ""
        current_text_2 = ""
        current_text_3 = ""
        for i, frame in enumerate(all_frames[idx]):
            if i in frame_ids[idx]:
                index = frame_ids[idx].index(i)
                current_text_1 = f"{predict_text[idx][index][0]}:{str(sims[idx][index][0])}"
                current_text_2 = f"{predict_text[idx][index][1]}:{str(sims[idx][index][1])}"
                current_text_3 = f"{predict_text[idx][index][2]}:{str(sims[idx][index][2])}"

            cv2.putText(frame, current_text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, current_text_2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, current_text_3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            out.write(frame)
        out.release()
        print(f"Annotated video saved to {output_path}")

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

    # prepare data
    video_infos, all_frames, frame_ids = [], [], []
    for item in os.listdir(settings['input']):
        item_path = os.path.join(settings['input'], item)
        infos_by_frame, all_frames_t, frame_ids_t = prepare_frames(item_path, config.DATA.NUM_FRAMES, detector, model.preprocess, settings['frame_step'])
        video_infos.append((dict(filename=item, data=infos_by_frame)))
        all_frames.append(all_frames_t)
        frame_ids.append(frame_ids_t)

    # inference
    # TODO: batch inference
    predict_label, file_name, sims = [], [], []
    for idx, video_info in enumerate(video_infos):
        t, sim = [], []
        for info in tqdm(video_info['data']):
            images = info
            images = torch.stack(images)
            images = torch.transpose(images, 0, 1)
            _ , image_features, _, attention_format_feature = model(images)
            image_features = attention_Fuc(attention_model, attention_format_feature, image_features)
            clip_logits = promptlearner_Fuc(prompt_learner, image_features, model)
            tip_adapter.eval()
            affinity = tip_adapter(image_features)
            cache_logits = ((-1) * (settings['beta'] - settings['beta'] * affinity)).exp() @ cache_values.to(affinity.device)
            tip_logits = clip_logits + cache_logits * settings['alpha']

            s, indices_3 = tip_logits.topk(3, dim=-1)
            s = torch.nn.functional.softmax(s, dim=-1).squeeze().tolist()
            s = [round(x, 3) for x in s]
            t.append(indices_3)
            sim.append(s)
        predict_label.append(t)
        sims.append(sim)
        file_name.append(video_info['filename'])

    # 映射一下，将predict_label中的索引映射到类别
    predict_text = []
    for i in range(len(predict_label)): #2
        ii = []
        for j in range(len(predict_label[i])): #15
            t = predict_label[i][j].squeeze().tolist()
            t = [class_names[x] for x in t]
            ii.append(t)
        predict_text.append(ii)

    visualize(all_frames, frame_ids, predict_text, file_name, sims, settings['output'])





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