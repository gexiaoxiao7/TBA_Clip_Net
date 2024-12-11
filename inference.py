from tqdm import tqdm

from model.tld import TeacherDetection
from utils.logger import create_logger
import argparse
import torch
import yaml
import clip
import os
import cv2
import numpy as np
from PIL import Image


def load_model(config):
    prompt_learner = torch.load(config['prompt_learner'])
    attention_model = torch.load(config['attention_model'])
    cache_keys = torch.load(config['cache_keys'])
    cache_values = torch.load(config['cache_values'])
    tip_adapter = torch.load(config['tip_adapter'])

    return prompt_learner, attention_model, cache_keys, cache_values, tip_adapter

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


def main(opt):
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    prompt_learner, attention_model, cache_keys, cache_values, tip_adapter = load_model(config)

    detector = TeacherDetection(config['arch_yolo'])
    # preprocess
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(config['arch_clip'], device=device)
    video_infos = []
    for item in os.listdir(config['input']):
        item_path = os.path.join(config['input'], item)
        data = prepare_frames(item_path, config['num_frames'], detector,preprocess)
        video_infos.append((dict(filename=item, data=data)))

    for idx, info in enumerate(tqdm(video_infos)):
        images = info['data']



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/inference.yaml')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--prompt_learner', '-pl', type=str)
    parser.add_argument('--attention_model', '-at', type=str)
    parser.add_argument('--cache_keys', '-ck', type=str)
    parser.add_argument('--cache_values', '-cv', type=str)
    parser.add_argument('--tip_adapter', '-ta', type=str)
    opt = parser.parse_args()

    logger = create_logger(output_dir='train_output', dist_rank=0, name=f"inference")
    logger.info(opt)

    main(opt)