import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.tld import TeacherDetection
import numpy as np
from PIL import Image
import cv2
import torch
import clip
from utils.tools import split_dataset

class VideoDataset():
    def __init__(self,config,preprocess,device,ann_file,shot=0,type = 'train'):
        self.labels_file = config.DATA.LABEL_LIST
        self.ann_file = ann_file
        self.data_prefix = config.DATA.ROOT
        self.num_frames = config.DATA.NUM_FRAMES
        self.input_size = config.DATA.INPUT_SIZE
        self.yolo_model = config.MODEL.YOLO
        self.preprocess = preprocess
        self.device = device
        self.shot = shot
        self.type = type
        self.if_teacher = config.DATA.IF_TEACHER
        self.detector = TeacherDetection(self.yolo_model)
        self.video_info = self.load_annotations()
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def prepare_frames(self, path):
        video_capture = cv2.VideoCapture(path)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        frame_ids = np.linspace(0, total_frames - 2, self.num_frames)
        frame_ids = np.floor(frame_ids).astype(int)
        for i in range(total_frames+1) :
            ret, frame = video_capture.read()
            if not ret:
                break
            if i in frame_ids:
                frames.append(frame)

        while len(frames) < self.num_frames:
            frames.extend(frames[:self.num_frames - len(frames)])
        video_capture.release()
        if self.if_teacher == 1:
            for i in range(len(frames)):
                frames[i] = self.detector(frames[i])
        frames = [
            self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device) for c in
            frames]
        return frames

    def load_annotations(self):
        video_infos = []
        #初始化一个列表，长度为num_classes，值为0
        class_counts = {}
        total_lines = sum(1 for line in open(self.ann_file, 'r'))
        if self.type == 'train_cache':
            with open(self.ann_file, 'r') as fin:
                lines = fin.readlines()
                # start_idx = int(total_lines * 2 / 3)  # Calculate the start index
                for idx in range(total_lines):  # Start from the last third
                    if idx % 50 == 0 and idx != 0:
                        progress = (idx / total_lines) * 100
                        print(f'Processed {idx} samples, progress: {progress:.2f}%')
                    line = lines[total_lines - idx - 1]
                    line_split = line.strip().split()
                    filename, label = line_split
                    label = int(label)
                    if label in class_counts and class_counts[label] >= self.shot:
                        continue
                    if label not in class_counts:
                        class_counts[label] = 1
                    else:
                        class_counts[label] += 1
                    data = self.prepare_frames(self.data_prefix + filename)
                    video_infos.append(dict(filename=filename, label=label, data=data))
        elif self.type == 'train_a':
            with open(self.ann_file, 'r') as fin:
                lines = fin.readlines()
                # start_idx = int(total_lines * 1 / 3)  # Calculate the start index
                for idx in range(total_lines):  # Start from the last third
                    if idx % 50 == 0 and idx != 0:
                        progress = (idx / total_lines) * 100
                        print(f'Processed {idx} samples, progress: {progress:.2f}%')
                    line = lines[total_lines - idx - 1]
                    line_split = line.strip().split()
                    filename, label = line_split
                    label = int(label)
                    if label in class_counts and class_counts[label] >= self.shot:
                        continue
                    if label not in class_counts:
                        class_counts[label] = 1
                    else:
                        class_counts[label] += 1
                    data = self.prepare_frames(self.data_prefix + filename)
                    video_infos.append(dict(filename=filename, label=label, data=data))
        else:
            with open(self.ann_file, 'r') as fin:
                for idx, line in enumerate(fin):
                    if idx % 50 == 0 and idx != 0:
                        progress = (idx / total_lines) * 100
                        print(f'Processed {idx} samples, progress: {progress:.2f}%')
                    line_split = line.strip().split()
                    filename, label = line_split
                    label = int(label)
                    if self.type == 'train_F':
                        if label in class_counts and class_counts[label] >= self.shot:
                            continue
                        if label not in class_counts:
                            class_counts[label] = 1
                        else:
                            class_counts[label] += 1
                        data = self.prepare_frames(self.data_prefix + filename)
                        video_infos.append(dict(filename=filename, label=label, data=data))
                    else:
                        data = self.prepare_frames(self.data_prefix + filename)
                        video_infos.append(dict(filename=filename, label=label, data=data))
        return video_infos

    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, idx):
        return self.video_info[idx]

class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices
    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
    def __len__(self):
        return len(self.indices)
    def set_epoch(self, epoch):
        self.epoch = epoch

def build_dataloader(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load(config.MODEL.ARCH, device=device)
    test_data = VideoDataset(config, preprocess=preprocess, device=device, ann_file=config.DATA.TEST_FILE,type='test')
    indices = list(range(len(test_data)))
    sampler = SubsetRandomSampler(indices)
    test_loader = DataLoader(test_data, batch_size=1,sampler=sampler)
    print("test_data_finished!")
    if config.TRAIN.IF_TEST == 0:
        train_chache_data = VideoDataset(config, preprocess=preprocess, device=device, ann_file=config.DATA.TRAIN_FILE,shot=config.DATA.CACHE_SIZE,type='train_cache')
        indices = list(range(len(train_chache_data)))
        sampler = SubsetRandomSampler(indices)
        train_loader_cache = DataLoader(train_chache_data, batch_size=config.TRAIN.BATCH_SIZE,sampler=sampler)

        train_data_F = VideoDataset(config, preprocess=preprocess, device=device, ann_file=config.DATA.TRAIN_FILE,
                                         shot=config.DATA.SHOTS, type='train_F')
        indices = list(range(len(train_data_F)))
        sampler = SubsetRandomSampler(indices)
        train_load_F = DataLoader(train_data_F, batch_size=config.TRAIN.BATCH_SIZE, sampler=sampler)
        val_data, val_loader,_,_ = split_dataset(train_data_F,config.TRAIN.BATCH_SIZE)
        print("val_data finished!")

        # Add new train_data_a and train_load_a
        train_data_a = VideoDataset(config, preprocess=preprocess, device=device, ann_file=config.DATA.TRAIN_FILE,
                                         shot=config.DATA.SHOTS, type='train_a')  # Change the type to 'train_a'
        indices = list(range(len(train_data_a)))
        sampler = SubsetRandomSampler(indices)
        train_load_a = DataLoader(train_data_a, batch_size=config.TRAIN.BATCH_SIZE, sampler=sampler)

        return train_chache_data, val_data, test_data,train_data_F,train_data_a, train_loader_cache, val_loader, test_loader,train_load_F, train_load_a
    else:
        return (None, None, test_data,None, None,
                None, None, test_loader,None, None)