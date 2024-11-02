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
import torch.distributed as dist
import os


class VideoDataset():
    def __init__(self,config,preprocess,device,ann_file,logger,shot=0,type = 'train'):
        self.labels_file = config.DATA.LABEL_LIST
        self.ann_file = ann_file
        self.logger = logger
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
        if not os.path.exists(path):
            self.logger.info(f"File {path} not found.")
            return None
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
        class_counts = {}
        total_lines = sum(1 for line in open(self.ann_file, 'r'))
        if self.type == 'train_cache':
            with open(self.ann_file, 'r') as fin:
                lines = fin.readlines()
                # start_idx = int(total_lines * 2 / 3)  # Calculate the start index
                for idx in range(total_lines):  # Start from the last third
                    if idx % 1 == 0 and idx != 0:
                        progress = (idx / total_lines) * 100
                        self.logger.info(f'Processed {idx} samples, progress: {progress:.2f}%')
                    line = lines[total_lines - idx - 1]
                    line_split = line.strip().split()
                    filename, label = line_split
                    label = int(label)
                    if label in class_counts and class_counts[label] >= self.shot:
                        continue
                    data = self.prepare_frames(self.data_prefix + filename)
                    if data is not None:
                        video_infos.append(dict(filename=filename, label=label, data=data))
                        if label not in class_counts:
                            class_counts[label] = 1
                        else:
                            class_counts[label] += 1
        elif self.type == 'train_a':
            with open(self.ann_file, 'r') as fin:
                lines = fin.readlines()
                # start_idx = int(total_lines * 1 / 3)  # Calculate the start index
                for idx in range(total_lines):  # Start from the last third
                    if idx % 1 == 0 and idx != 0:
                        progress = (idx / total_lines) * 100
                        self.logger.info(f'Processed {idx} samples, progress: {progress:.2f}%')
                    line = lines[total_lines - idx - 1]
                    line_split = line.strip().split()
                    filename, label = line_split
                    label = int(label)
                    if label in class_counts and class_counts[label] >= self.shot:
                        continue
                    data = self.prepare_frames(self.data_prefix + filename)
                    if data is not None:
                        video_infos.append(dict(filename=filename, label=label, data=data))
                        if label not in class_counts:
                            class_counts[label] = 1
                        else:
                            class_counts[label] += 1
        else:
            with open(self.ann_file, 'r') as fin:
                for idx, line in enumerate(fin):
                    if idx % 1 == 0 and idx != 0:
                        progress = (idx / total_lines) * 100
                        self.logger.info(f'Processed {idx} samples, progress: {progress:.2f}%')
                    line_split = line.strip().split()
                    filename, label = line_split
                    label = int(label)
                    if self.type == 'train_F':
                        if label in class_counts and class_counts[label] >= self.shot:
                            continue
                        data = self.prepare_frames(self.data_prefix + filename)
                        if data is not None:
                            video_infos.append(dict(filename=filename, label=label, data=data))
                            if label not in class_counts:
                                class_counts[label] = 1
                            else:
                                class_counts[label] += 1
                    else:
                        data = self.prepare_frames(self.data_prefix + filename)
                        if data is not None:
                            video_infos.append(dict(filename=filename, label=label, data=data))
                            if label not in class_counts:
                                class_counts[label] = 1
                            else:
                                class_counts[label] += 1
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

# TODO：都读成多线程？
def build_dataloader(config,logger):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load(config.MODEL.ARCH, device=device)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.TIP_ADAPTER.LOAD_PRE_FEAT == 0:
        test_data = VideoDataset(config, preprocess=preprocess, device=device, ann_file=config.DATA.TEST_FILE,type='test',logger=logger)
        indices = np.arange(dist.get_rank(), len(test_data), dist.get_world_size())
        sampler_test = SubsetRandomSampler(indices)
        test_loader = DataLoader(test_data, batch_size=config.TRAIN.BATCH_SIZE,sampler=sampler_test
                                 ,num_workers=16, pin_memory=True, drop_last=True)
    else:
        test_data = None
        test_loader = None
    logger.info("test_data_finished!")

    if config.TRAIN.IF_TEST == 0:
        train_chache_data = VideoDataset(config, preprocess=preprocess, device=device, ann_file=config.DATA.TRAIN_FILE,shot=config.DATA.CACHE_SIZE,type='train_cache',logger=logger)
        sampler_train_cache = torch.utils.data.DistributedSampler(
            train_chache_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        train_loader_cache = DataLoader(
            train_chache_data, sampler=sampler_train_cache,
            batch_size=config.TRAIN.BATCH_SIZE,
            num_workers=16,
            pin_memory=True,
            drop_last=True
        )

        train_data_F = VideoDataset(config, preprocess=preprocess, device=device, ann_file=config.DATA.TRAIN_FILE,
                                         shot=config.DATA.SHOTS, type='train_F', logger=logger)
        sampler_train_F = torch.utils.data.DistributedSampler(
            train_data_F, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        train_load_F = DataLoader(
            train_data_F, sampler=sampler_train_F,
            batch_size=config.TRAIN.BATCH_SIZE,
            num_workers=16,
            pin_memory=True,
            drop_last=True
        )
        val_data, _ = split_dataset(train_data_F)
        indices = np.arange(dist.get_rank(), len(val_data), dist.get_world_size())
        sampler_val = SubsetRandomSampler(indices)
        val_loader = DataLoader(val_data, batch_size=config.TRAIN.BATCH_SIZE,sampler=sampler_val
                                 ,num_workers=16, pin_memory=True, drop_last=True)
        logger.info("val_data finished!")

        # Add new train_data_a and train_load_a
        train_data_a = VideoDataset(config, preprocess=preprocess, device=device, ann_file=config.DATA.TRAIN_FILE,
                                         shot=config.DATA.SHOTS, type='train_a',logger=logger)  # Change the type to 'train_a'
        sampler_train_a = torch.utils.data.DistributedSampler(
            train_data_a, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        train_load_a = DataLoader(
            train_data_a, sampler=sampler_train_a,
            batch_size=config.TRAIN.BATCH_SIZE,
            num_workers=16,
            pin_memory=True,
            drop_last=True
        )
        return train_chache_data, val_data, test_data,train_data_F,train_data_a, train_loader_cache, val_loader, test_loader,train_load_F, train_load_a
    else:
        return (None, None, test_data,None, None,
                None, None, test_loader,None, None)