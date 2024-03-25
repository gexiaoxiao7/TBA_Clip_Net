import pandas as pd
from torch.utils.data import DataLoader
from model.tld import TeacherDetection
import numpy as np
from PIL import Image
import cv2
import torch
import clip

class VideoDataset():
    def __init__(self,config,preprocess,device,ann_file):
        self.labels_file = config.DATA.LABEL_LIST[1] if config.DATA.INSTRUCTION_FT else config.DATA.LABEL_LIST[0]
        self.ann_file = ann_file
        self.data_prefix = config.DATA.ROOT
        self.num_frames = config.DATA.NUM_FRAMES
        self.input_size = config.DATA.INPUT_SIZE
        self.yolo_model = config.MODEL.YOLO
        self.preprocess = preprocess
        self.device = device
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
        frame_ids = np.linspace(0, total_frames - 1, self.num_frames)
        for i in range(total_frames):
            ret, frame = video_capture.read()
            if not ret:
                break
            if i in frame_ids:
                frames.append(frame)
        video_capture.release()
        if self.if_teacher:
            for i in range(len(frames)):
                frames[i] = self.detector(frames[i])
        frames = [
            self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device) for c in
            frames]
        return frames

    def load_annotations(self):
        video_infos = []
        total_lines = sum(1 for line in open(self.ann_file, 'r'))
        with open(self.ann_file, 'r') as fin:
            for idx, line in enumerate(fin):
                if idx % 500 == 0 and idx != 0:
                    progress = (idx / total_lines) * 100
                    print(f'Processed {idx} samples, progress: {progress:.2f}%')
                line_split = line.strip().split()
                filename, label = line_split
                label = int(label)
                data = self.prepare_frames(self.data_prefix + filename)
                video_infos.append(dict(filename=filename, label=label, data=data))
        return video_infos

    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, idx):
        return self.video_info[idx]


def build_dataloader(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load(config.MODEL.ARCH, device=device)
    train_data = VideoDataset(config,preprocess = preprocess,device = device,ann_file=config.DATA.TRAIN_FILE)
    train_loader = DataLoader(train_data,batch_size=config.TRAIN.BATCH_SIZE)
    print("train_data finished!")
    val_data = VideoDataset(config,preprocess = preprocess,device = device,ann_file=config.DATA.VAL_FILE)
    val_loader = DataLoader(val_data,batch_size=1)
    print("val_data_finished!")
    return train_data, val_data, train_loader, val_loader