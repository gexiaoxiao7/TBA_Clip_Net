from clip import clip
import numpy as np
import torch
import torch.nn.functional as F
from model.tp import TemporalPooling
from torch import nn
from PIL import Image
import clip
import cv2


def load_clip(backbone_name, device):
    model, preprocess = clip.load(backbone_name, device=device)
    return model, preprocess

class VideoEncoder(nn.Module):
    def __init__(self, clip_model, preprocess, device):
        super().__init__()
        self.model = clip_model
        self.preprocess = preprocess
        self.dtype = clip_model.dtype
        self.device = device

    def forward(self, images):
        video_info = images
        video_info = [torch.from_numpy(x).to(self.device).type(self.dtype) for x in video_info]
        image_features = [self.model.encode_image(x) for x in video_info]
        image_features = torch.stack(image_features, dim=1).to(torch.half)
        temporal_pooling = TemporalPooling(feature_dim=image_features.shape[-1],nhead=8,num_layers=12).to(self.device).to(torch.half)
        video_features = temporal_pooling(image_features)
        return video_features

class Prompts_build(nn.Module):
    def __init__(self, classnames,device):
        super().__init__()
        self.classnames = classnames
        self.device = device
    def construct_prompts(self, ctx, prefix, suffix):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        prompt = prefix + ctx + suffix
        return prompt
    def forward(self):
        prefix = 'a photo of'
        suffix = ''
        prompts = [self.construct_prompts(x, prefix, suffix) for x in self.classnames]
        return prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model,device):
        super().__init__()
        self.tokenize = clip.tokenize
        self.model = clip_model
        self.device = device
        self.dtype = clip_model.dtype
    def forward(self, prompts):
        x = torch.cat([self.tokenize(prompt) for prompt in prompts]).to(self.device)
        text_features = self.model.encode_text(x)
        return text_features



class TBA_Clip(nn.Module):
    def __init__(self, clip_model, preprocess, classnames, device,config):
        super().__init__()
        self.model = clip_model
        self.prompts_learner = Prompts_build(classnames = classnames, device = device)
        self.preprocess = preprocess
        self.text_encoder = TextEncoder(clip_model,device)
        self.image_encoder = VideoEncoder(clip_model, preprocess, device)
        self.dtype = clip_model.dtype
        self.config = config
        self.device = device
        self.classnames = classnames
    def forward(self, image):
        prompts = self.prompts_learner()
        image_feature = self.image_encoder(image) if self.config.TRAINER.TRANS_FRAMES == 1 else (
            self.model.encode_image(torch.from_numpy(image).to(self.dtype).to(self.device)))
        text_features = self.text_encoder(prompts)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_feature @ text_features.T).softmax(dim=-1)
        return similarity

def returnCLIP(config,classnames,device):
    clip_model, preprocess = clip.load(config.MODEL.ARCH, device = device)
    model = TBA_Clip(clip_model, preprocess,classnames,device,config)
    return model