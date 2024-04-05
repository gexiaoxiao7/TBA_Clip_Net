from clip import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import clip
from model.tp import Attention
import cv2


def load_clip(backbone_name, device):
    model, preprocess = clip.load(backbone_name, device=device)
    return model, preprocess

class VideoEncoder(nn.Module):
    def __init__(self, clip_model, preprocess,config,device):
        super().__init__()
        self.model = clip_model
        self.preprocess = preprocess
        self.dtype = clip_model.dtype
        self.device = device
        self.config = config
    def forward(self, images):
        video_info = images
        video_info = [torch.from_numpy(x).to(self.device).type(self.dtype) for x in video_info]
        image_features = [self.model.encode_image(x) for x in video_info]
        image_features = torch.stack(image_features, dim=1).to(torch.half)
        attention_format_features = image_features
        video_feature = torch.mean(image_features, dim=1)
        video_features = torch.unsqueeze(video_feature, 0)
        if self.config.TEMPORAL_POOLING == 'mean':
            return video_features, None
        else:
            return video_features, attention_format_features

class Prompts_build(nn.Module):
    def __init__(self, classnames,device, config):
        super().__init__()
        self.classnames = classnames
        self.device = device
        self.config = config
    def construct_prompts(self, ctx):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        prompt = [x + ctx + self.config.SUFFIX for x in self.config.PREFIX]
        return prompt
    def forward(self):
        prompts = [self.construct_prompts(x) for x in self.classnames]
        return prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model,device):
        super().__init__()
        self.tokenize = clip.tokenize
        self.model = clip_model
        self.device = device
        self.dtype = clip_model.dtype
    def forward(self, prompts):
        x = [clip.tokenize(prompt).to(self.device) for prompt in prompts]
        clip_weights = [self.model.encode_text(i) for i in x]
        # x = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)
        clip_weights = torch.stack(clip_weights)
        clip_weights = clip_weights.mean(dim=1, keepdim=True)
        clip_weights = clip_weights.squeeze(dim=1)
        return clip_weights



class TBA_Clip(nn.Module):
    def __init__(self, clip_model, preprocess, classnames, device,config):
        super().__init__()
        self.model = clip_model
        self.attention = Attention(feature_dim=clip_model.visual.output_dim).to(device).to(torch.half)
        self.prompts_learner = Prompts_build(classnames = classnames, device = device, config = config)
        self.preprocess = preprocess
        self.text_encoder = TextEncoder(clip_model,device)
        self.image_encoder = VideoEncoder(clip_model, preprocess, config,device)
        self.dtype = clip_model.dtype
        self.config = config
        self.device = device
        self.classnames = classnames
    def forward(self, image):
        prompts = self.prompts_learner()
        image_features,attention_format_features = self.image_encoder(image)
        text_features = self.text_encoder(prompts)
        norm = text_features.norm(dim=-1, keepdim=True)
        text_feature = text_features / norm
        norm = image_features.norm(dim=-1, keepdim=True)
        image_feature = image_features / norm
        similarity = (100.0 * image_feature @ text_feature.T).softmax(dim=-1)
        return similarity,image_features,text_features,attention_format_features

def returnCLIP(config,classnames,device):
    clip_model, preprocess = clip.load(config.MODEL.ARCH, device = device)
    model = TBA_Clip(clip_model, preprocess,classnames,device,config)
    return model