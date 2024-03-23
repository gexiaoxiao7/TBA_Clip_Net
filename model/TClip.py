from clip import clip
import numpy as np
import torch
import torch.nn.functional as F
from model.tp import  TemporalPooling
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
    def forward(self,images):
        video_info = images
        image_inputs = [self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device) for c in video_info]
        image_features = [self.model.encode_image(x).to(self.device) for x in image_inputs]
        image_features = torch.stack(image_features, dim=1)
        temporal_pooling = TemporalPooling(feature_dim=image_features.shape[-1]).to(self.device)
        video_features = temporal_pooling(image_features)

        # image_input = self.preprocess(Image.fromarray(cv2.cvtColor(video_info, cv2.COLOR_BGR2RGB)).convert('RGB')).unsqueeze(0).to(self.device)
        # image_features = self.model.encode_image(image_input)
        return video_features

class TextEncoder(nn.Module):
    def __init__(self, clip_model, device):
        super().__init__()
        self.tokenize = clip.tokenize
        self.model = clip_model
        self.device = device
        self.dtype = clip_model.dtype
    def forward(self, prompts):
        x = torch.cat([self.tokenize(f"a photo of {prompt}") for prompt in prompts]).to(self.device)
        text_features = self.model.encode_text(x)
        return text_features


class Prompts_build(nn.Module):
    def __init__(self, classnames,device):
        super().__init__()
        self.classnames = classnames
        self.device = device
    def construct_prompts(self, ctx, prefix):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        prompts = torch.cat([clip.tokenize(prefix + x) for x in ctx]).to(self.device)
        return prompts

    def forward(self):
        prefix = 'a photo of'
        prompts = self.construct_prompts(self.classnames, prefix).to(self.device)
        return prompts
class TBA_Clip(nn.Module):
    def __init__(self, clip_model, preprocess, classnames, device):
        super().__init__()
        self.model = clip_model
        # self.prompts_learner = Prompts_build(classnames = classnames, device = device)
        self.preprocess = preprocess
        self.text_encoder = TextEncoder(clip_model,device)
        self.image_encoder = VideoEncoder(clip_model, preprocess, device)
        self.dtype = clip_model.dtype
        self.classnames = classnames
    def forward(self, image):
        # prompts = self.prompts_learner()
        video_features = self.image_encoder(image)
        text_features = self.text_encoder(self.classnames)
        video_features /= video_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * video_features @ text_features.T).softmax(dim=-1)
        return similarity


def returnCLIP(config,classnames,device):
    clip_model, preprocess = clip.load(config.MODEL.ARCH, device = device)
    model = TBA_Clip(clip_model, preprocess,classnames,device)
    return model