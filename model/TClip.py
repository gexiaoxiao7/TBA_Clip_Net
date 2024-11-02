import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import clip
from collections import OrderedDict
from model.tp import Attention
import cv2
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

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


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model,device):
        super().__init__()
        n_cls = len(classnames)
        n_ctx_pre = cfg.TEXT_PROMPT.N_CTX_PRE
        n_ctx_post = cfg.TEXT_PROMPT.N_CTX_POST
        ctx_pre_init = cfg.TEXT_PROMPT.CTX_PRE_INIT
        ctx_post_init = cfg.TEXT_PROMPT.CTX_POST_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        if ctx_pre_init:
            # use given words to initialize context vectors
            ctx_pre_init = ctx_pre_init.replace("_", " ")
            n_ctx_pre = len(ctx_pre_init.split(" "))
            prompt = clip.tokenize(ctx_pre_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_pre_vectors = embedding[0, 1: 1 + n_ctx_pre, :]
            prompt_prefix = ctx_pre_init
        else:
            # random initialization
            ctx_pre_vectors = torch.empty(n_ctx_pre, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_pre_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx_pre)

        if ctx_post_init:
            ctx_post_init = ctx_post_init.replace("_", " ")
            n_ctx_post = len(ctx_post_init.split(" "))
            prompt = clip.tokenize(ctx_post_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_post_vectors = embedding[0, 1: 1 + n_ctx_post, :]
            prompt_suffix = ctx_post_init
        else:
            ctx_post_vectors = torch.empty(n_ctx_post, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_post_vectors, std=0.02)
            prompt_suffix = " ".join(["X"] * n_ctx_post)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx_pre}")
        print(f'Final context: "{prompt_suffix}"')
        print(f"Number of context words (tokens): {n_ctx_post}")


        self.ctx_pre = nn.Parameter(ctx_pre_vectors)
        self.ctx_post = nn.Parameter(ctx_post_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ])).to(device).to(torch.half)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + " " + prompt_suffix for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p).to(device) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_cls", embedding[:, 1 + n_ctx_pre: -1 - n_ctx_post , :])  # CLS
        self.register_buffer("token_suffix", embedding[:, -1:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx_pre = n_ctx_pre
        self.n_ctx_post = n_ctx_post
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.device = device

    def construct_prompts(self, pre_ctx, cls, post_ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                pre_ctx,  # (dim0, n_ctx, dim)
                cls,  # (dim0, n_cls, dim)
                post_ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        im_features = im_features.reshape(-1, im_features.shape[-1])
        prefix = self.token_prefix
        cls = self.token_cls
        suffix = self.token_suffix
        ctx_pre = self.ctx_pre  # (n_ctx, ctx_dim)
        ctx_post = self.ctx_post  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx_pre = ctx_pre.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_post = ctx_post.unsqueeze(0)
        ctx_shifted_pre = ctx_pre.to(self.device) + bias.to(self.device)  # (batch, n_ctx, ctx_dim)
        ctx_shifted_post = ctx_post.to(self.device) + bias.to(self.device)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i_pre,ctx_shifted_i_post in zip(ctx_shifted_pre, ctx_shifted_post):
            ctx_i_pre = ctx_shifted_i_pre.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_i_post = ctx_shifted_i_post.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i_pre, cls, ctx_i_post, prefix,suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts # [batch, n_cls, n_tkn, ctx_dim]

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class TBA_Clip(nn.Module):
    def __init__(self, clip_model, preprocess, classnames, device,config):
        super().__init__()
        self.model = clip_model
        self.attention = Attention(feature_dim=clip_model.visual.output_dim).to(device).to(torch.half)
        self.prompts_learner = PromptLearner(config, classnames, clip_model,device).to(torch.half)
        self.tokenized_prompts = self.prompts_learner.tokenized_prompts
        self.preprocess = preprocess
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = VideoEncoder(clip_model, preprocess, config,device)
        self.dtype = clip_model.dtype
        self.config = config
        self.device = device
        self.classnames = classnames
        self.logit_scale = clip_model.logit_scale
    def forward(self, image):
        image_features,attention_format_features = self.image_encoder(image)
        norm = image_features.norm(dim=-1, keepdim=True)
        image_feature = image_features / norm
        if self.config.TRAIN.LP == 1:
            tokenized_prompts = self.tokenized_prompts
            logit_scale = self.logit_scale.exp()
            prompts = self.prompts_learner(image_feature)
            logits = []
            for pts_i, imf_i in zip(prompts, image_feature):
                text_features = self.text_encoder(pts_i, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = (logit_scale * imf_i @ text_features.t()).softmax(dim=-1)
                logits.append(l_i)
            logits = torch.stack(logits)
            return logits,image_features,text_features,attention_format_features
        else:
            x = [clip.tokenize(label).to(self.device) for label in self.classnames]
            clip_weights = [self.model.encode_text(i) for i in x]
            clip_weights = torch.stack(clip_weights)
            clip_weights = clip_weights.squeeze(dim=1)
            text_features = clip_weights
            norm = text_features.norm(dim=-1, keepdim=True)
            text_feature = text_features / norm
            similarity = (100.0 * image_feature @ text_feature.T).softmax(dim=-1)
            return similarity,image_features,text_features,attention_format_features

def returnCLIP(config,classnames,device):
    clip_model, preprocess = clip.load(config.MODEL.ARCH, device = device)
    model = TBA_Clip(clip_model, preprocess,classnames,device,config)
    return model