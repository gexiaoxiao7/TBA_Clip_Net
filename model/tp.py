import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention_fc = nn.Linear(feature_dim, 1).to(torch.float16)

    def forward(self, x):
        # x shape: [N, T, D]
        attention_weights = self.attention_fc(x)  # shape: [N, T, 1]
        attention_weights = F.softmax(attention_weights, dim=1)  # shape: [N, T, 1]
        return attention_weights

class TemporalPooling(nn.Module):
    def __init__(self, feature_dim,config):
        super(TemporalPooling, self).__init__()
        self.attention = Attention(feature_dim)
        self.config = config
        self.conv1d = nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1)  # 初始化Conv1d层

    def forward(self, x):
        if self.config.TEMPORAL_POOLING == 'mean':
            video_feature = torch.mean(x, dim=1)
            video_feature = torch.unsqueeze(video_feature, 0)
        elif self.config.TEMPORAL_POOLING == 'attention':
            attention_weights = self.attention(x)
            video_feature = torch.sum(attention_weights * x, dim=1)
            video_feature = torch.unsqueeze(video_feature, 0)
        elif self.config.TEMPORAL_POOLING == 'conv1d':
            x = x.permute(0, 2, 1)  # Conv1d expects inputs in the shape [N, D, T]
            x = self.conv1d(x)  # Apply Conv1d
            video_feature = torch.mean(x, dim=2)  # Average pooling
            video_feature = torch.unsqueeze(video_feature, 0)
        else:
            raise NotImplementedError
        return video_feature