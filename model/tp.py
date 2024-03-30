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
    def __init__(self, feature_dim):
        super(TemporalPooling, self).__init__()
        self.attention = Attention(feature_dim)

    def forward(self, x):
        # x shape: [N, T, D]
        attention_weights = self.attention(x)  # shape: [N, T, 1]
        video_feature = torch.sum(attention_weights * x, dim=1)  # shape: [N, D]
        video_feature = torch.unsqueeze(video_feature, 0)  # 添加一个维度，将形状变为[1, D]
        return video_feature