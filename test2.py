import matplotlib.pyplot as plt
import torch
from torch import nn
from model.transformer import FSAttention
import torch.nn.functional as F


# 假设你有一个输入tensor
input_tensor = torch.randn(1,8,512)
# 创建一个FSAttention对象
fsattention = FSAttention(dim=512)
# 通过FSAttention对象获取输出和attention map
output, attention_map = fsattention(input_tensor)
# 假设你有一个原始图像序列，形状为[1, 8, 3, H, W]
original_images = torch.randn(1, 8, 3, 224, 224)
# 对注意力图进行插值，使其大小与原始图像相同
attention_map_resized = F.interpolate(attention_map, size=(224, 224), mode='jet', align_corners=False)
# 将注意力图叠加到原始图像上
attention_on_images = original_images * attention_map_resized
# 可视化结果
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(attention_on_images[0, i].detach().numpy().transpose(1, 2, 0))
plt.show()