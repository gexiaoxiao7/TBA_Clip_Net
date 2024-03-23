import torch
import clip
import numpy as np
from PIL import Image
import cv2
from model.tp import TemporalPooling
import pandas as pd
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

classes_all = pd.read_csv('labels/hmdb51_org_base_labels.csv')
classnames = classes_all.values.tolist()
classnames = [class_name for i, class_name in classnames]
#修改classnames中的kiss为kissing
classnames[classnames.index('kiss')] = 'kissing'

# print(classnames)
def get_filenames_in_dir(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

directory = 'E:/DATASETS/hmdb51_org/kiss'  # 替换为你的目录路径
filenames = get_filenames_in_dir(directory)
def get_video_data(path):
    video_capture = cv2.VideoCapture(path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_ids = np.linspace(0, total_frames - 1, 8, dtype=np.int)
    for i in range(total_frames):
        ret, frame = video_capture.read()
        if not ret:
            break
        if i in frame_ids:
            frames.append(frame)
    video_capture.release()
    return frames



model, preprocess = clip.load("ViT-B/32", device=device)
np.set_printoptions(suppress=True, precision=6)

total = 0
acc = 0
for idx, filename in enumerate(filenames):
    path = os.path.join(directory, filename)
    image_input_lists = get_video_data(path)
    image_inputs = [preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device) for c in image_input_lists]
    image_features = [model.encode_image(x).to(device) for x in image_inputs]
    image_features = torch.stack(image_features, dim=1)

    # print(image_features.shape)
    temporal_pooling = TemporalPooling(feature_dim=image_features.shape[-1]).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in classnames]).to(device)
    with torch.no_grad():
        video_features = temporal_pooling(image_features)
        text_features = model.encode_text(text_inputs)
    video_features /= video_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * video_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(1)
    # print(indices)
    if classnames[indices.item()] == 'kissing':
        acc += 1
    total += 1
    if idx % 10 == 0:
        print(f'Accuracy: {acc/total*100:.2f}%')

print(f'Accuracy: {acc/total*100:.2f}%')
# print(image_input_lists)
# print(text_inputs.shape)

# text = clip.tokenize(["a teacher is writing","a teacher is not writing"]).to(device)

# with torch.no_grad():

# #
# #
# # Pick the top 5 most similar labels for the image
# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# print(similarity)
# values, indices = similarity[0]
# print(values)
# print(indices)
# print("\nTop predictions:\n")
# for value, index in zip(values, indices):
#     print(f"{text_inputs_list[index]:>16s}: {100 * value.item():.2f}%")