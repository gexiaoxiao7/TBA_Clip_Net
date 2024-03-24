import torch
import clip
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

def image_segmentation(path):
    model = YOLO('Yolo-model/yolov8n.pt')
    image = cv2.imread(path)
    results = model(image)
    dots = np.array(results[0].to('cpu').boxes.data)

    # 加个判断，提取出老师的图像，避免学生图像干扰
    temp = 0
    p1 = -1

    for idx,item in enumerate(dots):
        if item[5] == 0 : # 判断是不是person
            x1, y1, x2, y2 = item[:4]  # 提取检测框的坐标
            if abs(y2-y1) > temp:
                temp = abs(y2-y1)
                p1 = idx
    x1, y1, x2, y2 = dots[p1][:4]
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]  # 裁剪图像
    return cropped_image



def pose_estimation(cropped_image,texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    np.set_printoptions(suppress=True, precision=6)
    image = preprocess(Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    text = clip.tokenize(texts).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    # 使用该下标从 B 中获取对应的值
    result = probs[np.argmax(probs)]
    return result


if __name__ == "__main__":
    cropped_image = image_segmentation('teacher_blackboard.jpg')
    text = ['a person is writing on the blackboard', 'a person is not writing on the blackboard']
    print(pose_estimation(cropped_image,text))
