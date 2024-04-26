import torch
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("./base_model/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("./base_model/blip2-opt-2.7b")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# 注意，由于模型过大，我并没有把它放到GPU上
# 所以相应地在生成inputs的时候，也不要放在GPU上或者使用16精度计算
inputs = processor(images=raw_image, return_tensors="pt") # .to(device, torch.float16)
generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(generated_text)
# 'a rocky cliff with trees and water in the background'
