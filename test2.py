import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
from IPython.core.display import display


img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
display(raw_image.resize((596, 437)))

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# we associate a model with its preprocessors to make it easier for inference.
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
)
vis_processors.keys()
# dict_keys(['train', 'eval'])

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

model.generate({"image": image})

