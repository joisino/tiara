from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


model = models.resnet18(pretrained=True)
model.eval()

modules = list(model.children())[:-1]
extractor = nn.Sequential(*modules)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

with open('openimage_id.txt') as f:
    images = list(sorted([r.split('/')[1].strip() for r in f]))

output = {}
for i in tqdm(images):
    input_image = Image.open('imgs/' + i + '.jpg').convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output[i] = model(input_batch).reshape(-1).numpy()

with open('openimage_output.pickle', 'wb') as f:
    pickle.dump(output, f)
