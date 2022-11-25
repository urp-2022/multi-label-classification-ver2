import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
import time
from model_origin_resnet import resnet34

from PIL import Image
from datasets.loader import VOC

from helper_functions import mAP, CocoDetection

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

MODEL_PATH = '../result/model/model_origin.h5'
BATCH_SIZE = 32

# test dataset
test_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor()])

voc = VOC(batch_size=BATCH_SIZE, year="2007")
test_loader = voc.get_loader(transformer=test_transformer, datatype='test')

# load model
model = resnet34().to(device)

# load weight
model.load_state_dict(torch.load(MODEL_PATH))

# model eval
model = model.eval()

# tensor image generate
images = test_transformer(Image.open('test_img/cat2.jpg')).view(1, 3, 224, 224)
images = images.to(device)

# prediction

model=model.to(device)


pred = model(images)
pred_sigmoid = torch.sigmoid(pred)
# print(pred_sigmoid)
pred_rounded = torch.round(pred_sigmoid)
tmp=pred_rounded.cpu().detach().numpy()[0]

for i in range(20):
  if tmp[i]==1:
    print(VOC_CLASSES[i]) 

#mAP============================
from sklearn.metrics import average_precision_score, precision_recall_curve
np.seterr(invalid='ignore')

preds = []
targets = []

with torch.no_grad():
    for input, target in test_loader:
        input = input.to(device)
        output = model(input)
        output = torch.sigmoid(output).cpu() #shape=torch.Size([32, 1])
        preds.append(output.cpu())
        targets.append(target.cpu())

print(torch.cat(targets).numpy().shape)
mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
print(mAP_score)
for i in range(20):
  print(VOC_CLASSES[i]," mAP score:", mAP_score[i])
  
print("\nTotal mAP: ",100 * mAP_score.mean())
