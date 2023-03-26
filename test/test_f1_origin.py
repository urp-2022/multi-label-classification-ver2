MODEL_NAME = 'model_origin'
MAP_RESULT=''

f_map=open('../result/mAP/'+MODEL_NAME+'.txt','w')

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
import time
from model.model_origin_resnet import resnet34

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

MODEL_PATH = '../result/model/' + MODEL_NAME +'.h5'
BATCH_SIZE = 32

# test dataset
test_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor()])

voc = VOC(batch_size=BATCH_SIZE, year="2007")
test_loader = voc.get_loader(transformer=test_transformer, datatype='test')

# load model
model = resnet34().to(device)
num_classes = 20
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

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
from sklearn.metrics import f1_score, classification_report

preds = []
targets = []

with torch.no_grad():
    for input, target in test_loader:
        input = input.to(device)
        output = model(input)
        output = torch.sigmoid(output).cpu() #shape=torch.Size([32, 1])
        output = torch.round(output)
        preds.append(output.cpu())
        targets.append(target.cpu())

T=torch.cat(targets).numpy()
Y=torch.cat(preds).numpy()
T = np.int_(T)
Y = np.int_(Y)

f1 = f1_score(T, Y, average=None).mean()

# mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
# print(mAP_score)
# for i in range(20):
#   print(VOC_CLASSES[i]," mAP score:", mAP_score[i])
#   MAP_RESULT+=str(VOC_CLASSES[i])+" "+str(mAP_score[i])+"\n"
  
# print("\nTotal mAP: ",100 * mAP_score.mean())
# MAP_RESULT+="\n\nTotal mAP: "+str(100 * mAP_score.mean())

# f_map.write(MAP_RESULT)
# f_map.close()

print(classification_report(T, Y, target_names=list(VOC_CLASSES)))
print("Total F1 score: {}".format(f1))

