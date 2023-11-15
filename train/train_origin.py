MODEL_NAME = 'resnet101/model_origin_4'
MODEL_DESCRIPTION = ''

f_desc=open('../result/model_description/'+MODEL_NAME+'.txt','w')
f_train=open('../result/train_loss/'+MODEL_NAME+'.txt','w')
f_valid=open('../result/valid_loss/'+MODEL_NAME+'.txt','w')


import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tkinter import image_names
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.model_origin_resnet import resnet101,resnet50, resnet34, resnet18
import torchvision.transforms as transforms
from datasets.loader import VOC

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)
MODEL_PATH = '../result/model/'+MODEL_NAME+'.h5'
BATCH_SIZE = 16
EPOCH = 100

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)


train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),])

valid_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),])

voc = VOC(batch_size=BATCH_SIZE, year="2007")
train_loader = voc.get_loader(transformer=train_transformer, datatype='train')
valid_loader = voc.get_loader(transformer=valid_transformer, datatype='val')

    
# load model
model = resnet101(pretrained=True).to(device)
num_classes = 20
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.to(device)

# for name, child in model.named_children():
#     for param in child.parameters():
#         print(name)

# Momentum / L2 panalty
optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-5, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                           milestones=[30, 80],
                                           gamma=0.1)
criterion = nn.BCEWithLogitsLoss()

best_loss = 100
train_iter = len(train_loader)
valid_iter = len(valid_loader)


for e in range(EPOCH):
    print("epoch : "+str(e))
    train_loss = 0
    valid_loss = 0

    for i, (images, targets) in tqdm(enumerate(train_loader), total=train_iter):
        images = images.to(device)
        targets = targets.to(device)

        model = model.to(device)
        pred = model(images)
        # print(pred)
        # print(targets)
        loss = criterion(pred.double(), targets)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    total_train_loss = train_loss / train_iter
    scheduler.step()

    with torch.no_grad():
        for images, targets in valid_loader:
            images = images.to(device)
            targets = targets.to(device)

            pred = model(images)
            loss = criterion(pred.double(), targets)
            valid_loss += loss.item()

    total_valid_loss = valid_loss / valid_iter
    
    f_train.write("epoch "+str(e)+" : "+str(total_train_loss)+"\n")
    f_valid.write("epoch "+str(e)+" : "+str(total_valid_loss)+"\n")
    print("[train loss / %f] [valid loss / %f]" % (total_train_loss, total_valid_loss))

    if best_loss > total_valid_loss:
        print("model saved\n")
        torch.save(model.state_dict(), MODEL_PATH)
        best_loss = total_valid_loss

MODEL_DESCRIPTION+='BATCH_SIZE:'+str(BATCH_SIZE)+'\nEPOCH: '+str(EPOCH)+"\n\n"
MODEL_DESCRIPTION+='optimizer:\n'+str(optimizer.state_dict)+"\n\n"
MODEL_DESCRIPTION+="scheduler:\n"+str(scheduler.state_dict())+"\n\n"

f_desc.write(MODEL_DESCRIPTION)
f_desc.close()
f_train.close()
f_valid.close()
