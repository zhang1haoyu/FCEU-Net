import os

import imgviz
import torchvision.transforms as transforms
from lib.sseg import SSegmentationSet
import torch
import numpy as np
from PIL import Image
import copy

#模型定义
n_classes = 6
# create model
net = SSegmentationSet(model='hst_unet_wf', num_classes=6, pretrained='', img_size=256)
checkpoint = r"D:\Semantic Segmentation\HST-UNet-master\checkpoint.pth"
if not checkpoint is None:
    print('载入模型')
    checkpoint = torch.load(checkpoint)
    net.load_state_dict(checkpoint['model'])
net.cuda()
net.eval()

#传入Img读取的一张图片 返回模板上色后的图片 8位 p模式
palette = [
    [255, 255, 255],  # Surfaces
    [0, 0, 255],      # Building
    [0, 255, 255],    # Low Veg
    [0, 255, 0],      # Tree
    [255, 255, 0],    # Car
    [255, 0, 0]       # Background
]

def flatten_palette(palette):
    flat_palette = []
    for color in palette:
        flat_palette.extend(color)
    return flat_palette
def predict_to_palette(img,palette):
    flat_palette = flatten_palette(palette)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.469, 0.321, 0.318], [0.22, 0.16, 0.153]),
    ])
    im = to_tensor(img)
    im = torch.unsqueeze(im, dim=0)  # 增加一个维度
    im = im.cuda()
    with torch.no_grad():
        # predict class
        logits = net(im)['out']
    label = im
    label = label.squeeze(1)
    size = label.size()[-2:]
    probs = torch.softmax(logits, dim=1) #logits 1,6,256,256
    preds = torch.argmax(probs, dim=1)
    preds = preds.squeeze(1)
    preds = preds.cpu()
    pr = np.array(preds).reshape(size)
    img2 = Image.fromarray(np.uint8(pr))
    img2.convert('P', palette=palette)
    img2.putpalette(flat_palette)
    return img2


def predict_to_vis(img):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.469, 0.321, 0.318], [0.22, 0.16, 0.153]),
    ])
    im = to_tensor(img)
    im = torch.unsqueeze(im, dim=0)  # 增加一个维度
    im = im.cuda()
    with torch.no_grad():
        # predict class
        logits = net(im)['out']
    label = im
    label = label.squeeze(1)
    size = label.size()[-2:]
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    preds = preds.squeeze(1)
    preds = preds.cpu()
    pr = np.array(preds).reshape(size)
    tmp_img = np.array(img)
    class_names = ["Surfaces", "Building", "Low Veg", "Tree", "Car", "Background"]
    viz = imgviz.label2rgb(
        label=pr,  # 预测结果
        img=imgviz.rgb2gray(tmp_img),  # 原图
        font_size=15,
        label_names=class_names,
        loc="rb",
    )
    return viz


img_path = r'D:\Semantic Segmentation\HST-UNet-master\pictures'
save_path1 = r'D:\Semantic Segmentation\HST-UNet-master\predict_viz\viz_1'
save_path2 = r'D:\Semantic Segmentation\HST-UNet-master\predict_viz\viz_2'
def predict_to_save(img_path,save_path1,save_path2):
    for dr in os.listdir(img_path):
        path = os.path.join(img_path,dr)
        img = Image.open(path)
        img2 = predict_to_palette(img,palette)
        tmp_dr = dr.split('.')[0] + '.png'
        img2.save(os.path.join(save_path1,tmp_dr))
    print('存储完毕！')
predict_to_save(img_path,save_path1,save_path2)
