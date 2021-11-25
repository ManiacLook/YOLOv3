from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80  # coco 数据集
classes = load_classes("data/coco.names")

# Set up the neural network
print("Loading network ......")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()  # eval() 自动把BN和DropOut固定住，不会取平均，而是用训练好的值

read_dir = time.time()
# Detection phase
try:
    # 图像路径列表
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

# 若保存目录不存在，就创建该目录
if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
# 加载的图像列表
loaded_ims = [cv2.imread(x) for x in imlist]

# PyTorch Variables for images
# 将加载的图像转换为(批次 x 通道 x 高度 x 宽度)
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
# List containing dimensions of original images
# 存储原始图像维度
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

if CUDA:
    im_dim_list = im_dim_list.cuda()

# 将图像分批次存在 im_batches
leftover = 0
if len(im_dim_list) % batch_size:
    leftover = 1
if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i * batch_size:min((i + 1) * batch_size, len(im_batches))]))
                  for i in range(num_batches)]

write = 0
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    # load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    # 前向传播
    prediction = model(Variable(batch), CUDA)
    # 目标得分阈值化和非最大值抑制
    prediction = write_result(prediction, confidence, num_classes, nms_conf=nms_thresh)
    end = time.time()

    if type(prediction) == int:
        # 遍历当前批次图像路径
        for im_num, image in enumerate(imlist[i * batch_size:min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num  # 图像索引
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue  # 只打印一次
    # transform the attribute from index in batch to index in imlist
    # 将 prediction 中的索引属性转换成 imlist 中的索引
    prediction[:, 0] += i * batch_size

    if not write:  # If we haven't initialised output
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(imlist[i * batch_size:min((i + 1) * batch_size, len(imlist))]):
        im_id = i * batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]  # 检测类别结果
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()

try:
    output
except NameError:
    print("No detections were made")
    exit()

# 获得原始图像的尺寸列表
im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
# 填充图像相对于原始图像的缩放比例列表
scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)
# 去掉填充区域
output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

output[:, 1:5] /= scaling_factor

# 将在图像外部具有边界的边界框剪裁到图像的边缘
for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

output_recast = time.time()
class_load = time.time()
# RGB颜色列表
colors = pkl.load(open("pallete", "rb"))

draw = time.time()

def write(x, results, color):
    # 坐标
    c1 = tuple([int(x[1]), int(x[2])])
    c2 = tuple([int(x[3]), int(x[4])])
    # 图像
    img = results[int(x[0])]
    # 类别
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    return img

list(map(lambda x: write(x, loaded_ims, colors[0]), output))

det_names = pd.Series(imlist).apply(lambda x: "{}\det_{}".format("\\".join(x.split("\\")[:-1]) + "\\" + args.det, x.split("\\")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))
end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()