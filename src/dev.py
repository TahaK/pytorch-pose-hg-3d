
# coding: utf-8

# In[1]:


import sys
import torch

import matplotlib
matplotlib.use('Agg')
import random
from opts import opts
import ref
import glob
from utils.eval import getPreds
import cv2
import numpy as np

from utils.debugger import Debugger
import os
import json
import warnings

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
warnings.filterwarnings('ignore')


# In[2]:


from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-video_dir", default='../images/',
                    help="Directory of the source video")
parser.add_argument("-pose_dir", default='../images/',
                    help="Directory of the processed poses")
parser.add_argument("-output_dir", default='debug/',
                    help="Directory for the outputs")
parser.add_argument("-num_videos", type=int, default=6,
                    help="Directory for the outputs")

args = parser.parse_args()
video_dir = args.video_dir
pose_dir = args.pose_dir
output_dir = args.output_dir

print(video_dir)
print(pose_dir)
print(output_dir)
# In[3]:

def chose_right(pose):
    index = 0
    xmax = 0
    for i in range(len(pose['people'])):
        x = np.array(pose['people'][i]['pose_keypoints'][1::3])
        x_max = np.max(x[np.nonzero(x)])
        if x_max > xmax:
            xmax = x_max
            index = i
    return index


def get_boundries(pose,frame_width,frame_height):
    if len(pose['people']) > 1:
        index = chose_right(pose)
    else:
        index = 0
    
    x = np.array(pose['people'][index]['pose_keypoints'][1::3])
    x_min = np.min(x[np.nonzero(x)])
    x_max = np.max(x[np.nonzero(x)])
    y = np.array(pose['people'][index]['pose_keypoints'][0::3])
    y_min = np.min(y[np.nonzero(y)])
    y_max = np.max(y[np.nonzero(y)])
    
    center_x = (x_min + x_max)/2
    center_y = (y_min + y_max)/2

    width = x_max - x_min
    height = y_max - y_min
    
    length = max(width,height)
    
    return max(int(center_x - length/2) - 30, 0), min(int(center_x + length/2)+ 30,frame_width),max(int(center_y - length/2)- 30,0), min(int(center_y + length/2)+ 30,frame_height)


files = [os.path.basename(x)[:-4] for x in glob.glob(pose_dir+"*.avi")]
# predefined list ['vid0534_0530_20160115','vid0534_0533_20160115','vid0054_0573_20160809','vid0058_0715_20171004','vid0281_8597_20170316','vid0534_8911_20170726']


for i in range(args.num_videos):
    basename = random.choice(files)

    model = torch.load('../models/hgreg-3d.pth', map_location=lambda storage, loc: storage, pickle_module=pickle).cuda()
    poses =  sorted(glob.glob(pose_dir+basename+'_poses/*.json'), key=os.path.basename)
    vidcap = cv2.VideoCapture(video_dir+basename+'.mp4')
    # directory = input_dir'+ basename + '/'
    success,image = vidcap.read()
    count = 0
    success = True
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
    writer = FFMpegWriter(fps=30, metadata=metadata)
    fig = plt.figure(figsize=(9, 4.5))
    with writer.saving(fig, output_dir+basename+'.mp4',100):
        print( output_dir+basename+'.mp4')
        while success:
            success,image = vidcap.read()
            pose = json.load(open(poses[count]))
            if len(pose['people']) > 0:
                x1,x2,y1,y2 =get_boundries(pose,image.shape[0],image.shape[1])
                img = cv2.resize(image[x1:x2,y1:y2], (256, 256)) #[:,-464:]
                input = torch.from_numpy(img.transpose(2, 0, 1)).float() / 256.
                input = input.view(1, input.size(0), input.size(1), input.size(2))
                input_var = torch.autograd.Variable(input).float().cuda()
                output = model(input_var)
                pred = getPreds((output[-2].data).cpu().numpy())[0] * 4
                reg = (output[-1].data).cpu().numpy().reshape(pred.shape[0], 1)
                debugger = Debugger(fig)
                debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
                debugger.addPoint2D(pred, (255, 0, 0))
                debugger.addPoint3D(np.concatenate([pred, (reg + 1) / 2. * 256], axis = 1))
                debugger.show3D()
                debugger.showImg()
                writer.grab_frame()
            count += 1
