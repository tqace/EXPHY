from torch.utils.data import Dataset
import os
import random
from skimage.transform import rescale, rotate
from skimage import io
import numpy as np
import torch

class ComphyDataset(Dataset):

    def __init__(self,datapath,max_num_samples=60000,max_num_frames=6,down_sz=64):
        self.datapath = datapath
        self.max_num_samples = max_num_samples
        self.max_num_frames = max_num_frames
        self.down_sz = down_sz
        self.files = self.get_files()
        if len(self.files) < self.max_num_samples:
            self.max_num_samples = len(self.files)
    def __len__(self):
        return self.max_num_samples

    def __getitem__(self,idx):
        videopath = self.files[idx]
        all_frames = []
        all_imgs = os.listdir(videopath)
        all_imgs.sort(key = lambda x:int(x[:-4].split('_')[-1]))
        all_imgs = all_imgs[:self.max_num_frames]
        for imgname in all_imgs:
            imgpath = str(videopath) + "/" + imgname
            scaled_img = self.rescale_img(io.imread(imgpath))
            img = torch.tensor(scaled_img,dtype=torch.float32).permute((2,0,1))
            all_frames.append(img)
        return torch.stack(all_frames,dim=0)
    def rescale_img(self,img):
        H,W,C = img.shape
        down = rescale(img, [self.down_sz/H], order=3, mode='reflect', multichannel=True)
        return down
    def get_files(self):
        paths = []
        total_videos = os.listdir(self.datapath)
        for video in total_videos:
            video_path = os.path.join(self.datapath,video)
            paths.append(video_path)
        return paths
