#%%
import spatial_transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional
from torchvision import datasets, models, transforms
from collections.abc import Iterable
from torch import FloatTensor,LongTensor
import random

def to_pil(img):
    if torch.is_tensor(img):
        return torchvision.transforms.functional.to_pil_image(img,mode="F")
    if isinstance(img,np.ndarray):
        return Image.fromarray(img, mode="F")
    return img


class sample_RandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.image_transform = spatial_transforms.RandomHorizontalFlip(p)
        
    def _skeleton_transform(self,skeleton_position):
        if self.image_transform.random_p < self.image_transform.p:
           skeleton_position = torch.tensor([  [self.shape[0] - point[0].item(),point[1].item()]  for point in skeleton_position])
        return skeleton_position

    def __call__(self,images,image_points, image_resamples = None ):
        self.image_transform.randomize_parameters()
        try:
            self.shape = images[0].shape
        except:
            width, height = images[0].size
            self.shape = np.array([width,height])
        out = []
        for img in images:
            img = to_pil(img)
            temp = self.image_transform(img)
            out.append(temp)

        out_points = []
        for points in image_points:
            temp = self._skeleton_transform(points)
            out_points.append(temp)
        
        return out, out_points

class sample_resize():
    def __init__(self, size):
        self.size = size


    def _skeleton_transform(self,skeleton_position):
        width,height = self.shape
        skeleton_position = skeleton_position * np.array([self.size / width, self.size / height])
        return skeleton_position
    def __call__(self,images,image_points,image_resamples = None):
        self.shape = images[0].size
        out = []
        if image_resamples == None:
            image_resamples = [Image.BILINEAR for k in range(len(images))]

        for img,resample in zip(images,image_resamples):
            img = to_pil(img)
            temp = torchvision.transforms.functional.resize(img, self.size, resample)
            temp = torchvision.transforms.functional.center_crop(temp,(self.size,self.size))
            
            out.append(temp)

        out_points = []
        for points in image_points:
            temp = self._skeleton_transform(points)
            out_points.append(temp)
        
        return out, out_points
def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def default_crop_size(w,h): return [w,w] if w < h else [h,h]
def uniform(a,b): return a + (b-a) * random.random()


def warp(img, size, src_coords, resample=Image.BILINEAR):
    w,h = size
    targ_coords = ((0,0),(0,h),(w,h),(w,0))
    c = find_coeffs(src_coords,targ_coords)
    reversed_c = find_coeffs(targ_coords,src_coords)
    res = img.transform(size, Image.PERSPECTIVE, list(c), resample=resample)
    return res, reversed_c

def find_coeffs(orig_pts, targ_pts):
    matrix = []
    #The equations we'll need to solve.
    for p1, p2 in zip(targ_pts, orig_pts):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = FloatTensor(matrix)
    B = FloatTensor(orig_pts).view(8, 1)
    #The 8 scalars we seek are solution of AX = B
    return list(torch.solve(B,A)[0][:,0])

def process_sz(sz):
    sz = listify(sz)
    return tuple(sz if len(sz)==2 else [sz[0],sz[0]])

class PilTiltRandomCrop():
    """
    this is a modified version of https://github.com/fastai/course-v3 in nbs dl2 augmentation something
    this was License under Apache License 2.0
    """
    def __init__(self, size, scale = (0.3, 1.0), magnitude=0.1): 
        self.size,self.magnitude = process_sz(size),magnitude
        self.scale = scale 
        self.randomize = True
    
    def randomize_parameters(self):
        self.randomize = True

    def get_cropsize(self,size):
        return (uniform(self.scale[0],min(self.scale[1],0.999)) * torch.tensor([size,size])).long()

    def __call__(self, x, resample=Image.BILINEAR):
        if self.randomize:
            self.randomize = False
            csize = self.get_cropsize(default_crop_size(*x.size)[0])
            left,top = random.randint(0,x.size[0]-csize[0]),random.randint(0,x.size[1]-csize[1])
            top_magn = min(self.magnitude, left/csize[0], (x.size[0]-left)/csize[0]-1)
            lr_magn  = min(self.magnitude, top /csize[1], (x.size[1]-top) /csize[1]-1)
            up_t,lr_t = uniform(-top_magn, top_magn),uniform(-lr_magn, lr_magn)
            src_corners = torch.tensor([[-up_t, -lr_t], [up_t, 1+lr_t], [1-up_t, 1-lr_t], [1+up_t, lr_t]])
            src_corners = src_corners * csize.detach().float() + torch.tensor([left,top]).float()
            self.src_corners = tuple([(int(o[0].item()), int(o[1].item())) for o in src_corners])
            
        return warp(x, self.size, self.src_corners, resample = resample)


def project_points(skeleton_pos,c):
    def project_point(point,c):
        a,b,c,d,e,f,g,h = c
        x = point[0]
        y = point[1]        
        return torch.tensor([(a * x + b * y + c)/(g*x + h*y + 1),(d*x + e*y + f)/(g*x + h*y + 1)])
    out = []
    for point in skeleton_pos:
        temp = project_point(point,c)
        temp = [temp[0].item(),temp[1].item()]
        out.append(temp)
    return out



class sample_PilTiltRandomCrop():
    def __init__(self, size, scale = (0.3,1), magnitude=0.):
        self.image_transform = PilTiltRandomCrop(size, scale, magnitude)
        self.reversed_c = None
        
    def _skeleton_transform(self,skeleton_pos):
        def project_point(point,c):
            a,b,c,d,e,f,g,h = c
            x = point[0]
            y = point[1]        
            return torch.tensor([(a * x + b * y + c)/(g*x + h*y + 1),(d*x + e*y + f)/(g*x + h*y + 1)])
        c = self.reversed_c
        out = []
        for point in skeleton_pos:
            temp = project_point(point,c)
            temp = [temp[0].item(),temp[1].item()]
            out.append(temp)
        return torch.tensor(out)

    def __call__(self,images,image_points,image_resamples = None):
        self.image_transform.randomize_parameters()
        out = []
        if image_resamples == None:
            image_resamples = [Image.BILINEAR for k in range(len(images))]
        for img,resample in zip(images,image_resamples):
            img = to_pil(img)

            temp, self.reversed_c = self.image_transform(img,resample)
            out.append(temp)

        out_points = []
        for points in image_points:
            temp = self._skeleton_transform(points)
            out_points.append(temp)
        
        return out, out_points

class sample_compose():
    def __init__(self,transform_list):
        self.transform_list = transform_list
    def __call__(self,image,image_points,image_resamples = None):

        for transform in self.transform_list:
            image,image_points = transform(image,image_points,image_resamples)
        return image,image_points


def debug_show_image(skeleton_points, image):
    def plot_points(points):
        x = [k[0] for k in points]
        y = [k[1] for k in points]
        #print(x[0],y[0])
        plt.plot(x,y, 'ro')

    plot_points(skeleton_points)
    if not isinstance(image,torch.Tensor):
        plt.imshow(image)
    else:
        plt.imshow(image.permute(1,2,0))
    plt.show()

# %%
