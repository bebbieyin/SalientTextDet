import cv2
from torch import nn
import numpy as np
import torch

class GaborFilter(nn.Module):
    def __init__(self, img_size=224):
        super(GaborFilter, self).__init__() 
        
        self.img_size = img_size
        self.scale_range = [3,6,12]
        self.sigma = 3  
        self.orientation_range =[0, 90]
        self.frequency = 0.3 
        self.phase = 0
    
    def forward(self,img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('uint8')
        gaborFeatures = [] 
        for ksize in self.scale_range:
            for theta in self.orientation_range:
                gabor_kernel = cv2.getGaborKernel((ksize, ksize), self.sigma, theta, self.frequency, self.phase, ktype=cv2.CV_32F)

                # Normalize the kernel
                gabor_kernel /= np.sqrt((gabor_kernel ** 2).sum())

                # Apply the Gabor filter to the image
                filtered_image = cv2.filter2D(gray_image, cv2.CV_8U, gabor_kernel)
                gaborFeatures.append(filtered_image)
        
        gabor_reshaped = []
        for g in gaborFeatures:
            a = g.reshape(self.img_size, self.img_size, 1)
            gabor_reshaped.append(a)
        feat_gabor = np.concatenate(gabor_reshaped, axis=2)

        return feat_gabor
    
class SuperpixelSegment(nn.Module):

    def __init__(self):
        super(SuperpixelSegment, self).__init__() 
    
    def forward(self, img, img_size):
        from skimage.segmentation import felzenszwalb
        segments_fz = felzenszwalb(img, scale=1000, sigma=1, min_size=50)

        return segments_fz.reshape((img_size, img_size, 1))
    
class HSVImage(nn.Module):

    def __init__(self):
        super(HSVImage, self).__init__() 
    
    def forward(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

class LABImage(nn.Module):

    def __init__(self):
        super(LABImage, self).__init__() 
    
    def forward(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


class LowLevelFeat(nn.Module):
    def __init__(self,img_size=896):
        super(LowLevelFeat, self).__init__() 
        
        self.img_size = img_size
        self.edge_maps = GaborFilter(img_size=self.img_size)
        self.superpixel = SuperpixelSegment()
        self.hsv = HSVImage()
        self.lab = LABImage()
                       
    def forward(self,x):

        output_imgs = []
        batch = x.cpu().permute(0,2,3,1).numpy()
        for img in batch:

            segments = self.superpixel(img, self.img_size)
            gaborFeat = self.edge_maps(img)
            imghsv = self.hsv(img)
            imglab = self.lab(img)
                                   
            low_feat = np.concatenate((segments, imghsv, imglab,gaborFeat), axis=2)
            low_feat_tensor = torch.from_numpy(low_feat).permute(2, 0, 1).unsqueeze(0).float()
            output_imgs.append(low_feat_tensor)

        out = torch.Tensor(len(batch), self.img_size , self.img_size)
        torch.cat(output_imgs, out=out)

        return out
