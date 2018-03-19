import numpy as np
import cv2
import random
import os
if os.path.exists('data') == False:
    os.mkdir('data')
OUTPUT_DIR = './data/'
backg = [127,169,211,240]
salt = [0.02,0.04,0.06]
gauss = [0.5,1.0,1.5]
def translate(image, x, y):
    M = np.float32([[1,0,x],[0,1,y]])
    shifted = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return shifted

def rotate(image, angle, center = None, scale=1.0):
    (h,w) = image.shape[:2]
    if center is None:
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        return rotated

def gaussianNoise(img,mean,val):
    (h,w) = img.shape[:2]
    for i in range(h):
        for j in range(w):
            img[i][j]=img[i][j]+random.gauss(mean,val)
    return img

def saltpepperNoise(img,n):
    m=int((img.shape[0]*img.shape[1])*n)
    for a in range(m):
        i=int(np.random.random()*img.shape[1])
        j=int(np.random.random()*img.shape[0])
        if img.ndim==2:
            img[j,i]=255
        elif img.ndim==3:
            img[j,i,0]=255
            img[j,i,1]=255
            img[j,i,2]=255
    for b in range(m):
        i=int(np.random.random()*img.shape[1])
        j=int(np.random.random()*img.shape[0])
        if img.ndim==2:
            img[j,i]=0
        elif img.ndim==3:
            img[j,i,0]=0
            img[j,i,1]=0
            img[j,i,2]=0
    return img

def backgroundNoise(img,val):
    (h,w) = img.shape
    for i in range(h):
        for j in range(w):
            if img[i][j] > 127:
                pixelval =  random.gauss(val,2.5)
                img[i][j] = pixelval
    return img

def gen_from_seed(dirname):
    index = 0
    pathDir = os.listdir(dirname)
    slist = dirname.split('/')
    writedir = slist[len(slist)-1]
    print(writedir)
    for filename in pathDir:
        if not filename.startswith('.'):
            print(filename)
            curimg = cv2.imread(dirname+'/'+filename,0)
            h,w = curimg.shape
            for trans in range(1):
                origin1 = curimg.copy()
                if not trans == 0:
                    x = random.randint(0,int(w*0.1))
                    y = random.randint(0,int(h*0.1))
                    origin1 = translate(origin1,x,y)
                for rot in range(5):
                    origin2 = origin1.copy()
                    if not rot == 0:
                        
                        angle = random.randint(-45,45)
                        origin2 = rotate(origin2,angle)
                    for bg in range(3):
                        origin3 = origin2.copy()
                        origin3 = backgroundNoise(origin3,backg[bg])
                        for noise in range(2):
                            origin4 = origin3.copy()
                            if noise < 3:
                                origin4 = saltpepperNoise(origin4,salt[noise])
                            else:
                                origin4 = gaussianNoise(origin4,0,gauss[noise-3])
                            for blur in range(2):
                                origin5 = origin4.copy()
                                if blur == 1:
                                    origin5 = cv2.GaussianBlur(origin5,(5,5),1.5)
                                else:
                                    origin5 = origin4.copy()
                                if os.path.exists(OUTPUT_DIR+writedir) == False:
                                    os.mkdir(OUTPUT_DIR+writedir)
                                origin5 = cv2.resize(origin5,(28,28))
                                cv2.imwrite(OUTPUT_DIR+writedir+'/'+str(index)+'.jpg',origin5)
                                index = index + 1

                
        
    
