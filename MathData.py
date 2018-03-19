import cv2
import os
import numpy as np
import symclass
def load_data(data_path, image_size, classnum):
    images = []
    labels = []
    classList = os.listdir(data_path)
    for classes in classList:
        if not classes.startswith('.'):
            imageList = os.listdir(data_path+'/'+classes)
            for image in imageList:
                if not image.startswith('.'):
                    img = cv2.imread(data_path+'/'+classes+'/'+image,0)
                    #img = cv2.resize(img,(28,28))
                    img = img.reshape(28*28)
                    tmp = []

                    for i in range(784):
                        tmp.append(1.0 - float(img[i])/255.0)
                    img = np.array(tmp)
                    images.append(img)
                    clsint = symclass.classDict[classes]
                    classvec = np.zeros(classnum)
                    classvec[clsint]=1
                    labels.append(classvec)
    return images, labels
