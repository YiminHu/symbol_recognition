import cv2
import sys
import os
img_name = sys.argv[1]
os.mkdir(img_name+"result")
test= cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
test = cv2.copyMakeBorder(test,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
width,height=test.shape
if width%2==0:
    block=width+1
else:
    block=width
#ret,test_ex_bin= cv2.threshold(test,127,255,0)
#cv2.imwrite('test_bin.jpg',test_ex_bin)
test_ex_bin  = cv2.adaptiveThreshold(test,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block,0)
#test_ex_bin = cv2.resize(test_ex_bin,(2*327,2*67)
#test_ex_bin = cv2.erode(test_ex_bin,kernel)

image, contours, hierarchy = cv2.findContours(test_ex_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for i in range(0,len(contours)):
    if hierarchy[0][i][3]==0:
        x, y, w, h = cv2.boundingRect(contours[i])
        if w*h > 30:
            cv2.rectangle(image, (x,y), (x+w,y+h), (153,153,0), 1)
            newimage=test[y:y+h,x:x+w]
            cv2.imwrite(img_name+"result/"+str(i)+".jpg",newimage)
            print (w*h)

    #can add root judgement here
    
cv2.imwrite(img_name+"result/write.jpg",image)

