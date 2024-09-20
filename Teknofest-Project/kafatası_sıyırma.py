import os
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import cv2
import glob
import numpy as np
image_width = 512
image_height = 512


IMAGE_SIZE=(512,512)
img_dir="dataset\INMEYOK"
output_dir="dataset\INMEYOK"
data_path = os.path.join(img_dir,'*png')
files = glob.glob(data_path)
def ShowImage(title,img,ctype):
  plt.figure(figsize=(10, 10))
  if ctype=='bgr':
    b,g,r = cv2.split(img)
    rgb_img = cv2.merge([r,g,b])
    plt.imshow(rgb_img)
  elif ctype=='hsv':
    rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
  elif ctype=='gray':
    plt.imshow(img,cmap='gray')
  elif ctype=='rgb':
    plt.imshow(img)
  else:
    raise Exception("Unknown colour type")
  plt.axis('off')
  plt.title(title)
  plt.show()

def crop_img():
    for f1 in files:
        image = cv2.imread(f1)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret,thres = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
        #ShowImage('Applying Otsu', thres, 'gray')
        colormask = np.zeros(image.shape,dtype=np.uint8)
        colormask[thres != 0] = np.array((0,0,255))
        blended=cv2.addWeighted(image,0.7,colormask,0,1,0)
        #ShowImage('Blended', blended, 'bgr')
        ret,markers = cv2.connectedComponents(thres)
        marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0]
        if marker_area:
            largest_component = np.argmax(marker_area)+ 1
            brain_mask=markers==largest_component
            brain_out = image.copy()
            brain_out[brain_mask==False]= (0,0,0)
            brain_mask = np.uint8(brain_mask)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel,iterations=1)
            brain_out = image.copy()
            brain_out[opening == False] = (0, 0, 0)
            #ShowImage('Connected Components',brain_out,'rgb')
            crop_son = brain_out
            src_fname, ext = os.path.splitext(f1)
            save_fname = os.path.join(output_dir, os.path.basename(src_fname) + '.png')
            plt.imsave(save_fname,crop_son, cmap='gray')
crop_img()