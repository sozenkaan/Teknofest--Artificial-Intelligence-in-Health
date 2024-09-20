import cv2
import glob
import SimpleITK as sitk
import numpy as np
import os
from matplotlib import pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 8,8 # Bundan sonraki plot büyüklükleri için varsayılan değer
mask_path = glob.glob('C:\\Users\\bilge\\PycharmProjects\\pythonProject\\TRAINING\\KANAMA\\MASSK\\'+'*.png',recursive=True)
folder_paths= os.path.abspath("folder_path")
print("RESİM SAYISI :",len(mask_path))
for i in mask_path:
    reader = sitk.ImageFileReader()
    reader.SetFileName(i)
    mask = reader.Execute()
    npmask = sitk.GetArrayFromImage(mask)
    print(npmask.shape)
    print(npmask.dtype)
    #plt.imshow(npmask)
    #plt.show()
    print(type(npmask))
    plt.imsave("{}".format(i),npmask,cmap=plt.cm.gray)
