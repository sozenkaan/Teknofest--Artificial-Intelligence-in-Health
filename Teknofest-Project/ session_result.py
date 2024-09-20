from os import listdir
from os.path import isfile, join
from pathlib import Path
import numpy
import cv2
import argparse
import csv
import os
import tensorflow as tf
import cv2 as cv
from tensorflow.keras.models import load_model
import numpy as np
# Check whether the CSV
# exists or not if not then create one.
my_file = Path("/names.csv")
data = []
if my_file.is_file ():
    f = open(my_file, "w+")
    with open('names.csv', 'a', newline='') as file:
        writer = csv.writer (file)
        writer.writerow (["id","class"])
    f.close ()
    pass
else:
    with open('names.csv', 'w', newline='') as file:
        writer = csv.writer (file)
        writer.writerow(["id","class"])
# Argparse function to get
# the path of the image directory
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--test", default="test",help="path to input database")
args = vars(ap.parse_args())
# Program to find the
# colors and embed in the CSV
mypath=args["augmented_dataset"]
onlyfiles = [f for f in listdir (mypath) if isfile (join (mypath, f))]
images = numpy.empty(len (onlyfiles), dtype=object)
print(len(onlyfiles))
model=load_model('sınıflandırma-8.h5')                                                     #Teknofest2.h5
for n in range(0, len(onlyfiles)):
    path = join(mypath, onlyfiles[n])
    images[n] = cv.imread(join(mypath, onlyfiles[n]))
    print(images[n])
    imagess = cv.cvtColor(images[n],cv.COLOR_BGR2RGB)
    im_final = imagess.reshape( 1,256, 256, 3)
    ans = model.predict(im_final)
    itml = ans[0]
    if itml[0]<0.5:
        with open('oturum1.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([(onlyfiles[n].split('.',1)[0]),1])
            file.close()
    else:
        with open('oturum1.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([(onlyfiles[n].split('.',1)[0]),0])
            file.close()
