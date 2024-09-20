
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
import numpy as np
import cv2 as cv

BS = 8
data = []

new_model = tf.keras.models.load_model('sınıflandırma-7.h5')  #Test Edilecek model yüklendi.
#im= cv.imread('database/normal/1.jpg')               # Test edilecek  resim çekildi
#imi = cv.imread('database/covid/1.jpg')              # Test edilecek  resim çekildi

#image = cv.cvtColor(im, cv.COLOR_BGR2RGB)            # Normal
#image = cv.cvtColor(imi, cv.COLOR_BGR2RGB)          # Covid
#image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


#image = cv.resize(image, (224, 224))
#data.append(image)
#data = np.array(data) / 255.0

predIdxs = new_model.predict(data)
pr_normal = predIdxs[0][1] * 100;
pr_cov    = predIdxs[0][0] * 100;

print("Durum Normal: %.2f" %  pr_normal)
print("Durum Covid: %.2f" %  pr_cov)
