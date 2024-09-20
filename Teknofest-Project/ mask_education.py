import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime
import cv2
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, \
    MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pandas as pd
from PIL import Image
from tensorflow.keras import backend, optimizers

image_directory = 'C:\\Users\\bilge\\PycharmProjects\\pythonProject\\TRAINING\\ISKEMI\\egitim\\pnggg\\'
mask_directory = 'C:\\Users\\bilge\\PycharmProjects\\pythonProject\\TRAINING\\ISKEMI\\egitim\\msk\\'

SIZE = 512
image_dataset = []  # Many ways to handle data, you can use pandas. Here, we are using a list format.
mask_dataset = []  # Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):  # Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'png'):
        # print(image_directory+image_name)
        image = cv2.imread(image_directory + image_name, 1)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

# Iterate through all images in Uninfected folder, resize to 64 x 64
# Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(mask_directory + image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

# Normalize images
image_dataset = np.array(image_dataset) / 255.
# D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.10, random_state=0)

# Sanity check, view few mages
import random
import numpy as np

image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (512, 512, 1)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (512, 512)), cmap='gray')
plt.show()

#######################################
# Parameters for model

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
num_labels = 1  # Binary
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
batch_size = 8

# FOCAL LOSS AND DICE METRIC

###############################################################################
def dice_coef(y_pred, Y, smooth=100):
    y_flatten = K.flatten(Y)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_flatten * y_pred_flatten)
    union = K.sum(y_flatten) + K.sum(y_pred_flatten)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def jacard_coef(y_pred, Y, smooth=100):
    y_flatten = K.flatten(Y)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_flatten * y_pred_flatten)
    union = K.sum(y_flatten) + K.sum(y_pred_flatten)
    jacard = (intersection + smooth) / (union - intersection + smooth)
    return jacard


def jacard_coef_loss(y_pred, Y):
    return -jacard_coef(y_pred, Y)


def dice_coef_loss(y_pred, Y):
    return -dice_coef(y_pred, Y)


# Try various models: Unet, Attention_UNet, and Attention_ResUnet
# Rename original python file from 224_225_226_models.py to models.py

def conv_block(input_tensor, filter_size, kernel_size=3, batchnorm=True):
    # First layers of Convolution
    X = Conv2D(filters=filter_size, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal", padding="same")(input_tensor)

    if batchnorm:
        X = BatchNormalization()(X)

    X = Activation("relu")(X)

    # Second layers of Convolution
    X = Conv2D(filters=filter_size, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal", padding="same")(X)

    if batchnorm:
        X = BatchNormalization()(X)

    X = Activation("relu")(X)

    return X

def UNet(input_image, filter_size=16, dropout=0.1, batchnorm=True):
    # Encoder/Contracting Path of the Unet
    c1 = conv_block(input_image, filter_size * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv_block(p1, filter_size * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv_block(p2, filter_size * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv_block(p3, filter_size * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv_block(p4, filter_size * 16, kernel_size=3, batchnorm=batchnorm)

    # Decoder/Expanding Path of the Unet

    u6 = Conv2DTranspose(filter_size * 8, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv_block(u6, filter_size * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(filter_size * 4, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv_block(u7, filter_size * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(filter_size * 2, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv_block(u8, filter_size * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(filter_size * 1, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv_block(u9, filter_size * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)
    model = Model(inputs=[input_image], outputs=outputs)
    return model

input_img = Input((512,512,1), name='img')
model = UNet(input_img, filter_size=16, dropout=0.2, batchnorm=True)
model.summary()

unet_model = UNet(input_shape)
unet_model.compile(optimizer=Adam(lr=1e-2), loss=dice_coef_loss,
                   metrics=['accuracy', jacard_coef,dice_coef])

print(unet_model.summary())

start1 = datetime.now()
unet_history = unet_model.fit(X_train, y_train,
                              verbose=1,
                              batch_size=batch_size,
                              validation_data=[X_test, y_test],
                              shuffle=False,
                              epochs=50)

stop1 = datetime.now()
# Execution time of the model
execution_time_Unet = stop1 - start1
print("UNet execution time is: ", execution_time_Unet)

unet_model.save('mitochondria_UNet_50epochs_B_focal.hdf5')



unet_history_df = pd.DataFrame(unet_history.history)

with open('unet_history_df.csv', mode='w') as f:
    unet_history_df.to_csv(f)


#######################################################################
# Check history plots, one model at a time
history = unet_history


# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#acc = history.history['jacard_coef']
acc = history.history['accuracy']
#val_acc = history.history['val_jacard_coef']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Jacard')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#######################################################


model = unet_model

model_path = "models/mitochondria_UNet_50epochs_B_focal.hdf5"
# Load one model at a time for testing.
model = tf.keras.models.load_model(model_path, compile=False)

import random

test_img_number = random.randint(0, X_test.shape[0] - 1)
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]

test_img_input = np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img, cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:, :, 0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()

# IoU for a single image
from tensorflow.keras.metrics import MeanIoU

n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(ground_truth[:, :, 0], prediction)
print("Mean IoU =", IOU_keras.result().numpy())

# Calculate IoU for all test images and average

import pandas as pd

IoU_values = []
for img in range(0, X_test.shape[0]):
    temp_img = X_test[img]
    ground_truth = y_test[img]
    temp_img_input = np.expand_dims(temp_img, 0)
    prediction = (model.predict(temp_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)

    IoU = MeanIoU(num_classes=n_classes)
    IoU.update_state(ground_truth[:, :, 0], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)

    print(IoU)

df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU)
