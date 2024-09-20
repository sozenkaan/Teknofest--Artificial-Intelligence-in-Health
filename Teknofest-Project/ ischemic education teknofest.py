
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import cv2
import glob
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

image_width = 512
image_height = 512

train_files = glob.glob('C:\\Users\\bilge\\PycharmProjects\\pythonProject\\iskemii\\'+'*.png',recursive=True)
mask_filess =glob.glob('C:\\Users\\bilge\\PycharmProjects\\pythonProject\\TRAINING\\ISKEMI\\MAskk\\'+'*.png',recursive=True)

#for i in mask_filess:
 #   train_files.append(i)
print(train_files[:4])
print(mask_filess[:4])


rows, cols = 3,3
fig = plt.figure(figsize=(10,10))
for i in range(1, rows*cols+1):
    fig.add_subplot(rows, cols, i)
    img_path = train_files[i]
    msk_path = mask_filess[i]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    plt.imshow (img)
    msk = cv2.imread(msk_path)
    #plt.imshow(msk)
plt.show()



df = pd.DataFrame(data={'filename': train_files, 'mask': mask_filess})
df_train, df_test = train_test_split(df, test_size = 0.1)
df_train, df_val = train_test_split(df_train, test_size = 0.2)
print(df_train.values.shape)
print(df_val.values.shape)
print(df_test.values.shape)
print(len(df_train))
print(len(df_val))
def train_generator(data_frame, BATCH_SIZE, aug_dict,
        image_color_mode="rgb",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=(512,512),
        seed=42):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col = "filename",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = BATCH_SIZE,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        interpolation='bilinear',
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = BATCH_SIZE,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        interpolation='bilinear',
        seed = seed)

    train_gen = zip(image_generator, mask_generator)

    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img, mask)

def adjust_data(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] =1
    mask[mask <= 0.5] =0
    return (img, mask)


def conv_block(input_tensor, filter_size, kernel_size=3, batchnorm=True):
    # First layers of Convolution
    X=Conv2D(filters=filter_size, kernel_size=(kernel_size, kernel_size),
                kernel_initializer="he_normal", padding="same") (input_tensor)

    if batchnorm:
        X= BatchNormalization()(X)

    X=Activation("relu")(X)

    # Second layers of Convolution
    X = Conv2D (filters=filter_size, kernel_size=(kernel_size, kernel_size),
                kernel_initializer="he_normal", padding="same") (X)

    if batchnorm:
        X = BatchNormalization () (X)

    X = Activation ("relu") (X)

    return X


def UNet(input_image, filter_size=32, dropout=0.1, batchnorm=True):
    # Encoder/Contracting Path of the Unet
    c1 = conv_block (input_image, filter_size * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D ((2, 2)) (c1)
    p1 = Dropout (dropout) (p1)

    c2 = conv_block (p1, filter_size * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D ((2, 2)) (c2)
    p2 = Dropout (dropout) (p2)

    c3 = conv_block (p2, filter_size * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D ((2, 2)) (c3)
    p3 = Dropout (dropout) (p3)

    c4 = conv_block (p3, filter_size * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D ((2, 2)) (c4)
    p4 = Dropout (dropout) (p4)

    c5 = conv_block (p4, filter_size * 16, kernel_size=3, batchnorm=batchnorm)

    # Decoder/Expanding Path of the Unet

    u6 = Conv2DTranspose (filter_size * 8, (2, 2), strides=(2, 2), padding="same") (c5)
    u6 = concatenate ([u6, c4])
    u6 = Dropout (dropout) (u6)
    c6 = conv_block (u6, filter_size * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose (filter_size * 4, (2, 2), strides=(2, 2), padding="same") (c6)
    u7 = concatenate ([u7, c3])
    u7 = Dropout (dropout) (u7)
    c7 = conv_block (u7, filter_size * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose (filter_size * 2, (2, 2), strides=(2, 2), padding="same") (c7)
    u8 = concatenate ([u8, c2])
    u8 = Dropout (dropout) (u8)
    c8 = conv_block (u8, filter_size * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose (filter_size * 1, (2, 2), strides=(2, 2), padding="same") (c8)
    u9 = concatenate ([u9, c1])
    u9 = Dropout (dropout) (u9)
    c9 = conv_block (u9, filter_size * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D (1, (1, 1), activation="sigmoid") (c9)
    model = Model (inputs=[input_image], outputs=outputs)
    return model

input_img = Input((512, 512, 3), name='img')
model = UNet(input_img, filter_size=32, dropout=0.1, batchnorm=True)
model.summary()




def dice_coef(y_pred, Y, smooth = 100):
    y_flatten = K.flatten(Y)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_flatten * y_pred_flatten)
    union = K.sum(y_flatten) + K.sum(y_pred_flatten)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def jacard_coef(y_pred, Y,smooth = 100):
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

EPOCHS = 200
BATCH_SIZE = 8
learning_rate = 1e-4

train_generator_args = dict(fill_mode='nearest')

train_gen = train_generator(df_train,BATCH_SIZE,
                           train_generator_args,
                           target_size = (512,512))

validation_gen = train_generator(df_val,BATCH_SIZE,
                           dict(),
                           target_size = (512,512))

optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
model.compile(optimizer = "adam", loss = dice_coef_loss, metrics = ["binary_accuracy", jacard_coef, dice_coef])
# model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])

earlystopping = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=295)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="iskemikyeniunetv8.hdf5",verbose=1,save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',mode='min',verbose=1,patience=295,min_delta=0.0001,factor=0.1)

tf.keras.backend.clear_session()
history = model.fit(train_gen, steps_per_epoch=len(df_train) / BATCH_SIZE, epochs=EPOCHS, validation_data = validation_gen,
                    validation_steps=len(df_val) / BATCH_SIZE,
                    callbacks = checkpointer)


a = history.history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['jacard_coef'])
plt.plot(history.history['val_jacard_coef'])
plt.title("SEG UNet Model jacard_coef")
plt.ylabel("jacard_coef")
plt.xlabel("Epochs")
plt.legend(['train', 'val'])

plt.subplot(1,2,2)
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title("SEG UNet Model dice_coef")
plt.ylabel("dice_coef")
plt.xlabel("Epochs")
plt.legend(['train', 'val'])

model = load_model('iskemikyeniunetv8.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'jacard_coef': jacard_coef, 'dice_coef': dice_coef})

for i in range(30):
    index=np.random.randint(1,len(df_test.index))
    img = cv2.imread(df_test['filename'].iloc[index])
    img = cv2.resize(img ,(image_height, image_width))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    pred=model.predict(img)

    plt.figure(figsize=(12,12))
    plt.subplot(1,3,1)
    plt.imshow(np.squeeze(img))
    plt.title('Original Image')
    plt.subplot(1,3,2)
    plt.imshow(np.squeeze(cv2.imread(df_test['mask'].iloc[index])))
    plt.title('Original Mask')
    plt.subplot(1,3,3)
    plt.imshow(np.squeeze(pred).round(),cmap='binary')
    plt.title('Prediction')
    plt.show()
