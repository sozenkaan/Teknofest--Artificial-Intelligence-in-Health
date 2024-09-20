from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Dropout, Dense, Activation, BatchNormalization, Flatten,concatenate,Conv2DTranspose
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os


print("[INFO] loading images...")


DIRECTORY = r"C:\Users\bilge\PycharmProjects\pythonProject\sınıflandırma_teknofest"
CATEGORIES = ["INMEYOK", "INME"]


data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(512, 512))
        image = img_to_array(image)
        data.append(image)
        labels.append(category)


# perform one-hot encoding on the labels

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

print("X_train shape", trainX.shape)
print("y_train shape", trainY.shape)
print("X_test shape", testX.shape)
print("y_test shape", testY.shape)

trainX = trainX.reshape(trainX.shape[0], 512 , 512, 3)
testX = testX.reshape(testX.shape[0], 512, 512, 3)
input_shape = (512, 512, 3)

trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255
testX /= 255

print('x_train shape:', trainX.shape)
print(trainX.shape[0], 'train samples')
print(testX.shape[0], 'test samples')

aug = ImageDataGenerator(
    rotation_range=20,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode="nearest")

def ortalama(conf_matrix):
    sensitivity1 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    print('Sensitivity : ', sensitivity1)
    specificity1 = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    print('Specificity : ', specificity1)
    return print("Ortalama:", ((sensitivity1+specificity1)/2.0))

INIT_LR = 1e-4
EPOCHS = 20
BS = 8

def cnn(input_size=(512,512,3)):
    inputs = Input(input_size)

    conv1 = Conv2D(16, (3, 3), padding='same')(inputs)
    bn1 = Activation('relu')(conv1)
    conv1 = Conv2D(16, (3, 3), padding='same')(bn1)
    bn1 = BatchNormalization(axis=3)(conv1)
    bn1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(32, (3, 3), padding='same')(pool1)
    bn2 = Activation('relu')(conv2)
    conv2 = Conv2D(32, (3, 3), padding='same')(bn2)
    bn2 = BatchNormalization(axis=3)(conv2)
    bn2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(64, (3, 3), padding='same')(pool2)
    bn3 = Activation('relu')(conv3)
    conv3 = Conv2D(64, (3, 3), padding='same')(bn3)
    bn3 = BatchNormalization(axis=3)(conv3)
    bn3 = Activation('relu')(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(128, (3, 3), padding='same')(pool3)
    bn4 = Activation('relu')(conv4)
    conv4 = Conv2D(128, (3, 3), padding='same')(bn4)
    bn4 = BatchNormalization(axis=3)(conv4)
    bn4 = Activation('relu')(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(256, (3, 3), padding='same')(pool4)
    bn5 = Activation('relu')(conv5)
    conv5 = Conv2D(256, (3, 3), padding='same')(bn5)
    bn5 = BatchNormalization(axis=3)(conv5)
    bn5 = Activation('relu')(bn5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(bn5)

    conv6 = Conv2D(512, (3, 3), padding='same')(pool5)
    bn6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, (3, 3), padding='same')(bn6)
    bn6 = BatchNormalization(axis=3)(conv6)
    bn6 = Activation('relu')(bn6)

    fl = Flatten()(bn6)
    dn = Dense(1024, activation="relu")(fl)
    bn6 = Activation('relu')(dn)
    bn6 = BatchNormalization()(bn6)
    dr = Dropout(0.5)(bn6)
    de = Dense(1, activation="sigmoid")(dr)

    return Model(inputs=[inputs], outputs=[de])


model = cnn()
model.summary()

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["binary_accuracy"])


H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
    )

#print("The model has successfully trained")
model.save('inme-class-teknofest_v2.h5')


#model = load_model('inme-class-teknofest_v1')

print("[INFO] evaluating network...")
score = model.evaluate(testX, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability

predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
# show a nicely formatted classification report
#print("Classification Report: \n", classification_report(testY,y_pred))

#printing the confusion matrix
LABELS = ['INMEYOK', 'INME']
conf_matrix = confusion_matrix(testY.argmax(axis=1), predIdxs)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix, xticklabels = LABELS,
            yticklabels = LABELS, annot = True, fmt ="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# plot the training and validation accuracy and loss at each epoch
loss = H.history['loss']
val_loss = H.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#acc = history.history['jacard_coef']
acc = H.history['accuracy']
#val_acc = history.history['val_jacard_coef']
val_acc = H.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

ortalama(conf_matrix)