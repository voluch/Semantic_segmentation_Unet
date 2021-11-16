# This file is used to train segmentator that would predict mask of the ship on image.
# For segmentation masks are predictible vvars and outputs for Unet. Storing masks in any variable for many
# images costs a lot of memory. That is why, I decided to save masks for images in other directory and then
# at training segmentator upload it in memory. Saving masks was done by one of Preparation's method (see preparations.py)
# To train segmentator was randomly choosen 8000 images with ships and saved masks to it.



import os
import time
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import tensorflow.keras.layers as tfl
import random
import matplotlib.pyplot as plt

# Image size define wich size of input image will be. It is less then original size (768, 768),
# cause big size of input images eats a lot of memory
img_size = (256, 256)
num_classes = 2
batch_size = 16

# There paths are created that would be used in generators
input_dir = r'/content/sample_data/with_ship/image/'
target_dir = r'/content/sample_data/with_ship/mask/'



input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(target_img_paths))

# Print to see are paths correctly created
for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)







# This is a data generator that will be used in training
class AirBusShip(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img = 1 - np.rint(np.asarray(img))
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y


# By the task this metric should be used in training.
def dice_score(mask_val, mask_pred):
    """
    mask_val: validation mask
    mask_pred: predicted mask
    Returns Dice score
    """
    # Flatten is not necessary, but could be used to find intersection by using np.intersection1d
    np_mask_pred = np.around(mask_pred.numpy().flatten())
    np_mask_val = np.around(mask_val.numpy().flatten())

    # Predicted numpy array has not only 0 and 1, it's values are between 0 and 1. But to calculate Dice score
    # mask_pred and mask_val have to have the same structure - values 0 and 1, not between.
    # Threshold 0.9 was choosen, but it needs to be improved
    np_mask_pred[np_mask_pred > 0.9] = 1
    np_mask_pred[np_mask_pred <= 0.9] = 0
    np_mask_val[np_mask_val > 0.9] = 1
    np_mask_val[np_mask_val <= 0.9] = 0

    volume_sum = np_mask_pred.sum() + np_mask_val.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = np.sum((np_mask_pred == np_mask_val) & (np_mask_pred == 1))
    return 2 * volume_intersect / volume_sum


### https://medium.com/projectpro/u-net-convolutional-networks-for-biomedical-image-segmentation-435699255d26
# Define Unet model as in example that you can find by link over. Number of filters in model layers differs from example's one.
# I decreased it to save time for training model, but, may be, bigger numbers will give better perfomance of model.
def unet(inputsize=(256, 256, 3)):
    inputs = tf.keras.Input(shape=(inputsize))
    # Down - sampling
    conv1 = tfl.Conv2D(8, 3, activation='relu', padding="same", kernel_initializer='he_normal')(inputs)
    conv1 = tfl.Conv2D(8, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv1)
    pool1 = tfl.MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = tfl.Conv2D(16, 3, activation='relu', padding="same", kernel_initializer='he_normal')(pool1)
    conv2 = tfl.Conv2D(16, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv2)
    pool2 = tfl.MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = tfl.Conv2D(32, 3, activation='relu', padding="same", kernel_initializer='he_normal')(pool2)
    conv3 = tfl.Conv2D(32, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv3)
    pool3 = tfl.MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = tfl.Conv2D(64, 3, activation='relu', padding="same", kernel_initializer='he_normal')(pool3)
    conv4 = tfl.Conv2D(64, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv4)
    drop4 = tfl.Dropout(0.5)(conv4)
    pool4 = tfl.MaxPool2D(pool_size=(2, 2))(drop4)

    conv5 = tfl.Conv2D(128, 3, activation='relu', padding="same", kernel_initializer='he_normal')(pool4)
    conv5 = tfl.Conv2D(128, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv5)
    drop5 = tfl.Dropout(0.5)(conv5)

    # Up - sampling
    up6 = tfl.Conv2D(64, 2, activation='relu', padding="same", kernel_initializer='he_normal')(
        tfl.UpSampling2D(size=(2, 2))(drop5))

    merge6 = tfl.concatenate([drop4, up6], axis=3)
    conv6 = tfl.Conv2D(64, 3, activation='relu', padding="same", kernel_initializer='he_normal')(merge6)
    conv6 = tfl.Conv2D(64, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv6)

    up7 = tfl.Conv2D(32, 2, activation='relu', padding="same", kernel_initializer='he_normal')(
        tfl.UpSampling2D(size=(2, 2))(conv6))

    merge7 = tfl.concatenate([conv3, up7], axis=3)
    conv7 = tfl.Conv2D(32, 3, activation='relu', padding="same", kernel_initializer='he_normal')(merge7)
    conv7 = tfl.Conv2D(32, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv7)

    up8 = tfl.Conv2D(16, 2, activation='relu', padding="same", kernel_initializer='he_normal')(
        tfl.UpSampling2D(size=(2, 2))(conv7))

    merge8 = tfl.concatenate([conv2, up8], axis=3)
    conv8 = tfl.Conv2D(16, 3, activation='relu', padding="same", kernel_initializer='he_normal')(merge8)
    conv8 = tfl.Conv2D(16, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv8)

    up9 = tfl.Conv2D(8, 2, activation='relu', padding="same", kernel_initializer='he_normal')(
        tfl.UpSampling2D(size=(2, 2))(conv8))

    merge9 = tfl.concatenate([conv1, up9], axis=3)
    conv9 = tfl.Conv2D(8, 3, activation='relu', padding="same", kernel_initializer='he_normal')(merge9)
    conv9 = tfl.Conv2D(8, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv9)
    conv9 = tfl.Conv2D(2, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv9)

    conv9 = tfl.Conv2D(1, 1, activation='sigmoid')(conv9)
    outputs = conv9
    final_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return final_model


# Build model and print it's summary
model = unet()
print(model.summary())


# Split img paths into a training and a validation sets
val_samples = 1000
random.Random(1).shuffle(input_img_paths)
random.Random(1).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]


# Instantiate data generators for each split
train_gen = AirBusShip(
    batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen = AirBusShip(batch_size, img_size, val_input_img_paths, val_target_img_paths)


# Compile and fit model, track time that is needed to fit model
tic = time.perf_counter()
model.compile(optimizer=RMSprop(learning_rate=0.001), loss="binary_crossentropy", run_eagerly=True, metrics=[dice_score])
epochs = 10
history = model.fit(train_gen, batch_size=1, epochs=epochs, validation_data=val_gen,validation_steps=1)
print(f"Time spent for model fitting {(time.perf_counter() - tic)/60}")




# Retrieve a list of list results on training and test data
# sets for each training epoch
acc=history.history['dice_score']
val_acc=history.history['val_dice_score']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))


# Plot training and validation accuracy per epoch
plt.figure()
plt.plot(epochs, acc, 'r', "Training dice_score")
plt.plot(epochs, val_acc, 'b', "Validation dice_score")
plt.title('Training and validation dice_score')
plt.show()


# Plot training and validation accuracy per epoch
plt.figure()
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation Loss')
plt.show()


# Save fitted model to use it without training each time, when it is needed to predict mask of image
model.save('/content/drive/MyDrive/segmentatator')