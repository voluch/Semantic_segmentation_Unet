# This file is used to train classificator that would predict is there ship on image or not. In segmentation problem classificator could be used to choose
# wich image should be passed to segmentation. But cause of classificator's bad improvement segmentation could produce bad results too.
# So, use classificator has sense in case it has good perfomance.
# Before training, special folder system was created and images were stored in it:
# -- train/labeled_images
#     | -- 1                (images with ships)
#     | -- 0                (images without ships)
# -- validation/labeled_images
#     | -- 1                (images with ships)
#     | -- 0                (images without ships)
# Code of this preparations you can find in preparations.py
#
# Note: Classificator was fitted only on 3000 images, so it has bad score. Only 3000 images were choosen to save fitting time.




# Original image size
IMG_SIZE = (768, 768)


import tensorflow as tf
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# To see how much time do we need to fit model would be used var tic
tic = time.perf_counter()

# Data augmentation, to give more various data on input to training. Commented rows can be used to add more augmantation options.
# It is commented cause of time saving and slow model fitting.
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   # rotation_range = 40,
                                   # width_shift_range = 0.2,
                                   # height_shift_range = 0.2,
                                   # shear_range = 0.2,
                                   # zoom_range = 0.2,
                                   horizontal_flip = True)

# Validation data should not be augmented too
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 8 using train_datagen generator
train_generator = train_datagen.flow_from_directory(r'./input/images/train/labeled_images/',
                                                    batch_size = 8,
                                                    class_mode = 'binary',
                                                    target_size = IMG_SIZE)

# Flow validation images in batches of 8 using train_datagen generator
validation_generator = test_datagen.flow_from_directory( r'./input/images/validation/labeled_images/',
                                                          batch_size  = 8,
                                                          class_mode  = 'binary',
                                                          target_size = IMG_SIZE)

# Define a Callback class that stops training once metric score reaches 90%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('f1_metric') > 0.9):
            print("\nReached 90% f1_metric so cancelling training!")
            self.model.stop_training = True



# Define model. It should be changed for better classification
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(768, 768, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# As classes are imbalanced (There are much more images without ships (see EDA), accuracy metric would be bad to evaluate model.
# That is why f1_metric was used.
import keras.backend as K
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# Model compiling
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=[f1_metric])

# To see information about model print we can print summary
print(model.summary())

# Defining Callback variable
callbacks = myCallback()

# To see model loss and scores per epoch it should be created var history.
history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 3,
            validation_steps = 50,
            verbose = 2,
            callbacks=callbacks)

print(f"Time spent for model fitting {(time.perf_counter() - tic)/60}")



# Retrieve a list of list results on training and test data
# sets for each training epoch
acc=history.history['f1_metric']
val_acc=history.history['val_f1_metric']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs


# Plot training and validation metric score per epoch
plt.figure()
plt.plot(epochs, acc, 'r', "Training f1_metric")
plt.plot(epochs, val_acc, 'b', "Validation f1_metric")
plt.title('Training and validation f1_metric')
plt.show()


# Plot training and validation loss per epoch
plt.figure()
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation Loss')
plt.show()



# Save fitted model to use it without training each time, when it is needed to predict class of image
model.save('classificator')
