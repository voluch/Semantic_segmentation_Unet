from tensorflow import keras
import os
import keras.backend as K
import numpy as np
from keras.preprocessing import image
import pandas as pd
from preparations import Preparation

# Classificator has custom metric, so to load classificator and use it, it is necessary to define custom metric there too.
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val



# Load model
model = keras.models.load_model("classificator", custom_objects={'f1_metric': f1_metric})


# This function allows to predict is ship or not on images from folder
def predict(path):
    lis = os.listdir(path)
    lis = [x for x in lis if '.jpg' in x]
    y_pred = []
    for img in lis:

        # predicting images
        im = image.load_img(path + img, target_size=(768, 768))
        x = image.img_to_array(im)
        x = np.expand_dims(x, axis=0)

        classes = model.predict(x, batch_size=10)
        if classes[0] > 0:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return lis, y_pred

imgs, y_pred = predict(r'./input/images/predict/')

# This is needed to get tru class to compare with predicted values
prep = Preparation()
y_true = prep.unique_img_ids.loc[prep.unique_img_ids.ImageId.isin(imgs), 'is_ship'].tolist()

# Dataframe with predicted and true classes
print(pd.DataFrame({'imgs': imgs, 'y_true': y_true, 'y_pred': y_pred}))
