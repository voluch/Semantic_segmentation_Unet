import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow import keras
import matplotlib.pyplot as plt



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
    # Threshold 0.9 was choosed, but it needs to be improved
    np_mask_pred[np_mask_pred > 0.9] = 1
    np_mask_pred[np_mask_pred <= 0.9] = 0
    np_mask_val[np_mask_val > 0.9] = 1
    np_mask_val[np_mask_val <= 0.9] = 0

    volume_sum = np_mask_pred.sum() + np_mask_val.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = np.sum((np_mask_pred == np_mask_val) & (np_mask_pred == 1))
    return 2 * volume_intersect / volume_sum

# Function to encode predicted mask
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)



# Load model with custom metric
model = keras.models.load_model("segmentatator", custom_objects={'dice_score': dice_score})


# Input image size was (256, 256), so prediction should be done for image the same size
img_size = (256, 256)


# To see predicted mask please choose image (.jpg) from folder validation_set and copy its name (without .jpg). Change ImageId by copied name
ImageId = '0318bab62'

# Load image, change its dimensions to input image dimensions
x = load_img('input/images/predict/' + ImageId + '.jpg', target_size=img_size)
x = np.asarray(x)
image = np.expand_dims(x, axis=0)

# Load mask, true mask
y = load_img('input/images/predict/' + ImageId + '.png', color_mode="grayscale", target_size=img_size)

# Predict. Threshold 0.9 can be changed
y_pred = model.predict(image, batch_size=10)
y_pred[y_pred > 0.9] = 1
y_pred[y_pred <= 0.9] = 0

# Print RLE of predicted mask
print(f"RLE mask is: {rle_encode(y_pred)}")



# Plot original image, true mask and predicted mask
fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
axarr[0].title.set_text('Original Image')
axarr[0].axis('off')
axarr[1].title.set_text('True Mask')
axarr[1].axis('off')
axarr[2].title.set_text('Predicted mask')
axarr[2].axis('off')
axarr[0].imshow(x)
axarr[1].imshow(x)
axarr[1].imshow(y, alpha=0.4)
axarr[2].imshow(x)
axarr[2].imshow(y_pred.reshape((256,256)), alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)  # to adjust automatically axis to subplot area
plt.show()