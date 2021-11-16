**# Semantic_segmentation_Unet**

**Task**: train model to Semantic Segmentation on data https://www.kaggle.com/c/airbus-ship-detection/data using Unet architecture of NN.

**Solution**: At first, train classification model wich will predict is there ship on image. At second, train segmentation model Unet to predict mask of ships on image.

**EDA**: Input data are images and .csv file with masks over it (rle encoded). It is good idea to see how many images have ship on it and don't have, how many ships can be on one image, how mask cover image. All of it you can see in file airbus-ship-detection-eda.ipynb. There are 192556 images. It is a big dataset. But training models was done on local machine, so models were training on lower datasets.


**Classification**: For classification dataset was train - 3000 images, validation 1000. In this sets almost 75% of images were withot ships (like in original dataset). Images stored in folder system like:

input/train/class1

input/train/class0

input/validation/class1

input/validation/class0


To improve classification and get more various data ImageDataGenerator was used. Also to stop fitting model at good score early stop Callback was written.
As classes are imbalanced, accuracy metric to evaluate model is bad. That is why f1_score was used instead - custom metric.
To see model you can in **train_classificator.py**, but it shoul be improved to get better scores. As a little data was used to train model and model is raw it predicts not good.
But the main goal was to create model, fit it, save it, and use it to predict is ther ship on image or not (as in real project). 
If you want to see how model works and predict, please, use file **predict_class.py** and images from folder **input/images/predict/**. 
Model is in current repository in folder **classificator/**.

**Segmentation:** For training segmentation model was used 7000 images with different number of ships on it and 1000 for validation. To train segmentaion it was necessary to encode masks from .csv file. Encoding 8000 masks and store it in variable costs a lot of time and memory. So decision was to save masks on local pc and load them to model. 
As a result data was stored in folder system like:

with_ship/image

with_ship/mask

Also, the input size of images was decreased from (768, 768) to (256, 256), cause it saves time and resources for model fitting. To load images and masks was written AirBusShip data generator.
Unet architecture was build by example https://medium.com/projectpro/u-net-convolutional-networks-for-biomedical-image-segmentation-435699255d26. To evaluate model was used metric dice_score.
All of described over you can see in file **train_segmentator.py**.  
Segmentation has not bad perfomance and, i think, it could be much more better if increase data to fit it. You can test segmentator by using file **predict_segmentation.py**, images from folder **input/images/predict/** and model from folder **segmentatator/.**


Note: some additional operations were needed to prepare data defore training. You can see this steps in file **preparations.py**.
Note 2: data used for training aren't commited to current repository. But you can download it from https://www.kaggle.com/c/airbus-ship-detection/data, then store it in system folder like described over.



