# This file was used to make different preparations for solving main problems like training classification and segmentation

import numpy as np
import pandas as pd
import shutil
import os
import imageio

class Preparation():
    def __init__(self):
        self.masks = pd.read_csv(r"./input/train_ship_segmentations_v2.csv")
        self.unique_img_ids = self.unique_ids_df()

    def unique_ids_df(self):
        """
        This function gets unique ImageId from self.masks DataFrame. Also it defines column is_ship which is indicator of ship on image
        :return: DataFrame
        """
        masks = self.masks.copy()
        masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0).copy()
        unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
        unique_img_ids['is_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
        return unique_img_ids

    def copy_images_to_file_system_for_class(self, path_from, path_to):
        """
        Copy files from folder to folder system for classification task
        :param path_from:
        :param path_to:
        """
        labels = [0,1]
        unique_img_ids = self.unique_img_ids.loc[self.unique_img_ids.ImageId.isin(os.listdir(path_from)), :]
        for label in labels:
            ids = unique_img_ids.loc[unique_img_ids.is_ship == label, 'ImageId'].tolist()
            for id in ids:
                shutil.copy(path_from + id, path_to + str(label) + r'/' + id)

    def rle_decode(self, mask_rle, IMG_SIZE = (768, 768)):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(IMG_SIZE[0]*IMG_SIZE[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(IMG_SIZE).T


    def save_masks(self, path_input_img, path_to_save, IMG_SIZE = (768, 768)):
        """
        :param path_input_img: path to folder There are stored images for which masks should be calculated and saved
        :param path_to_save: path to save masks
        :param IMG_SIZE: size of masks
        """
        lis = os.listdir(path_input_img)
        for img_id in lis:
            img_masks = self.masks.loc[self.masks['ImageId'] == img_id, 'EncodedPixels'].tolist()
            all_masks = np.zeros(IMG_SIZE, dtype=np.uint8)
            if img_masks:
                if isinstance(img_masks[0], str):
                    for m in img_masks:
                        all_masks += self.rle_decode(m)
                del img_masks
                imageio.imsave(path_to_save + img_id.split(".")[0] + ".png", all_masks)
                del all_masks
        print(f"Masks are saved to {path_to_save}!")




prep = Preparation()

# # In folders train and validation were 3000 and 1000 randomly choosen images respectivly
# prep.copy_images_to_file_system_for_class(r'./input/images/train/', r'./input/images/train/labeled_images/')
# prep.copy_images_to_file_system_for_class(r'./input/images/validation/', r'./input/images/validation/labeled_images/')

prep.save_masks(r'./input/images/predict/', r'./input/images/')