# %%
import os
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import pathlib
from dataset.my_enum import *

# %%
class Dataset(Dataset):
    def __init__(self, root_path, transform_train, roi_transform_train, transform_test, roi_transform_test, class_number=8, data_type=EnumType.roi,roi_image_data_type=EnumType.align):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.imgs = []
        self.lbls = []
        self.is_roi = []
        self.eye_imgs = []
        self.nose_imgs = []
        self.mouth_imgs = []        
        self.transform_train = transform_train
        self.roi_transform_train = roi_transform_train
        self.transform_test = transform_test
        self.roi_transform_test = roi_transform_test
        self.class_number_sum = []

        if data_type.name == EnumType.roi.name:     
            # AffectNet/test/roi/eye       
            folder_path = os.path.join(root_path, data_type.name, EnumRoi.eye.name)
        elif data_type.name == EnumType.re.name:     
            # AffectNet/test/original
            # roi+error = original
            folder_path = os.path.join(root_path, EnumType.original.name)
        else:
            # AffectNet/test/align
            # AffectNet/test/original
            # AffectNet/test/error
            folder_path = os.path.join(root_path, data_type.name)       

        # AffectNet/test/roi
        roi_path = os.path.join(root_path, EnumType.roi.name)
                
        # 0~7
        for class_folder in range(0, class_number):                
            class_folder = str(class_folder)
            
            # AffectNet/test/original/0
            class_folder_path = os.path.join(folder_path, class_folder)

            class_folder_listdir = os.listdir(class_folder_path)
            self.class_number_sum.append(len(class_folder_listdir))

            for image in class_folder_listdir: 
                
                # 檢查 roi 的檔案是否存在
                roi_eye_path = os.path.join(roi_path, EnumRoi.eye.name, class_folder, image)
                is_roi = os.path.isfile(roi_eye_path)
                self.is_roi.append(is_roi)

                self.lbls.append(int(class_folder))

                if data_type.name == EnumType.roi.name:
                    # whole is align image
                    self.imgs.append(os.path.join(root_path, roi_image_data_type.name, class_folder, image))
                elif data_type.name == EnumType.re.name:                                             
                    # 檢查 roi 的檔案是否存在
                    # 存在時為 align，不存在時為 error
                    if is_roi:                        
                        self.imgs.append(os.path.join(root_path, EnumType.align.name, class_folder, image))
                    else:                        
                        self.imgs.append(os.path.join(root_path, EnumType.error.name, class_folder, image))
                else:
                    # orininal, align, error, re(re us original) use
                    self.imgs.append(os.path.join(class_folder_path, image))
                
                if is_roi:
                    self.eye_imgs.append(os.path.join(roi_path, EnumRoi.eye.name, class_folder, image))
                    self.nose_imgs.append(os.path.join(roi_path, EnumRoi.nose.name, class_folder, image))
                    self.mouth_imgs.append(os.path.join(roi_path, EnumRoi.mouth.name, class_folder, image))
                else:
                    self.eye_imgs.append("")
                    self.nose_imgs.append("")
                    self.mouth_imgs.append("")                
            

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform)
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------

        # label
        label = self.lbls[index]
        
        eye_image, nose_image, mouth_image = torch.zeros(size=(1,0)), torch.zeros(size=(1,0)), torch.zeros(size=(1,0))
               
        # original              
        img_path = self.imgs[index]
        image_open = Image.open(img_path)
        image = {"train": self.transform_train(image_open), "test" : self.transform_test(image_open) }

        # have roi
        is_roi = self.is_roi[index]
        
        # eye        
        if is_roi:
            eye_img_path = self.eye_imgs[index]
            eye_image_open = Image.open(eye_img_path)            
            eye_image = {"train": self.transform_train(eye_image_open), "test" : self.transform_test(eye_image_open) }

        # nose        
        if is_roi:
            nose_img_path = self.nose_imgs[index]
            nose_image_open = Image.open(nose_img_path)            
            nose_image = {"train": self.transform_train(nose_image_open), "test" : self.transform_test(nose_image_open) }

        # mouth        
        if is_roi:
            mouth_img_path = self.mouth_imgs[index]
            mouth_image_open = Image.open(mouth_img_path)            
            mouth_image = {"train": self.transform_train(mouth_image_open), "test" : self.transform_test(mouth_image_open) }

        return label, image, is_roi, eye_image, nose_image, mouth_image

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.imgs)
    
    # def get_split(self):        
    #     return self.lbls, self.imgs, self.class_number_sum