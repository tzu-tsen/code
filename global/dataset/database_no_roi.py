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
    def __init__(self, root_path, transform, class_number=8, data_type=EnumType.original,roi_image_data_type=EnumType.align):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.imgs = []
        self.lbls = []
        # self.is_roi = []
        # self.eye_imgs = []
        # self.nose_imgs = []
        # self.mouth_imgs = []        
        self.transform = transform
        # self.roi_transform = roi_transform
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
                
        # 0~7
        for class_folder in range(0, class_number):                
            class_folder = str(class_folder)           
            
            # 類別資料夾的數量
            class_folder_path = os.path.join(folder_path, class_folder)
            class_folder_listdir = os.listdir(class_folder_path)
            self.class_number_sum.append(len(class_folder_listdir))

            for image in class_folder_listdir: 
                self.lbls.append(int(class_folder))
                if data_type.name == EnumType.roi.name:   
                    self.imgs.append(os.path.join(root_path, roi_image_data_type.name, class_folder, image))                              
                elif data_type.name == EnumType.re.name:  
                    # 檢查 roi 的檔案是否存在
                    roi_eye_path = os.path.join(root_path, EnumType.roi.name, EnumRoi.eye.name, class_folder, image)
                    is_roi = os.path.isfile(roi_eye_path)

                    # 檢查 roi 的檔案是否存在
                    # 存在時為 align，不存在時為 error
                    if is_roi:                        
                        self.imgs.append(os.path.join(root_path, EnumType.align.name, class_folder, image))
                    else:                        
                        self.imgs.append(os.path.join(root_path, EnumType.error.name, class_folder, image))
                else:                   
                    self.imgs.append(os.path.join(class_folder_path, image))                              
            

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform)
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------

        # label
        label = self.lbls[index]
        
              
        # original              
        img_path = self.imgs[index]
        image = Image.open(img_path)
        image = self.transform(image)

        return label, image

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.imgs)
    
    def get_split(self):        
        return self.lbls, self.imgs, self.class_number_sum