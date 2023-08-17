# %%
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from PIL import Image
import random
from dataset.my_enum import *

# %%
class Dataset(Dataset):
    def __init__(self, root_path, transform, run = EnumRun.train, data_type=EnumType.original,roi_image_data_type=EnumType.align):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.imgs = []
        self.lbls = []          
        self.transform = transform
        self.class_number_sum = []
        
        # label
        label_path = os.path.join(root_path, "label",run.name)
        self.class_number_sum = np.load(os.path.join(label_path,'class_number_sum.npy'))
        
        # folder
        root_path = os.path.join(root_path, run.name)
        if data_type.name == EnumType.roi.name:     
            # FERPlus/test/roi/eye       
            folder_path = os.path.join(root_path, data_type.name, EnumRoi.eye.name)
        elif data_type.name == EnumType.re.name:     
            # FERPlus/test/original
            # roi+error = original
            folder_path = os.path.join(root_path, EnumType.original.name)
        else:
            # FERPlus/test/align
            # FERPlus/test/original
            # FERPlus/test/error
            folder_path = os.path.join(root_path, data_type.name)
                

        # label_listdir = os.listdir(label_path)
        # label_listdir.remove('class_number_sum.npy')
        # label_listdir.remove('label.csv')

        for image in os.listdir(folder_path):
            # 檢查 label 的檔案是否存在
            basename = os.path.basename(image) # basename - example.py
            filename = os.path.splitext(basename)[0]  # filename - example             
            image_label_path = os.path.join(label_path, f'{filename}.npy')            
            if os.path.isfile(image_label_path):
                # label
                lbls = [float(x) for x in np.load(image_label_path)]
                self.lbls.append(lbls)
                
                #image
                if data_type.name == EnumType.roi.name:   
                    self.imgs.append(os.path.join(root_path, roi_image_data_type.name, image))                              
                elif data_type.name == EnumType.re.name:  
                    # 檢查 roi 的檔案是否存在
                    roi_eye_path = os.path.join(root_path, EnumType.roi.name, EnumRoi.eye.name, image)
                    is_roi = os.path.isfile(roi_eye_path)

                    # 檢查 roi 的檔案是否存在
                    # 存在時為 align，不存在時為 error
                    if is_roi:                        
                        self.imgs.append(os.path.join(root_path, EnumType.align.name, image))
                    else:                        
                        self.imgs.append(os.path.join(root_path, EnumType.error.name, image))
                else:                   
                    self.imgs.append(os.path.join(folder_path, image))
       
                    
    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform)
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------

        # label
        label = np.array(self.lbls[index])
               

        # original
        image = self.imgs[index] 
        img_path = self.imgs[index]                  
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)         

        return label,image

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.imgs)

    def get_split(self):        
        return self.lbls, self.imgs, self.class_number_sum