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
    def __init__(self, root_path, run, transform, roi_transform, data_type=EnumType.roi,roi_image_data_type=EnumType.align):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.imgs = []
        self.lbls = []
        self.lblsPath = []
        self.is_roi = []
        self.eye_imgs = []
        self.nose_imgs = []
        self.mouth_imgs = []        
        self.transform = transform
        self.roi_transform = roi_transform
        self.class_number_sum = []

        # label
        label_path = os.path.join(root_path, "label", run.name)
        self.class_number_sum = np.load(os.path.join(label_path,'class_number_sum.npy'))
        
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

        # AffectNet/test/roi
        roi_path = os.path.join(root_path, EnumType.roi.name)
              
        for image in os.listdir(folder_path): 
            # 檢查 label 的檔案是否存在
            basename = os.path.basename(image) # basename - example.py
            filename = os.path.splitext(basename)[0]  # filename - example             
            image_label_path = os.path.join(label_path, f'{filename}.npy')            
            
            if os.path.isfile(image_label_path):
                # label
                lbls = [float(x) for x in np.load(image_label_path)]
                self.lbls.append(lbls)
                self.lblsPath.append(image_label_path)
                
                # 檢查 roi 的檔案是否存在
                roi_eye_path = os.path.join(roi_path, EnumRoi.eye.name, image)
                is_roi = os.path.isfile(roi_eye_path)
                self.is_roi.append(is_roi)

                if data_type.name == EnumType.roi.name:
                    # roi_image_data_type default is align image
                    self.imgs.append(os.path.join(root_path, roi_image_data_type.name, image))
                elif data_type.name == EnumType.re.name:                                             
                    # 檢查 roi 的檔案是否存在
                    # 存在時為 align，不存在時為 error
                    if is_roi:                        
                        self.imgs.append(os.path.join(root_path, EnumType.align.name, image))
                    else:                        
                        self.imgs.append(os.path.join(root_path, EnumType.error.name, image))
                else:
                    # orininal, align, error, re(re us original) use
                    self.imgs.append(os.path.join(folder_path, image))
                
                if is_roi:
                    self.eye_imgs.append(os.path.join(roi_path, EnumRoi.eye.name, image))
                    self.nose_imgs.append(os.path.join(roi_path, EnumRoi.nose.name, image))
                    self.mouth_imgs.append(os.path.join(roi_path, EnumRoi.mouth.name, image))
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
        label = np.array(self.lbls[index])
        
        eye_image, nose_image, mouth_image = torch.zeros(size=(1,0)), torch.zeros(size=(1,0)), torch.zeros(size=(1,0))

        # have roi
        is_roi = self.is_roi[index]
       
        # original              
        img_path = self.imgs[index]
        image = Image.open(img_path)      
        if is_roi:
            image = Image.open(img_path).convert('RGB')
        image = self.transform(image)   
        
        # eye        
        if is_roi:
            eye_img_path = self.eye_imgs[index]
            eye_image = Image.open(eye_img_path)
            eye_image = self.roi_transform(eye_image)

        # nose        
        if is_roi:
            nose_img_path = self.nose_imgs[index]
            nose_image = Image.open(nose_img_path)
            nose_image = self.roi_transform(nose_image)

        # mouth        
        if is_roi:
            mouth_img_path = self.mouth_imgs[index]
            mouth_image = Image.open(mouth_img_path)
            mouth_image = self.roi_transform(mouth_image)   

        # print(index,img_path)
        # print("is_roi:", is_roi, self.eye_imgs[index])   
        # print("lblsPath:",self.lblsPath[index]) 
        # print("label:",label)     
        # print('-------------')     

        return label, image, is_roi, eye_image, nose_image, mouth_image

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.imgs)
    
    def get_split(self):        
        return self.lbls, self.imgs, self.class_number_sum