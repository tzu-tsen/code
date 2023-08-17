from torchvision import transforms
import os
from datetime import datetime
from enum import Enum

class EnumType(Enum):
    original = 0 #原始
    align = 1 #對齊
    error = 2 #無法對齊和取roi
    roi = 3 
    re = 4 #roi+error

class Parameter:
    # 建構式
    def __init__(self,                      
        num_workers=8,         
        database="AffectNet",
        cm_normalize = True,
        size=224,
        roi_size=90,
        n_head=8,
        num_class_end=8,
        num_class_start=0
        ):
        
        self.n_head = n_head
        
        self.cm_normalize = cm_normalize
        self.ls = 0.1

        # 核心數
        self.num_workers = num_workers

        self.batch_size_test=1
        
        # 類別
        self.num_class = num_class_end-num_class_start
        class_names=['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
        # 預設8類，到 contempt。若為 7 類，會到 Anger
        self.class_names= class_names[num_class_start:num_class_end]
        self.num_class_start=num_class_start
        self.num_class_end=num_class_end

        #資集的路徑
        path_database = "/autohome/user/tzutsen/database"        
        self.path_train = os.path.join(path_database, database, "train")
        self.path_test = os.path.join(path_database, database, "test")        

        # 日期和時間戳        
        now = datetime.now() # current date and time
        self.timestamp = now.strftime("%m%d_%H-%M-%S")

        self.pin_memory = True   
        self.shuffle = True

        self.size = size
        self.roi_size = roi_size
        # transform
        self.array_mean=[0.485, 0.456, 0.406]
        self.array_std=[0.229, 0.224, 0.225]
                
        self.transform_val = transforms.Compose(
        [       
            transforms.Resize((self.size, self.size)),     
            transforms.ToTensor(),
            transforms.Normalize(self.array_mean,self.array_std)
        ])

        self.roi_transform_train_no_padding_randomerasing = transforms.Compose(
        [
            transforms.Resize((self.roi_size,self.roi_size)), #90*90
            transforms.RandomHorizontalFlip(),
            #亮度（brightness）、對比度（contrast）、飽和度（saturation）和色调（hue)
            # transforms.ColorJitter(brightness=(0.5,1.3), contrast=(0.7,1.2)),  
            transforms.ColorJitter(brightness=0.2, contrast=0.2),              
            transforms.RandomApply([
                # 角度20
                # scale 縮放比例 0.8～1
                # translate 平移區間 寬：-w*0.2<dx<w*0.2。高：-h*0.2<dx<h*0.2
                    transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
                ], p=0.7),
            transforms.ToTensor(),
            transforms.Normalize(self.array_mean,self.array_std),
            transforms.RandomErasing(p=0.2,scale=(0.02, 0.25))
        ])
        
        self.roi_transform_val_no_padding = transforms.Compose(
        [
            transforms.Resize((self.roi_size,self.roi_size)), #90*90            
            transforms.ToTensor(),
            transforms.Normalize(self.array_mean,self.array_std)
        ])