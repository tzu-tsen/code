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
        batch_size_roi=64, 
        batch_size_whole=64,
        num_epochs=100, 
        num_workers=8, 
        lr_roi=0.5,
        lr_whole=0.1,
        milestones_roi = [10,20,30,40,50,60,70,80,90],
        milestones_whole = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95],
        gamma_roi = 0.1,
        gamma_whole = 0.75,
        weight_decay=0.0001, 
        momentum = 0.9,
        num_class=7, 
        num_class_samples_roi=3000,
        num_class_samples_whole=3000,
        database="RAF-DB",
        cm_normalize = True,
        beta_roi = [1, 0.6, 1],
        size=224,
        roi_size=90,        
        n_head=8):
        
        self.num_resnet_cbam_roi = 1        
        self.n_head = n_head
        
        self.cm_normalize = cm_normalize
                
        # 一個批次大小是幾張圖
        self.batch_size_roi = batch_size_roi
        self.batch_size_whole = batch_size_whole
        self.batch_size_test = 1

        # 總循環週期
        self.num_epochs = num_epochs
        # 核心數
        self.num_workers = num_workers
        
        # 類別
        self.num_class = num_class
        class_names=['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
        # 預設8類，到 contempt。若為 7 類，會到 Anger
        self.class_names= class_names[:self.num_class]

        # 學習率
        self.lr_roi = lr_roi
        self.lr_whole = lr_whole
        
        # 標籤平滑
        self.ls=0.1

        # 幾 epoch 個下降學習率
        self.milestones_roi = milestones_roi
        self.milestones_whole = milestones_whole
        # 每次下降多少
        self.gamma_roi = gamma_roi
        self.gamma_whole = gamma_whole

        # dampening for momentum (default: 0)
        # SGD use
        self.momentum = momentum  

        # 權重衰減（L2 懲罰）（默認值：0）即 L2regularization，
        # 選擇一個合適的權重衰減係數λ非常重要，這個需要根據具體的情況去嘗試，初步嘗試可以使用 1e-4 或者1e-3   
        self.weight_decay=weight_decay
        
        #超參數
        self.beta_roi=beta_roi

        self.num_samples_roi=num_class_samples_roi*num_class
        self.num_samples_whole = num_class_samples_whole*num_class
        
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
        self.transform_train = transforms.Compose(
        [
            transforms.Resize((self.size, self.size)),            
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
            transforms.RandomErasing(scale=(0.02, 0.25))
        ])
        
        self.transform_val = transforms.Compose(
        [       
            transforms.Resize((self.size, self.size)),     
            transforms.ToTensor(),
            transforms.Normalize(self.array_mean,self.array_std)
        ])

        self.roi_transform_train_no_padding = transforms.Compose(
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
            transforms.RandomErasing(scale=(0.02, 0.25))
        ])        
    

        self.roi_transform_val_no_padding = transforms.Compose(
        [
            transforms.Resize((self.roi_size,self.roi_size)), #90*90            
            transforms.ToTensor(),
            transforms.Normalize(self.array_mean,self.array_std)
        ])