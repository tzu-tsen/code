# %%
from operator import index
import cv2
import dlib
import numpy as np
import os
import shutil
from PIL import Image

# %%
predictor_model = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()# dlib人臉檢測器
predictor = dlib.shape_predictor(predictor_model)

# %%
#圖檔資料夾
annotation_folders = '/autohome/user/tzutsen/database/RAF-DB/my/test'
annotation_folders = str(input(f"Enter annotation_folders (default:{annotation_folders}): ") or annotation_folders)

#新的圖檔資料夾
new_folders = "test"
new_folders = str(input(f"Enter new_folders (default:{new_folders}): ") or new_folders)

# 返回資料夾名稱列表。
folders = os.listdir(annotation_folders)

sides={0:"face_alignment",1:"no_face_alignment"}
#圖檔尺寸
image_size=224

#%%
for folder in folders:
# for folder in range(1,2):
#     folder = str(folder)
    # 子資料夾路徑
    child_folder = os.path.join(annotation_folders,folder)    
    # 返回子資料夾名稱列表。
    child_files = os.listdir(child_folder)
    # 所有子資料夾資料
    for child_file in child_files:  
        # 子資料夾+圖檔路徑
        image_path = os.path.join(child_folder,child_file)
        # cv2讀取圖像
        img = cv2.imread(image_path)
        # 取灰度
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # opencv的顏色空間是BGR，需要轉為RGB才能用在dlib中
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 檢測圖片中的人臉
        rects = detector(rgb_img, 1)
        if len(rects)>0:
            # 識別人臉特徵點，並儲存下來
            faces = dlib.full_object_detections()
            for rect in rects:
                faces.append(predictor(rgb_img, rect))
            
            # 人臉對齊
            images = dlib.get_face_chips(rgb_img, faces, size=image_size)

            image_cnt=0
            # 顯示對齊結果
            for image in images:
                image_cnt += 1
                # 先轉換為numpy陣列
                cv_rgb_image = np.array(image).astype(np.uint8)
                # opencv下顏色空間為bgr，所以從rgb轉換為bgr
                cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)
                new_folder = os.path.join(new_folders, sides[0], folder)
                if not os.path.isdir(new_folder):
                    os.makedirs(new_folder)
                new_image_path = os.path.join(new_folder, child_file)
                cv2.imwrite(new_image_path, cv_bgr_image)            
        else: 
        # if len(rects)<=0:         
            new_folder = os.path.join(new_folders, sides[1],str(folder))
            if not os.path.isdir(new_folder):
                os.makedirs(new_folder)
            new_image_path = os.path.join(new_folder, child_file)
            cv_bgr_image = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(new_image_path, cv_bgr_image)                    

        print(child_file)
        
    print(folder, "完成")    

print("完成")
# %%
