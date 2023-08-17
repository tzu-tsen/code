# %%
from operator import index
import cv2
import dlib
import numpy as np
import os
import shutil
from PIL import Image
import csv

# %% [markdown]
# #### 圖檔資料夾

# %%
#圖檔資料夾
annotation_folders = 'test/face_alignment'
annotation_folders = str(input(f"Enter annotation_folders (default:{annotation_folders}): ") or annotation_folders)
#新的圖檔資料夾
new_folders = 'face_roi/roi'
new_folders = str(input(f"Enter new_folders (default:{new_folders}): ") or new_folders)
rois={0:"eye", 1:"nose", 2:"mouth", 3:"original"}
error_folders = 'face_roi/error'
error_folders = str(input(f"Enter error_folders (default:{error_folders}): ") or error_folders)

# %% [markdown]
# #### 載入 model

# %%
predictor_model = '/autohome/user/tzutsen/dlib_model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()# dlib人臉檢測器
predictor = dlib.shape_predictor(predictor_model)

# %% [markdown]
# #### 儲存 ROI

# %%
def saveRoi(img_path):
    img_path_array = img_path.split('/')
    img_name = img_path_array[-1]
    img_folder = img_path_array[-2]
    new_image_folder = new_folders

    img_gray = cv2.imread(img_path)

    # 人臉數rects（rectangles）
    rects = detector(img_gray, 0)

    pos_x = {}
    pos_y = {}
    pos_x_nose = {}
    pos_y_nose = {}
    pos_x_eye = {}
    pos_y_eye = {}
    pos_x_mouth = {}
    pos_y_mouth = {}

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y]
                              for p in predictor(img_gray, rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68點的座標
            pos = (point[0, 0], point[0, 1])
            pos_x[idx] = pos[0]
            pos_y[idx] = pos[1]

    # print(pos_x)
    # print(pos_y)

    # 眉
    for i in range(17, 27):
        pos_x_eye[i] = pos_x[i]
        pos_y_eye[i] = pos_y[i]

    # 眼
    for i in range(36, 48):
        pos_x_eye[i] = pos_x[i]
        pos_y_eye[i] = pos_y[i]

    # 鼻
    for i in range(27, 36):
        pos_x_nose[i] = pos_x[i]
        pos_y_nose[i] = pos_y[i]

    # 嘴巴
    for i in range(48, 60):
        pos_x_mouth[i] = pos_x[i]
        pos_y_mouth[i] = pos_y[i]

    # 眼睛
    pos_x_eye_max = pos_x_eye[max(pos_x_eye, key=pos_x_eye.get)]
    pos_x_eye_min = pos_x_eye[min(pos_x_eye, key=pos_x_eye.get)]
    
    eye_pt1 = (pos_x_eye_min-3, pos_y_eye[min(pos_y_eye, key=pos_y_eye.get)]-3)
    eye_pt2 = (pos_x_eye_max+3, pos_y_eye[max(pos_y_eye, key=pos_y_eye.get)]+6)
    eye_img = cv2.imread(img_path)
    # 在圖片上畫一個綠色方框
    # cv2.rectangle(eye_img, eye_pt1, eye_pt2, (0, 255, 0), 1)
    eye_folder = os.path.join(new_image_folder, rois[0], img_folder)
    if not os.path.isdir(eye_folder):
        os.makedirs(eye_folder)
    eye_img_path = os.path.join(eye_folder, img_name)
    # cv2.imwrite(nose_img_path, eye_img)
    # 截切
    cv2.imwrite(eye_img_path, eye_img[eye_pt1[1]:eye_pt2[1], eye_pt1[0]:eye_pt2[0]],[int( cv2.IMWRITE_JPEG_QUALITY), 100])

    # 鼻子
    pos_x_nose_max = pos_x_nose[max(pos_x_nose, key=pos_x_nose.get)]
    pos_x_nose_min = pos_x_nose[min(pos_x_nose, key=pos_x_nose.get)]

    nose_pt1 = (pos_x_nose_min-int((pos_x[33]-pos_x[31])/2),
                pos_y_nose[min(pos_y_nose, key=pos_y_nose.get)]-4)
    nose_pt2 = (pos_x_nose_max+int((pos_x[35]-pos_x[33])/2),
                pos_y_nose[max(pos_y_nose, key=pos_y_nose.get)]+4)
    nose_img = cv2.imread(img_path)
    # 在圖片上畫一個綠色方框
    # cv2.rectangle(nose_img, nose_pt1, nose_pt2, (0, 255, 0), 1)
    nose_folder = os.path.join(new_image_folder, rois[1], img_folder)
    if not os.path.isdir(nose_folder):
        os.makedirs(nose_folder)
    nose_img_path = os.path.join(nose_folder, img_name)
    # cv2.imwrite(nose_img_path, nose_img)
    # 截切
    cv2.imwrite(nose_img_path, nose_img[nose_pt1[1]:nose_pt2[1], nose_pt1[0]:nose_pt2[0]],[int( cv2.IMWRITE_JPEG_QUALITY), 100])

    # 嘴巴
    pos_x_mouth_max = pos_x_mouth[max(pos_x_mouth, key=pos_x_mouth.get)]
    pos_x_mouth_min = pos_x_mouth[min(pos_x_mouth, key=pos_x_mouth.get)]
    
    mouth_pt1 = (pos_x_mouth_min-6,
                 pos_y_mouth[min(pos_y_mouth, key=pos_y_mouth.get)]-6)
    mouth_pt2 = (pos_x_mouth_max+6,
                 pos_y_mouth[max(pos_y_mouth, key=pos_y_mouth.get)]+6)
    mouth_img = cv2.imread(img_path)
    # 在圖片上畫一個綠色方框
    # cv2.rectangle(mouth_img, mouth_pt1, mouth_pt2, (0, 255, 0), 1)
    mouth_folder = os.path.join(new_image_folder, rois[2], img_folder)
    if not os.path.isdir(mouth_folder):
        os.makedirs(mouth_folder)
    mouth_img_path = os.path.join(mouth_folder, img_name)
    # cv2.imwrite(mouth_img_path, mouth_img)
    # 截切
    cv2.imwrite(mouth_img_path, mouth_img[mouth_pt1[1]:mouth_pt2[1], mouth_pt1[0]:mouth_pt2[0]],[int(cv2.IMWRITE_JPEG_QUALITY), 100])


# %% [markdown]
# ### 依路徑執行 ROI

# %%
folder_path = os.path.join(annotation_folders)        
# 返回資料夾名稱列表。
# 0-7
folders = os.listdir(folder_path)    
for folder in folders:    
    # 子資料夾路徑
    child_folder = os.path.join(folder_path,folder)            
    # 返回子資料夾名稱列表。
    child_files = os.listdir(child_folder)        
    # 所有子資料夾資料
    for child_file in child_files:  
        # 子資料夾+圖檔路徑
        image_path = os.path.join(child_folder, child_file)        
        print(image_path) 
        
        try:   
            # roi
            saveRoi(image_path)

            # 原始
            new_image_folder = os.path.join(new_folders, rois[3], folder)
            if not os.path.isdir(new_image_folder):
                os.makedirs(new_image_folder)
            new_image_path = os.path.join(new_image_folder, child_file)
            shutil.copyfile(image_path, new_image_path)           
        except:   
            error_folder = os.path.join(error_folders,folder)         
            if not os.path.isdir(error_folder):
                os.makedirs(error_folder)
            error_image_path = os.path.join(error_folder, child_file)
            shutil.copyfile(image_path, error_image_path)
            
                                   
print("完成")


