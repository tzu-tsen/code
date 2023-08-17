# %%
# 自建立的 Dataset 和 Model
from torch.utils.tensorboard.summary import hparams
import dataset.database as Dataset

import package.confusion_matrix as ConfusionMatrix
import package.parameter as parameter
import package.fun_ensemble_rafdb as fun

import datetime
import os

import torch
import torch.nn as nn

from pathlib import Path
# from tqdm import tqdm_notebook as tqdm
from torch.utils.tensorboard import SummaryWriter


# %% [markdown]
#  ### 會調整的參數設定
p = parameter.Parameter(database="CK+")

# gpu
cuda_drivers = "0"
# cuda_drivers = str(input(f"Enter cuda (default:{cuda_drivers}): ") or cuda_drivers)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_drivers

# 訓練好的模型路徑，有二個
# 以下的 roi 指 Global-ROI Net
# 以下的 global 指 Global Net
path = 'model/affectnet8'
path_model_roi = os.path.join(path, 'roi.path')
path_model_global = os.path.join(path, 'global.path')
path_model_roi = str(input(f"Enter path_model_roi (default:{path_model_roi}): ") or path_model_roi)
path_model_global = str(input(f"Enter path_model_global (default:{path_model_global}): ") or path_model_global)

#資料庫路徑
p.path_test = str(input(f"Enter db path (default:{p.path_test}): ") or p.path_test)

#權重
beta_global = 0.45
# beta_global = float(input(f"Enter beta_global (default:{beta_global}): ") or beta_global)
beta_roi = 1-beta_global
# beta_roi = float(input(f"Enter beta_roi (default:{beta_roi}): ") or beta_roi)
beta = [beta_roi, beta_global]
print("beta[beta_roi, beta_global]:",beta)
# %%
# 檔案名稱
path_file = os.path.realpath(__file__)
file_name = Path(path_file).stem
print('file_name:', file_name)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter(f"runs/{file_name}/{p.timestamp}")
exp, ssi, sei = hparams(
    {    
        'beta_roi':beta_roi,
        'beta_global':beta_global     
    },
    metric_dict={
        'roi/BestAcc/validation': 0,
        'global/BestAcc/validation': 0,        
        'BestAcc/validation': 0,        
        
        'roi/BestLoss/validation': 0,        
        'global/BestLoss/validation': 0,
        'BestLoss/validation': 0,
    })
writer.file_writer.add_summary(exp)
writer.file_writer.add_summary(ssi)
writer.file_writer.add_summary(sei)
# %%
criterion = nn.CrossEntropyLoss(label_smoothing=p.ls).cuda()  # 交叉熵损失函数

# %%

# %% [markdown]
#  ### 訓練、測試、驗證資料

# %%
# test, AffectNet only val, val is test
# roi
val_dataset_roi = Dataset.Dataset(root_path=p.path_test,transform=p.transform_val,class_number=p.num_class, 
                              roi_transform=p.roi_transform_val_no_padding, data_type=parameter.EnumType.roi)
val_loader_roi = torch.utils.data.DataLoader(val_dataset_roi, batch_size=p.batch_size_test,shuffle=p.shuffle,num_workers=p.num_workers)    

# # global
val_dataset_global = Dataset.Dataset(root_path=p.path_test,transform=p.transform_val,class_number=p.num_class, 
                              roi_transform=p.roi_transform_val_no_padding, data_type=parameter.EnumType.original)
val_loader_global = torch.utils.data.DataLoader(val_dataset_global, batch_size=p.batch_size_test,shuffle=p.shuffle,num_workers=p.num_workers)    

# roi+global(error)
val_dataset = Dataset.Dataset(root_path=p.path_test,transform=p.transform_val,class_number=p.num_class, 
                              roi_transform=p.roi_transform_val_no_padding, data_type=parameter.EnumType.re)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=p.batch_size_test,shuffle=p.shuffle,num_workers=p.num_workers)    

print("\nroi的val數量：")
print(len(val_loader_roi))
print("\nglobal的val數量：")
print(len(val_loader_global))
print("\nval數量：")
print(len(val_loader))

# %%

starttime = datetime.datetime.now()

# evaluate on validation set
best_model_roi = torch.load(path_model_roi).cuda()
best_model_global = torch.load(path_model_global).cuda()

val_acc_roi, val_loss_roi, val_target_lists_roi, val_predict_lists_roi, bingo_cnt_roi, sample_cnt_roi = fun.validate_roi(
        val_loader_roi, best_model_roi, criterion)

print("bingo_cnt_roi:",bingo_cnt_roi, " sample_cnt_roi:",sample_cnt_roi, " acc_roi:", bingo_cnt_roi/sample_cnt_roi)

val_acc_global, val_loss_global, val_target_lists_global, val_predict_lists_global, bingo_cnt_global, sample_cnt_global = fun.validate_global(
         val_loader_global, best_model_global, criterion)

print("bingo_cnt_global:",bingo_cnt_global, " sample_cnt_global:",sample_cnt_global, " acc_global:", bingo_cnt_global/sample_cnt_global)

val_acc, val_loss, val_target_lists, val_predict_lists, bingo_cnt, sample_cnt = fun.validate(
    val_loader, best_model_roi, best_model_global, criterion, beta=beta)

print("bingo_cnt:",bingo_cnt, " sample_cnt:",sample_cnt, " acc:", bingo_cnt/sample_cnt)

print('-------------------------------------------------')
print('validation roi ', val_acc_roi, val_loss_roi)
print('validation global ', val_acc_global, val_loss_global)
print('validation ', val_acc, val_loss)

# confusion matrix
val_figure_cm_roi = ConfusionMatrix.figure_confusion_matrix(
    val_target_lists_roi, val_predict_lists_roi, p.class_names, p.cm_normalize)
writer.add_figure('Confusion Matrix val roi', val_figure_cm_roi, 1)

val_figure_cm_global = ConfusionMatrix.figure_confusion_matrix(
    val_target_lists_global, val_predict_lists_global, p.class_names, p.cm_normalize)
writer.add_figure('Confusion Matrix val global', val_figure_cm_global, 1)

val_figure_cm = ConfusionMatrix.figure_confusion_matrix(
    val_target_lists, val_predict_lists, p.class_names, p.cm_normalize)
writer.add_figure('Confusion Matrix val', val_figure_cm, 1)

# BestAcc
writer.add_scalar('roi/BestAcc/validation', val_acc_roi)
writer.add_scalar('global/BestAcc/validation', val_acc_global)
writer.add_scalar('BestAcc/validation', val_acc)

writer.add_scalar('roi/BestLoss/validation', val_loss_roi)
writer.add_scalar('global/BestLoss/validation', val_loss_global)
writer.add_scalar('BestLoss/validation', val_loss)

# long running
endtime = datetime.datetime.now()
totaltime = endtime - starttime
print('total time: ', totaltime)
print('total time (seconds): ', totaltime.seconds)

# %%
writer.flush()
writer.close()
