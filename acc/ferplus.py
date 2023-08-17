# %%
# 自建立的 Dataset 和 Model
from torch.utils.tensorboard.summary import hparams
import dataset.FERPlus as Dataset
import dataset.FERPlus_no_roi as DatasetNoRoi
from dataset.my_enum import *
import package.parameter_ferplus as parameter
import package.fun_ensemble_ferplus as fun

import datetime
import os

import torch
import torch.nn as nn

import numpy as np
from pathlib import Path
# from tqdm import tqdm_notebook as tqdm
from torch.utils.tensorboard import SummaryWriter


# %% [markdown]
#  ### 會調整的參數設定

# %%
p = parameter.Parameter()

# gpu
cuda_drivers = "0"
# cuda_drivers = str(input(f"Enter cuda (default:{cuda_drivers}): ") or cuda_drivers)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_drivers

# 訓練好的模型路徑，有二個
# 以下的 roi 指 Global-ROI Net
# 以下的 global 指 Global Net
path = 'model/ferplus'
path_model_roi = os.path.join(path, 'roi.path')
path_model_global = os.path.join(path, 'global.path')
path_model_roi = str(input(f"Enter path_model_roi (default:{path_model_roi}): ") or path_model_roi)
path_model_global = str(input(f"Enter path_model_global (default:{path_model_global}): ") or path_model_global)

#資料庫路徑
p.path = str(input(f"Enter db path (default:{p.path}): ") or p.path)

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
        'global/BestAcc/train': 0,        
        'BestAcc/validation': 0,        
        
        'roi/BestLoss/validation': 0,        
        'global/BestLoss/validation': 0,
        'BestLoss/validation': 0,
    })
writer.file_writer.add_summary(exp)                 
writer.file_writer.add_summary(ssi)                 
writer.file_writer.add_summary(sei) 

# %%
criterion = nn.BCEWithLogitsLoss().cuda() # 沒有 label_smoothing

# %%

# %% [markdown]
#  ### 訓練、測試、驗證資料

# %%
# roi
test_dataset_roi = Dataset.Dataset(root_path=p.path, run=parameter.EnumRun.test ,transform=p.transform_val,
                              roi_transform=p.roi_transform_val_no_padding, data_type=parameter.EnumType.roi)
test_loader_roi = torch.utils.data.DataLoader(test_dataset_roi, batch_size=p.batch_size_test,shuffle=p.shuffle,num_workers=p.num_workers)    

# global
test_dataset_global = DatasetNoRoi.Dataset(root_path=p.path, run=parameter.EnumRun.test ,transform=p.transform_val,
                             data_type=parameter.EnumType.original)
test_loader_global = torch.utils.data.DataLoader(test_dataset_global, batch_size=p.batch_size_test,shuffle=p.shuffle,num_workers=p.num_workers)    

# roi+global(error)
test_dataset = Dataset.Dataset(root_path=p.path, run=parameter.EnumRun.test, transform=p.transform_val, 
                              roi_transform=p.roi_transform_val_no_padding, data_type=parameter.EnumType.re)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=p.batch_size_test,shuffle=p.shuffle,num_workers=p.num_workers)    

print("\nroi的test數量：")
print(len(test_loader_roi))
print("\nglobal的test數量：")
print(len(test_loader_global))
print("\ntest數量：")
print(len(test_loader))

# %%
starttime = datetime.datetime.now()

# evaluate on validation set
best_model_roi = torch.load(path_model_roi).cuda()
best_model_global = torch.load(path_model_global).cuda()

best_acc_test_roi, best_loss_test_roi, test_target_lists_roi, test_predict_lists_roi, bingo_cnt_roi, sample_cnt_roi = fun.test_roi(test_loader_roi, criterion, path_model_roi)        
print("bingo_cnt_roi:",bingo_cnt_roi, " sample_cnt_roi:",sample_cnt_roi, " acc_roi:", bingo_cnt_roi/sample_cnt_roi)

best_acc_test_global, best_loss_test_global, test_target_lists_global, test_predict_lists_global, bingo_cnt_global, sample_cnt_global = fun.test_global(test_loader_global, criterion, path_model_global)        
print("bingo_cnt_global:",bingo_cnt_global, " sample_cnt_global:",sample_cnt_global, " acc_global:", bingo_cnt_global/sample_cnt_global)

best_acc_test, best_loss_test, test_target_lists, test_predict_lists, bingo_cnt, sample_cnt = fun.test(test_loader, criterion, path_model_roi, path_model_global, beta)        
print("bingo_cnt:",bingo_cnt, " sample_cnt:",sample_cnt, " acc:", bingo_cnt/sample_cnt)

print('-------------------------------------------------')
print('test roi ', best_acc_test_roi, best_loss_test_roi)
print('test global ', best_acc_test_global, best_loss_test_global)
print('test ', best_acc_test, best_loss_test)

# BestAcc
writer.add_scalar('roi/BestAcc/validation', best_acc_test_roi)
writer.add_scalar('global/BestAcc/validation', best_acc_test_global)
writer.add_scalar('BestAcc/validation', best_acc_test)

writer.add_scalar('roi/BestLoss/validation', best_loss_test_roi)
writer.add_scalar('global/BestLoss/validation', best_loss_test_global)
writer.add_scalar('BestLoss/validation', best_loss_test)

#long running
endtime = datetime.datetime.now()
totaltime = endtime - starttime
print('total time: ', totaltime)
print('total time (seconds): ', totaltime.seconds)

# %%
writer.flush()
writer.close()