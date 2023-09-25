# %%
# 自建立的 Dataset 和 Model
import dataset.ck as Dataset

import package.confusion_matrix as ConfusionMatrix
import package.parameter_ck as parameter
import package.fun_ensemble_ck as fun

import datetime
import os

import torch
import torch.nn as nn

import numpy as np
from pathlib import Path
# from tqdm import tqdm_notebook as tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

# %% [markdown]
#  ### 會調整的參數設定

# %%
p = parameter.Parameter()

# input
cuda_drivers = "0"
cuda_drivers = str(input(f"Enter cuda (default:{cuda_drivers}): ") or cuda_drivers)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_drivers
# %%
# 訓練好的模型路徑，有二個
# 以下的 roi 指 Global-ROI Net
# 以下的 global 指 Global Net
path = 'model'
path_model_roi_folder = os.path.join(path, 'ck', 'roi')
path_model_global_folder = os.path.join(path, 'ck', 'global')
path_model_roi_folder = str(input(f"Enter path_model_roi_folder (default:{path_model_roi_folder}): ") or path_model_roi_folder)
path_model_global_folder = str(input(f"Enter path_model_global_folder (default:{path_model_global_folder}): ") or path_model_global_folder)

#資料庫路徑
p.path_test = str(input(f"Enter db path (default:{p.path_test}): ") or p.path_test)

#權重
beta_global = 0.45
# beta_global = float(input(f"Enter beta_global (default:{beta_global}): ") or beta_global)
beta_roi = 1-beta_global
# beta_roi = float(input(f"Enter beta_roi (default:{beta_roi}): ") or beta_roi)
beta = [beta_roi, beta_global]
print("beta[beta_roi, beta_global]:",beta)


# 檔案名稱
path_file = os.path.realpath(__file__)
file_name = Path(path_file).stem
print('file_name:', file_name)
   
# %%
criterion = nn.CrossEntropyLoss(label_smoothing=p.ls).cuda() # 交叉熵损失函数

# roi
dataset_roi =    Dataset.Dataset(root_path=p.path_test,transform_train=p.transform_train, transform_test=p.transform_val,class_number=p.num_class, roi_transform_train=p.roi_transform_train_no_padding,roi_transform_test=p.roi_transform_val_no_padding, data_type=parameter.EnumType.roi)

#global
dataset_global = Dataset.Dataset(root_path=p.path_test,transform_train=p.transform_train, transform_test=p.transform_val,class_number=p.num_class, roi_transform_train=p.roi_transform_train_no_padding,roi_transform_test=p.roi_transform_val_no_padding, data_type=parameter.EnumType.original)

# roi+global(error)
dataset =        Dataset.Dataset(root_path=p.path_test,transform_train=p.transform_train, transform_test=p.transform_val, class_number=p.num_class, roi_transform_train=p.roi_transform_train_no_padding,roi_transform_test=p.roi_transform_val_no_padding, data_type=parameter.EnumType.re)

roi_bastacc_test_k_fload  = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
roi_bastloss_test_k_fload  = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]


global_bastacc_test_k_fload  = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
global_bastloss_test_k_fload  = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

bastacc_test_k_fload  = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
bastloss_test_k_fload  = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

starttime = datetime.datetime.now()

for i in range(10):
    print("===================================")
    print(f"Fold {i}:")        

    '-- k-fload dataset --'
    path = os.path.join("k_fload",f'{i}')
    
    # ck+ 的圖像可以全部分出 roi，所以不用特別取 roi、global和無法取出 roi 的圖像
    test_index = np.load(os.path.join(path, "test_index.npy"))
    test_sampler = torch.utils.data.SubsetRandomSampler(test_index)

    test_loader_roi = torch.utils.data.DataLoader(dataset_roi, batch_size=p.batch_size_whole,sampler=test_sampler,num_workers=p.num_workers)    
    test_loader_global = torch.utils.data.DataLoader(dataset_global, batch_size=p.batch_size_whole,sampler=test_sampler,num_workers=p.num_workers)    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=p.batch_size_whole,sampler=test_sampler,num_workers=p.num_workers)    

    print("test數量：",len(test_index))
    
    '--- writer ---'
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(f"runs/ck/{p.timestamp}/{i}")
    
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
    '--- model  ---'
    # evaluate on validation set
    path_model_roi = os.path.join(path_model_roi_folder, f'{i}.path')
    path_model_global = os.path.join(path_model_global_folder, f'{i}.path')
    
    model_roi = torch.load(path_model_roi).cuda()
    model_global = torch.load(path_model_global).cuda()
    
    '--- test ---'   
    # etestuate on test set                       
    test_acc_roi, test_loss_roi, test_target_lists_roi, test_predict_lists_roi = fun.test_roi(test_loader_roi, model_roi, criterion)
    test_acc_global, test_loss_global, test_target_lists_global, test_predict_lists_global = fun.test_global(test_loader_global, model_global, criterion)
    test_acc, test_loss, test_target_lists, test_predict_lists = fun.test(test_loader, model_roi, model_global, criterion, beta)

    print('-------------------------------------------------')
    print('validation roi ', test_acc_roi, test_loss_roi)
    print('validation global ', test_acc_global, test_loss_global)
    print('validation ', test_acc, test_loss)  

    # confusion_maatrix    
    test_figure_cm_roi = ConfusionMatrix.figure_confusion_matrix(test_target_lists_roi, test_predict_lists_roi, p.class_names,p.cm_normalize) 
    writer.add_figure('Confusion Matrix test roi', test_figure_cm_roi, 1)

    test_figure_cm_global = ConfusionMatrix.figure_confusion_matrix(test_target_lists_global, test_predict_lists_global, p.class_names, p.cm_normalize) 
    writer.add_figure('Confusion Matrix test global', test_figure_cm_global, 1)

    test_figure_cm = ConfusionMatrix.figure_confusion_matrix(test_target_lists, test_predict_lists, p.class_names,p.cm_normalize) 
    writer.add_figure('Confusion Matrix test', test_figure_cm, 1)

    # BestAcc
    writer.add_scalar('roi/BestAcc/validation', test_acc_roi)
    writer.add_scalar('global/BestAcc/validation', test_acc_global)
    writer.add_scalar('BestAcc/validation', test_acc)

    writer.add_scalar('roi/BestLoss/validation', test_loss_roi)
    writer.add_scalar('global/BestLoss/validation', test_loss_global)
    writer.add_scalar('BestLoss/validation', test_loss)

    writer.flush()
    writer.close()

    # bastacc_test_k_fload & bastloss_test_k_fload
    roi_bastacc_test_k_fload[i]  = test_acc_roi
    roi_bastloss_test_k_fload[i]  = test_loss_roi

    global_bastacc_test_k_fload[i]  = test_acc_global
    global_bastloss_test_k_fload[i]  = test_loss_global

    bastacc_test_k_fload[i] = test_acc
    bastloss_test_k_fload[i]  = test_loss

# avg
# Writer will output to ./runs/ directory by default
writer1 = SummaryWriter(f"runs/{file_name}/result/{p.timestamp}")
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
writer1.file_writer.add_summary(exp)
writer1.file_writer.add_summary(ssi)
writer1.file_writer.add_summary(sei)

print('\n\n----------')
print(p.timestamp)

avg_test_acc_roi = sum(roi_bastacc_test_k_fload)/10
avg_test_loss_roi = sum(roi_bastloss_test_k_fload)/10

avg_test_acc_global = sum(global_bastacc_test_k_fload)/10
avg_test_loss_global = sum(global_bastloss_test_k_fload)/10

avg_test_acc = sum(bastacc_test_k_fload)/10
avg_test_loss = sum(bastloss_test_k_fload)/10

print('\navg_roi_bastacc_test_k_fload:',roi_bastacc_test_k_fload,'\navg:',avg_test_acc_roi)
print('\navg_roi_bastloss_test_k_fload:',roi_bastloss_test_k_fload,'\navg:',avg_test_loss_roi)

print('\navg_global_bastacc_test_k_fload:',global_bastacc_test_k_fload,'\navg:',avg_test_acc_global)
print('\navg_global_bastloss_test_k_fload:',global_bastloss_test_k_fload,'\navg:',avg_test_loss_global)

print('\navg_bastacc_test_k_fload:',bastacc_test_k_fload,'\navg:',avg_test_acc)
print('\navg_bastloss_test_k_fload:',bastloss_test_k_fload,'\navg:',avg_test_loss)

writer1.add_scalar('roi/BestAcc/validation', avg_test_acc_roi)
writer1.add_scalar('roi/BestLoss/validation', avg_test_loss_roi)

writer1.add_scalar('global/BestAcc/validation', avg_test_acc_global)
writer1.add_scalar('global/BestLoss/validation', avg_test_loss_global)

writer1.add_scalar('BestAcc/validation', avg_test_acc)
writer1.add_scalar('BestLoss/validation', avg_test_loss)

#long running
endtime = datetime.datetime.now()
totaltime = endtime - starttime
print('total time: ', totaltime)
print('total time (seconds): ', totaltime.seconds)

writer1.flush()
writer1.close()
