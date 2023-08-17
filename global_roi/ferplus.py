# %%
# 自建立的 Dataset 和 Model
import dataset.FERPlus as Dataset
from dataset.my_enum import *
import package.model1 as Model
import package.parameter_ferplus as parameter
import package.fun_ferplus as fun

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
cuda_drivers = str(input(f"Enter cuda (default:{cuda_drivers}): ") or cuda_drivers)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_drivers

# 資料庫路徑
p.path = str(input(f"Enter path (default:{p.path}): ") or p.path)

# %%
# 訓練的模型
model = Model.ResNet(num_classes=p.num_class, n_head=p.n_head, resnet_cbam=p.num_resnet_cbam_roi)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.cuda()

# 檔案名稱
path_file = os.path.realpath(__file__)
file_name = Path(path_file).stem
print('file_name:', file_name)

# model 儲存路徑
path_model_dir = os.path.join("model",file_name)
path_model = os.path.join(path_model_dir, f'{p.timestamp}.path')
if not os.path.isdir(path_model_dir):
    os.makedirs(path_model_dir)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter(f"runs/{file_name}/{p.timestamp}")
from torch.utils.tensorboard.summary import hparams
exp, ssi, sei = hparams(
    {
        'hp_all_loss': p.beta_roi[0],
        'hp_whole_loss': p.beta_roi[1],
        'hp_roi_loss': p.beta_roi[2],
        'epoch':p.num_epochs,
        'batch_size':p.batch_size_roi,
		'num_workers':p.num_workers,
        'num_class':p.num_class, 
		'num_samples':p.num_samples_roi,
        'num_cbam':p.num_resnet_cbam_roi,
		'optimizer':'SGD',
		'lr':p.lr_roi,
		'rsize':p.roi_size, 
		'weight_decay':p.weight_decay,
        'momentum':p.momentum,        
        'label_smoothing':p.ls,
        'lr_scheduler':'StepLR',
        'lr_scheduler/milestones':str(p.milestones_roi), 
		'lr_scheduler/gamma':p.gamma_roi       
    },
    metric_dict={
        'Acc/train':0,
        'Acc/validation':0,        

        'BestAcc/train':0,
        'BestAcc/validation':0,
        'BestAcc/test':0,

        'Loss/train':0,
        'Loss/validation':0,        

        'BestLoss/train':0,
        'BestLoss/validation':0,
        'BestLoss/test':0, 

        'LR':0
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
train_dataset = Dataset.Dataset(root_path=p.path,run=parameter.EnumRun.train, transform=p.transform_train, roi_transform=p.roi_transform_train_no_padding,  data_type=parameter.EnumType.roi)
train_lbls, train_images, train_number = train_dataset.get_split()
train_number_sum = len(train_images)
class_weights = [(1/(number/train_number_sum)) for number in train_number]
print("train_number_sum:",train_number_sum)
print("train_number:",train_number)
print("class_weights:",class_weights)

sample_weights = []
for i in  train_lbls:        
    i = np.array(i)
    indexs = np.where(i>0)[0]
    sample_weight=0
    for index in indexs:
        sample_weight+=class_weights[index]
    sample_weight = sample_weight/len(indexs)
    sample_weights.append(sample_weight)
sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, num_samples=p.num_samples_roi, replacement=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=p.batch_size_roi,num_workers=p.num_workers,pin_memory=p.pin_memory,sampler=sampler) 

# val
# test, AffectNet only val, val is test
val_dataset = Dataset.Dataset(root_path=p.path, run=parameter.EnumRun.val, transform=p.transform_val, 
                              roi_transform=p.roi_transform_val_no_padding, data_type=parameter.EnumType.roi)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=p.batch_size_roi,shuffle=p.shuffle,num_workers=p.num_workers)    

# test
test_dataset = Dataset.Dataset(root_path=p.path, run=parameter.EnumRun.test, transform=p.transform_val, 
                              roi_transform=p.roi_transform_val_no_padding, data_type=parameter.EnumType.roi)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=p.batch_size_roi,shuffle=p.shuffle,num_workers=p.num_workers)    

print("\ntrain數量：")
print(len(train_loader))
print("\nval數量：")
print(len(val_loader))
print("\ntest數量：")
print(len(test_loader))

# %%
best_acc_train, best_acc_val, best_acc_test = 0, 0, 0
best_loss_train, best_loss_val, best_loss_test = np.Inf, np.Inf, np.Inf # 跟踪驗證損失的變化 track change in validation loss

optimizer = torch.optim.SGD(model.parameters(), p.lr_roi, weight_decay=p.weight_decay, momentum=p.momentum)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=p.milestones_roi, gamma=p.gamma_roi)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.6)

starttime = datetime.datetime.now()
for epoch in range(p.num_epochs):
    print('running epoch: {}'.format(epoch))

    # 學習率
    optlr = optimizer.param_groups[0]['lr']
    print('lr:', optlr)
    writer.add_scalar('LR', optlr, epoch)

    # train for one epoch    
    train_acc, train_loss, train_class_number, train_target_lists, train_predict_lists = fun.train(train_loader, model, criterion, optimizer, hyperparameter=p.beta_roi)
    
    # evaluate on validation set                       
    val_acc, val_loss, val_target_lists, val_predict_lists = fun.validate(val_loader, model, criterion, hyperparameter=p.beta_roi)

    scheduler.step()      
    if val_acc > best_acc_val:        
        torch.save(model, path_model)

    best_acc_train = max(best_acc_train, train_acc)
    best_loss_train = min(best_loss_train, train_loss)

    best_acc_val = max(best_acc_val, val_acc)
    best_loss_val = min(best_loss_val, val_loss)

    # Acc
    writer.add_scalar('Acc/train', train_acc, epoch)
    writer.add_scalar('Acc/validation', val_acc, epoch)    
    writer.add_scalars('Acces', {'train':train_acc,'valid':val_acc}, epoch)        

    # Loss  
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)         
    writer.add_scalars('Losses', {'train':train_loss,'valid':val_loss}, epoch)        

    # print training/validation statistics      
    print('Train Class Number:',train_class_number)    
    print('Training ',train_acc,train_loss)  
    print('validation ',val_acc,val_loss)  
    print('Current best accuracy: ', best_acc_val)

# 測試 model
best_acc_test, best_loss_test, test_target_lists, test_predict_lists = fun.test(test_loader, criterion, path_model, hyperparameter=p.beta_roi)

# BestAcc
writer.add_scalar('BestAcc/train', best_acc_train)
writer.add_scalar('BestAcc/validation', best_acc_val)
writer.add_scalar('BestAcc/test', best_acc_test)

writer.add_scalar('BestLoss/train', best_loss_train)
writer.add_scalar('BestLoss/validation', best_loss_val)
writer.add_scalar('BestLoss/test', best_loss_test)

#long running
endtime = datetime.datetime.now()
totaltime = endtime - starttime
print('total time: ', totaltime)
print('total time (seconds): ', totaltime.seconds)

# %%
writer.flush()
writer.close()