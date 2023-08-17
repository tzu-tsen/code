# %%
# 自建立的 Dataset 和 Model
import dataset.database as Dataset
from package import model1

import package.confusion_matrix as ConfusionMatrix
import package.parameter_affectnet as parameter
import package.fun as fun

import datetime
import os

import torch
import torch.nn as nn

import numpy as np
from pathlib import Path
# from tqdm import tqdm_notebook as tqdm
from torch.utils.tensorboard import SummaryWriter


# %% [markdown]
p = parameter.Parameter()
# gpu
cuda_drivers = 0
cuda_drivers = str(input(f"Enter cuda (default:{cuda_drivers}): ") or cuda_drivers)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_drivers

# 類別數量
p.num_class = int(input(f"Enter num_class (default:{p.num_class}): ") or p.num_class)

# 資料庫路徑
p.path_train = str(input(f"Enter path_train (default:{p.path_train}): ") or p.path_train)
p.path_test = str(input(f"Enter path_test(val) (default:{p.path_test}): ") or p.path_test)


# %%
# 訓練的模型
model = model1.ResNet(num_classes=p.num_class, n_head=p.n_head, resnet_cbam=p.num_resnet_cbam_roi)

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
writer = SummaryWriter(f"runs/{file_name}_{p.num_class}/{p.timestamp}")

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

        'Loss/train':0,
        'Loss/validation':0,        

        'BestLoss/train':0,
        'BestLoss/validation':0,

        'LR':0
    })
writer.file_writer.add_summary(exp)                 
writer.file_writer.add_summary(ssi)                 
writer.file_writer.add_summary(sei) 
# %%
criterion = nn.CrossEntropyLoss(label_smoothing=p.ls).cuda() # 交叉熵损失函数

# %%

# %% [markdown]
#  ### 訓練、測試、驗證資料

# %%
# roi
train_dataset_roi = Dataset.Dataset(root_path=p.path_train,transform=p.transform_train,class_number=p.num_class, roi_transform=p.roi_transform_train_no_padding, data_type=parameter.EnumType.roi)
train_lbls_roi, train_images_roi, train_number_roi = train_dataset_roi.get_split()

train_number_sum_roi = sum(train_number_roi)
class_weights_roi = [(1/(number/train_number_sum_roi)) for number in train_number_roi]
sample_weights_roi = [class_weights_roi[i] for i in train_lbls_roi]
sampler_roi = torch.utils.data.sampler.WeightedRandomSampler(sample_weights_roi, num_samples=p.num_samples_roi, replacement=True)
train_loader = torch.utils.data.DataLoader(train_dataset_roi, batch_size=p.batch_size_roi,num_workers=p.num_workers,pin_memory=p.pin_memory,sampler=sampler_roi) 

# test, AffectNet only val, val is test
val_dataset_roi = Dataset.Dataset(root_path=p.path_test,transform=p.transform_val,class_number=p.num_class, 
                              roi_transform=p.roi_transform_val_no_padding, data_type=parameter.EnumType.roi)
val_loader = torch.utils.data.DataLoader(val_dataset_roi, batch_size=p.batch_size_roi,shuffle=p.shuffle,num_workers=p.num_workers)    

print("train每個類別數量:",train_number_roi)
print("train總數量：", train_number_sum_roi)
print("class weighets:",class_weights_roi)
print("\ntrain數量：")
print(len(train_loader))
print("\nval數量：")
print(len(val_loader))
# print("\ntest數量：")
# print(len(test_loader))

# %%
best_acc_train, best_acc_val = 0, 0
best_loss_train, best_loss_val = np.Inf, np.Inf # 跟踪驗證損失的變化 track change in validation loss

optimizer = torch.optim.SGD(model.parameters(), p.lr_roi, weight_decay=p.weight_decay, momentum=p.momentum)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=p.milestones_roi, gamma=p.gamma_roi)

starttime = datetime.datetime.now()
for epoch in range(p.num_epochs):
    print('running epoch: {}'.format(epoch))

    # 學習率
    optlr = optimizer.param_groups[0]['lr']
    print('lr:', optlr)
    writer.add_scalar('LR', optlr, epoch)

    # train for one epoch    
    train_acc, train_loss, train_class_number, train_target_lists, train_predict_lists = fun.train(train_loader, model, criterion, optimizer, hyperparameter=p.beta_roi, num_class=p.num_class)
    
    # evaluate on validation set                       
    val_acc, val_loss, val_target_lists, val_predict_lists = fun.validate(val_loader, model, criterion, hyperparameter=p.beta_roi)

    scheduler.step()      
    if val_acc > best_acc_val:        
        torch.save(model, path_model)
        val_figure_cm = ConfusionMatrix.figure_confusion_matrix(val_target_lists, val_predict_lists, p.class_names, p.cm_normalize) 
        writer.add_figure('Confusion Matrix val best', val_figure_cm, epoch)

    best_acc_train = max(best_acc_train, train_acc)
    best_loss_train = min(best_loss_train, train_loss)

    best_acc_val = max(best_acc_val, val_acc)
    best_loss_val = min(best_loss_val, val_loss)

    # confusion_maatrix
    train_figure_cm = ConfusionMatrix.figure_confusion_matrix(train_target_lists, train_predict_lists, p.class_names, p.cm_normalize) 
    val_figure_cm = ConfusionMatrix.figure_confusion_matrix(val_target_lists, val_predict_lists, p.class_names,p.cm_normalize) 
    writer.add_figure('Confusion Matrix train', train_figure_cm, epoch)
    writer.add_figure('Confusion Matrix val', val_figure_cm, epoch)    

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

# BestAcc
writer.add_scalar('BestAcc/train', best_acc_train)
writer.add_scalar('BestAcc/validation', best_acc_val)

writer.add_scalar('BestLoss/train', best_loss_train)
writer.add_scalar('BestLoss/validation', best_loss_val)

#long running
endtime = datetime.datetime.now()
totaltime = endtime - starttime
print('total time: ', totaltime)
print('total time (seconds): ', totaltime.seconds)

# %%
writer.flush()
writer.close()


