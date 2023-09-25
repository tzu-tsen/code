# %%
# 自建立的 Dataset 和 Model
import dataset.ck_no_roi as Dataset
import package.resnet_cbam as Model
import package.confusion_matrix as ConfusionMatrix
import package.parameter_ck as parameter
import package.fun_ck as fun

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

# 資料庫路徑
p.path_test = str(input(f"Enter path_test (default:{p.path_test}): ") or p.path_test)

# %%

# 檔案名稱
path_file = os.path.realpath(__file__)
file_name = Path(path_file).stem
print('file_name:', file_name)

# model 儲存目錄
path_model_dir = os.path.join("model", file_name, f'{p.timestamp}')      
if not os.path.isdir(path_model_dir):
    os.makedirs(path_model_dir)
    
# %%
criterion = nn.CrossEntropyLoss(label_smoothing=p.ls).cuda() # 交叉熵损失函数

dataset = Dataset.Dataset(root_path=p.path_test,transform_train=p.transform_train, transform_test=p.transform_val,class_number=p.num_class, 
                              data_type=parameter.EnumType.original)

bast_train_k_fload = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
bast_test_k_fload  = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
loss_train_k_fload = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
loss_test_k_fload  = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

starttime = datetime.datetime.now()

for i in range(10):
    print("===================================")
    print(f"Fold {i}:")    

    # 訓練的模型
    model = Model.Resnet(num_classes=p.num_class,pretrained=True)
    model.cuda()

    '-- k-fload dataset --'
    path = os.path.join("k_fload",f'{i}')
    
    train_index = np.load(os.path.join(path, "train_index.npy"))
    test_index = np.load(os.path.join(path, "test_index.npy"))

    number_class = np.load(os.path.join(path, "number_class.npy"))
    # print(number_class)
    # class_weights = np.load(os.path.join(path, "class_weights.npy"))
    # print(class_weights)
    sample_weights = np.load(os.path.join(path, "sample_weights.npy"))
    # print(sample_weights)
    
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, num_samples=p.num_samples_whole, replacement=True)
    # 只取train_index 的資料
    train_dataset = torch.utils.data.Subset(dataset, train_index)    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=p.batch_size_whole,num_workers=p.num_workers,pin_memory=p.pin_memory,sampler=sampler) 

    test_sampler = torch.utils.data.SubsetRandomSampler(test_index)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=p.batch_size_whole,sampler=test_sampler,num_workers=p.num_workers)    

    print("\ntrain數量：",len(train_index))    
    print("test數量：",len(test_index))
    
    '-- 儲存 --'
    # model 儲存路徑    
    path_model = os.path.join(path_model_dir, f'{i}.path')

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(f"runs/{file_name}/{p.timestamp}/{i}")
    
    exp, ssi, sei = hparams(
        {
            'name': file_name,        
            'epoch':p.num_epochs,
            'batch_size':p.batch_size_whole,
            'num_workers':p.num_workers,
            'num_class':p.num_class, 		
            'num_samples':p.num_samples_whole,
            'optimizer':'SGD',
            'lr':p.lr_whole,
            'rsize':p.roi_size, 
            'weight_decay':p.weight_decay,
            'momentum':p.momentum,        
            'label_smoothing':p.ls,
            'lr_scheduler':'MultiStepLR',
            'lr_scheduler/milestones':str(p.milestones_whole), 
            'lr_scheduler/gamma':p.gamma_whole,
            'model':'resnet_cbam'        
        },
        metric_dict={
            'Acc/train':0,
            'Acc/test':0,        

            'BestAcc/train':0,            
            'BestAcc/test':0,

            'Loss/train':0,
            'Loss/test':0,        

            'BestLoss/train':0,
            'BestLoss/test':0,           

            'LR':0
        })
    writer.file_writer.add_summary(exp)                 
    writer.file_writer.add_summary(ssi)                 
    writer.file_writer.add_summary(sei)

    # %%
    '--- train&test ---'
    best_acc_train, best_acc_test = 0, 0
    best_loss_train, best_loss_test = np.Inf, np.Inf # 跟踪驗證損失的變化 track change in test loss

    optimizer = torch.optim.SGD(model.parameters(), p.lr_whole, weight_decay=p.weight_decay, momentum=p.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=p.milestones_whole, gamma=p.gamma_whole)
    
    for epoch in range(p.num_epochs):
        print('running epoch: {}'.format(epoch))

        # 學習率
        optlr = optimizer.param_groups[0]['lr']
        print('lr:', optlr)
        writer.add_scalar('LR', optlr, epoch)

        # train for one epoch    
        train_acc, train_loss, train_class_number, train_target_lists, train_predict_lists = fun.train(train_loader, model, criterion, optimizer, num_class=p.num_class)
        
        # etestuate on test set                       
        test_acc, test_loss, test_target_lists, test_predict_lists = fun.test(test_loader, model, criterion)

        scheduler.step()      
        if test_acc > best_acc_test:        
            torch.save(model, path_model)
            test_figure_cm = ConfusionMatrix.figure_confusion_matrix(test_target_lists, test_predict_lists, p.class_names, p.cm_normalize) 
            writer.add_figure('Confusion Matrix test best', test_figure_cm, epoch)

        best_acc_train = max(best_acc_train, train_acc)
        best_loss_train = min(best_loss_train, train_loss)

        best_acc_test = max(best_acc_test, test_acc)
        best_loss_test = min(best_loss_test, test_loss)

        # confusion_maatrix
        train_figure_cm = ConfusionMatrix.figure_confusion_matrix(train_target_lists, train_predict_lists, p.class_names, p.cm_normalize) 
        test_figure_cm = ConfusionMatrix.figure_confusion_matrix(test_target_lists, test_predict_lists, p.class_names,p.cm_normalize) 
        writer.add_figure('Confusion Matrix train', train_figure_cm, epoch)
        writer.add_figure('Confusion Matrix test', test_figure_cm, epoch)    

        # Acc
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/test', test_acc, epoch)
        # writer.add_scalar('Acc/test', test_acc, epoch)
        writer.add_scalars('Acces', {'train':train_acc,'test':test_acc}, epoch)        

        # Loss  
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        # writer.add_scalar('Loss/test', test_loss, epoch)        
        writer.add_scalars('Losses', {'train':train_loss,'test':test_loss}, epoch)        

        # print training/test statistics      
        print('Train Class Number:',train_class_number)    
        print(f'{i} Training ',train_acc,train_loss)  
        print(f'{i} test ',test_acc,test_loss)  
        print(f'{i} Current best accuracy: ', best_acc_test)

    # BestAcc
    writer.add_scalar('BestAcc/train', best_acc_train)
    writer.add_scalar('BestAcc/test', best_acc_test)    
    bast_train_k_fload[i] = best_acc_train
    bast_test_k_fload[i] = best_acc_test
    
    writer.add_scalar('BestLoss/train', best_loss_train)
    writer.add_scalar('BestLoss/test', best_loss_test)
    loss_train_k_fload[i] = best_loss_train
    loss_test_k_fload[i] = best_loss_test
    
    # %%
    writer.flush()
    writer.close()

# avg
# Writer will output to ./runs/ directory by default
# writer1 = SummaryWriter(f"runs/{file_name}/result/{p.timestamp}")

# exp, ssi, sei = hparams(
#     {
#         'name': file_name,        
#         'epoch':p.num_epochs,
#         'batch_size':p.batch_size_whole,
#         'num_workers':p.num_workers,
#         'num_class':p.num_class, 		
#         'num_samples':p.num_samples_whole,
#         'optimizer':'SGD',
#         'lr':p.lr_whole,
#         'rsize':p.roi_size, 
#         'weight_decay':p.weight_decay,
#         'momentum':p.momentum,        
#         'label_smoothing':p.ls,
#         'lr_scheduler':'MultiStepLR',
#         'lr_scheduler/milestones':str(p.milestones_whole), 
#         'lr_scheduler/gamma':p.gamma_whole,
#         'model':'resnet_cbam'        
#     },
#     metric_dict={
#         'Acc/train':0,
#         'Acc/test':0,       

#         'Loss/train':0,
#         'Loss/test':0, 
         
#     })
# writer1.file_writer.add_summary(exp)                 
# writer1.file_writer.add_summary(ssi)                 
# writer1.file_writer.add_summary(sei)

print('\n\n----------')
print(p.timestamp)
avg_train_acc = sum(bast_train_k_fload)/10
avg_test_acc = sum(bast_test_k_fload)/10
avg_train_loss = sum(loss_train_k_fload)/10
avg_test_loss = sum(loss_test_k_fload)/10
print('\navg_train_acc:',bast_train_k_fload,'\navg:',avg_train_acc)
print('\navg_test_acc:',bast_test_k_fload,'\navg:',avg_test_acc)
print('\navg_train_loss:',loss_train_k_fload,'\navg:',avg_train_loss)
print('\navg_test_loss:',loss_test_k_fload,'\navg:',avg_test_loss)

# writer1.add_scalar('AvgAcc/train', avg_train_acc)
# writer1.add_scalar('AvgAcc/test', avg_test_acc)

# writer1.add_scalar('AvgLoss/train', avg_train_loss)
# writer1.add_scalar('AvgLoss/test', avg_test_loss)

#long running
endtime = datetime.datetime.now()
totaltime = endtime - starttime
print('total time: ', totaltime)
print('total time (seconds): ', totaltime.seconds)

# %%
# writer1.flush()
# writer1.close()