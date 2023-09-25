import torch
import numpy as np

def test(val_loader, model_roi, model_global, criterion, beta=[1,1]):
    with torch.no_grad():        
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        target_lists, predict_lists = [], []
            
        # switch to evaluate mode
        model_roi.eval()
        model_global.eval()
        for step, all_data in enumerate(val_loader):
            target, datas, is_roi, eye_datas, nose_datas, mouth_datas = all_data
            target_lists+=target.tolist()
            target, data, eye_data, nose_data, mouth_data = target.cuda(), datas["test"].cuda(), eye_datas["test"].cuda(), nose_datas["test"].cuda(), mouth_datas["test"].cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            # if is_roi:
            #     output1,output_whole,output_roi = model_roi(data, eye_data, nose_data, mouth_data)      
            #     output2 = model_global(data)
                
            #     output = beta[0]*output1+beta[1]*output2                
            # else:
            #     output = model_global(data) 
            
            # 不判斷 is_roi，因為都有 roi
            output1,output_whole,output_roi = model_roi(data, eye_data, nose_data, mouth_data)      
            output2 = model_global(data)
                
            output = beta[0]*output1+beta[1]*output2                
            
            # update average validation loss 
            loss = criterion(output, target)
        
            # measure accuracy and record loss
            running_loss += loss
            iter_cnt+=1
            _, predicts = torch.max(output, 1)
            correct_num  = torch.eq(predicts,target)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += output.size(0)

            # 將preds預測結果detach出來，並轉成numpy格式               
            predict_lists+=predicts.view(-1).detach().cpu().numpy().tolist()        
        
        running_loss = running_loss/iter_cnt 
        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)

    return acc, running_loss, target_lists, predict_lists

def test_roi(val_loader, model, criterion):
    with torch.no_grad():        
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        target_lists, predict_lists = [], []
            
        # switch to evaluate mode
        model.eval()
        for step, all_data in enumerate(val_loader):
            target, datas, is_roi, eye_datas, nose_datas, mouth_datas = all_data
            target_lists+=target.tolist()
            target, data, eye_data, nose_data, mouth_data = target.cuda(), datas["test"].cuda(), eye_datas["test"].cuda(), nose_datas["test"].cuda(), mouth_datas["test"].cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            output,output_whole,output_roi = model(data, eye_data, nose_data, mouth_data)      

            # update average validation loss 
            loss = criterion(output, target)
        
            # measure accuracy and record loss
            running_loss += loss
            iter_cnt+=1
            _, predicts = torch.max(output, 1)
            correct_num  = torch.eq(predicts,target)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += output.size(0)

            # 將preds預測結果detach出來，並轉成numpy格式               
            predict_lists+=predicts.view(-1).detach().cpu().numpy().tolist()        
        
        running_loss = running_loss/iter_cnt 
        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)

    return acc, running_loss, target_lists, predict_lists

def test_global(val_loader, model, criterion):
    with torch.no_grad():        
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        target_lists, predict_lists = [], []
            
        # switch to evaluate mode
        model.eval()
        for step, all_data in enumerate(val_loader):
            target, datas, is_roi, eye_datas, nose_datas, mouth_datas = all_data
            target_lists+=target.tolist()
            target, data = target.cuda(), datas["test"].cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)      

            # update average validation loss 
            loss = criterion(output, target)

            # measure accuracy and record loss
            running_loss += loss
            iter_cnt+=1
            _, predicts = torch.max(output, 1)
            correct_num  = torch.eq(predicts,target)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += output.size(0)

            # 將preds預測結果detach出來，並轉成numpy格式               
            predict_lists+=predicts.view(-1).detach().cpu().numpy().tolist()        
        
        running_loss = running_loss/iter_cnt 
        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)

    return acc, running_loss, target_lists, predict_lists