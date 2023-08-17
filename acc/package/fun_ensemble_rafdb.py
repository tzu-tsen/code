import torch
import numpy as np

def validate(val_loader, model_roi, model_global, criterion, beta=[1,1]):
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
            target, data, is_roi, eye_data, nose_data, mouth_data = all_data
            target_lists+=target.tolist()
            target, data, eye_data, nose_data, mouth_data = target.cuda(), data.cuda(), eye_data.cuda(), nose_data.cuda(), mouth_data.cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            if is_roi:
                output1, output_global, output_roi = model_roi(data, eye_data, nose_data, mouth_data) 
                output2 = model_global(data)
                
                output = beta[0]*output1+beta[1]*output2                
            else:
                output = model_global(data)                      
            
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

    return acc, running_loss, target_lists, predict_lists, bingo_cnt.float(), sample_cnt

def validatev2(val_loader, model_roi, model_global, criterion):
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
            target, data, is_roi, eye_data, nose_data, mouth_data = all_data
            target_lists+=target.tolist()
            target, data, eye_data, nose_data, mouth_data = target.cuda(), data.cuda(), eye_data.cuda(), nose_data.cuda(), mouth_data.cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            if is_roi:
                output, output_global, output_roi = model_roi(data, eye_data, nose_data, mouth_data)                   
            else:
                output = model_global(data)                      
            
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

    return acc, running_loss, target_lists, predict_lists, bingo_cnt.float(), sample_cnt

def validate_roi(val_loader, model_roi, criterion):
    with torch.no_grad():        
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        target_lists, predict_lists = [], []
            
        # switch to evaluate mode
        model_roi.eval()        
        for step, all_data in enumerate(val_loader):
            target, data, is_roi, eye_data, nose_data, mouth_data = all_data
            target_lists+=target.tolist()
            target, data, eye_data, nose_data, mouth_data = target.cuda(), data.cuda(), eye_data.cuda(), nose_data.cuda(), mouth_data.cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            output, output_global, output_roi = model_roi(data, eye_data, nose_data, mouth_data) 
                
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

    return acc, running_loss, target_lists, predict_lists, bingo_cnt.float(), sample_cnt

def validate_global(val_loader, model_global, criterion):
    with torch.no_grad():        
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        target_lists, predict_lists = [], []
            
        # switch to evaluate mode        
        model_global.eval()
        for step, all_data in enumerate(val_loader):
            target, data, is_roi, eye_data, nose_data, mouth_data = all_data
            target_lists+=target.tolist()
            target, data, eye_data, nose_data, mouth_data = target.cuda(), data.cuda(), eye_data.cuda(), nose_data.cuda(), mouth_data.cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model_global(data)                      
            
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

    return acc, running_loss, target_lists, predict_lists, bingo_cnt.float(), sample_cnt