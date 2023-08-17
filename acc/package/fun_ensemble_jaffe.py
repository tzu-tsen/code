import torch
import numpy as np

def del_tensor_ele_n(arr, index, n,start=1):
    """
    arr: 输入tensor
    index: 需要删除位置的索引
    n: 从index开始，需要删除的行数
    """
    if index < len(arr[0:]):
        arr1 = arr[start:index]
        arr2 = arr[index+n:]
        arr = torch.cat((arr1,arr2),dim=0).detach().cpu().numpy()
    else:
        arr = (arr[start:]).detach().cpu().numpy()
    return torch.tensor([arr]).cuda()

def validate(val_loader, model_roi, model_global, criterion, beta):
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
                output_c, output_global, output_roi = model_roi(data, eye_data, nose_data, mouth_data) 
                output1 = del_tensor_ele_n(output_c[0],7,1) # 删除tensor1中索引为7的元素                         
                output2 = del_tensor_ele_n(model_global(data)[0],7,1) # 删除tensor1中索引为7的元素                         
                
                output = beta[0]*output1+beta[1]*output2                
            else:
                # output = model_global(data)      
                output = del_tensor_ele_n(model_global(data)[0],7,1) # 删除tensor1中索引为7的元素 
            
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
            output_c, output_global, output_roi = model_roi(data, eye_data, nose_data, mouth_data) 
            output = del_tensor_ele_n(output_c[0],7,1) # 删除tensor1中索引为7的元素                         
                
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
            output = del_tensor_ele_n(model_global(data)[0],7,1) # 删除tensor1中索引为7的元素 
            
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