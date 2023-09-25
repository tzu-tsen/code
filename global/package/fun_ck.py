import torch
import numpy as np

def train(train_loader, model, criterion, optimizer, num_class=8):
    loss_sum = 0.0
    correct_sum = 0
    iter_cnt = 0
    target_lists, predict_lists = [], []    
    
    # train_loss = 0.0
    sample_cnt=0

    # switch to train mode
    model.train()
    for step, all_data in enumerate(train_loader):    
        iter_cnt += 1              

        # 梯度清零
        optimizer.zero_grad()
                
        target, datas = all_data        
        target_lists+=target.tolist()
        target, data = target.cuda(), datas['train'].cuda()
            
        output = model(data)           
        loss = criterion(output, target)
        
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # 執行單個優化步驟（參數更新）
        optimizer.step()
        
        loss_sum += loss        
        _, predicts = torch.max(output, 1)  
        # predicts_lists.append(predicts)
        correct_sum += torch.eq(predicts,target).sum().cpu()
        # train_loss += loss.item()*output.size(0)
        sample_cnt += output.size(0)

        # 將preds預測結果detach出來，並轉成numpy格式               
        predict_lists+=predicts.view(-1).detach().cpu().numpy().tolist()
        
    acc = correct_sum.float() / float(sample_cnt)    
    acc = np.around(acc.numpy(),4)
    
    running_loss = loss_sum/iter_cnt
    running_loss = running_loss.cpu().detach().numpy()

    target_lists_number = [target_lists.count(i) for i in range(num_class)]
   
    return acc, running_loss, target_lists_number, target_lists, predict_lists

def test(val_loader, model, criterion):
    with torch.no_grad():        
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        target_lists, predict_lists = [], []
            
        # switch to evaluate mode
        model.eval()
        for step, all_data in enumerate(val_loader):
            target, datas = all_data
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