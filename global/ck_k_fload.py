# %%
# 自建立的 Dataset 和 Model
import dataset.ck_no_roi as Dataset
import package.parameter_ck as parameter
import os

import torch.nn as nn

import numpy as np

from sklearn.model_selection import KFold

p = parameter.Parameter(batch_size_whole=500)

# %%
criterion = nn.CrossEntropyLoss(label_smoothing=p.ls).cuda() # 交叉熵损失函数
dataset = Dataset.Dataset(root_path=p.path_test,transform_train=p.transform_train, transform_test=p.transform_val,class_number=p.num_class, 
                              data_type=parameter.EnumType.original)

kf = KFold(n_splits=10, shuffle=True)
# kf.get_n_splits(dataset)
print("kf:",kf)

for i, (train_index, test_index) in enumerate(kf.split(dataset)):
    print(f"Fold {i}:")

    '-- 儲存 --'
    path = os.path.join("k_fload",f'{i}')
    if not os.path.isdir(path):
        os.makedirs(path)

    np.save(os.path.join(path,'train_index'), train_index)
    np.save(os.path.join(path,'test_index'), test_index)

    '-- k-fload dataset --'
    number_class = [0,0,0,0,0,0,0,0]
    class_weights = [0,0,0,0,0,0,0,0]
    
    # class number
    number_sum = len(train_index)

    for ts in train_index:
        label = dataset[ts][0]        
        number_class[label] = number_class[label]+1
    print("number_class:",number_class)
    
    number_sum = len(train_index)
    class_weights = [(1/(number/number_sum)) for number in number_class]
    print("class_weights:",class_weights)

    sample_weights = []
    for ts in train_index:
        label = dataset[ts][0]        
        sample_weights.append(class_weights[label])

    np.save(os.path.join(path,'number_class'), number_class)
    np.save(os.path.join(path,'class_weights'), class_weights)
    np.save(os.path.join(path,'sample_weights'), sample_weights)

    # # print(sample_weights)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, num_samples=p.num_samples_whole, replacement=True)
    
    # # 只取train_index 的資料
    # new_dataset = torch.utils.data.Subset(dataset, train_index)
    
    # train_loader = torch.utils.data.DataLoader(new_dataset, batch_size=p.batch_size_whole,num_workers=p.num_workers,pin_memory=p.pin_memory,sampler=sampler) 

    # test_sampler = torch.utils.data.SubsetRandomSampler(test_index)
    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=p.batch_size_whole,sampler=test_sampler,num_workers=p.num_workers)    

    print("\ntrain數量：")
    print(len(train_index))
    print("\ntest數量：")
    print(len(test_index))    