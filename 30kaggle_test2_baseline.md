# Kaggle竞赛baseline
## 1.导入包
```python
# 此处只给出了几个重点使用的包
import numpy as np
import pandas as pd
import torch
from torch import nn
# 自定义数据集
from torch.utils.data import Dataset, DataLoader
# 显示循环的进度条
from tqdm import tqdm
# torchvision，该包主由3个子包组成：torchvision.datasets、torchvision.models、torchvision.transforms
# torchvision.models中包含alexnet、densenet、inception、resnet、squeezenet、vgg等常用的网络结构，并且提供了预训练模型，可以通过简单调用来读取网络结构和预训练模型。
# 例子：model = models.resnet50(pretrained=True)
import torchvision.models as models
```

## 2.读数据
从csv文件中
```python
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# 查看训练集的前五行（看看大概的格式）
train_data.head(5)
# 如果训练数据列数太多，选择性拿出来几个看看
print(train_data.iloc[0:4, [0, 1]])
```

## 3.整理标签
如果标签是str，转成数字进行训练
```python
# 先取出无重复的label，然后排序
# 计算一下标签的个数
data_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(data_labels)
print(n_classes)

# 把label转成对应的数字，方便训练
class_to_num = dict(zip(data_labels, range(n_classes)))
# 再转换回来，方便最后预测的时候使用
num_to_class = {v : k for k, v in class_to_num.items()}
```

## 4.自定义Dataset类进行数据读取和初始化
先把原始数据转变成 torch.utils.data.Dataset类。随后再把得到的torch.utils.data.Dataset类当作一个参数传递给torch.utils.data.DataLoader类，得到一个数据加载器，这个数据加载器每次可以返回一个 Batch 的数据供模型训练使用。这一过程通常可以让我们把一张原始img通过标准化、resize等操作转变成我们需要的 [B,C,H,W] 形状的 Tensor。
### 4.1定义Dataset
```python
class some_data(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        # 统一图片尺寸（原始数据中的img尺寸可能不一样）
        self.resize_height = resize_height
        self.resize_width = resize_width
        
        self.file_path = file_path
        self.mode = mode
        
        # 读取csv文件
        # header设为None是说这个表没有标题
        self.data_info = pd.read_csv(csv_path, header=None)  
        # 设为None之后，会为其加上索引，所以计算长度要-1
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))
        
        # 分成训练、验证、测试三种不同的模式
        if mode == 'train':
            #self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            self.train_image = np.asarray(self.data_info.iloc[1:self.train_len, 0])
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[1:self.train_len, 0])
            self.valid_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image
            
        self.real_len = len(self.image_arr)
        
        print('Finished reading the {} set of some_data Dataset ({} samples found)'.format(mode, self.real_len))
        
        
    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名,此处self.image_arr[0]='images/0.jpg'
        single_image_name = self.image_arr[index]         
        
        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)
        
         #设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
            # 随机裁剪图像，所得图像为原始面积的0.08到1之间，高宽比在3/4和4/3之间。
            # 然后，缩放图像以创建224 x 224的新图像
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip(),
            # 随机更改亮度，对比度和饱和度
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # 添加随机噪声
            transforms.ToTensor(),
            # 标准化图像的每个通道
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize(256),
                # 从图像中心裁切224x224大小的图片
                transforms.CenterCrop(224),
                transforms.ToTensor(), 
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        
        img_as_img = transform(img_as_img)
        
        if self.mode == 'test':
            return img_as_img  #测试集只需要返回图像
        else: #训练以及测试有效性
            # 得到图像的 string label
            label = self.label_arr[index]   #例子self.label_arr[0] = maclura_pomifera
            # number label
            number_label = class_to_num[label] #查阅字典  将类型转换为数字

            return img_as_img, number_label  #返回每一个index对应的图片数据和对应的label
    
    def __len__(self):
        return self.real_len
```

### 4.2获取数据集
```python
#设置文件路径并得到数据集
train_path = './train.csv'
test_path = './test.csv'

# csv文件中已经images的路径了，因此这里只到上一级目录
img_path = '/home/x/xx/xxx/'

train_dataset = some_data(train_path, img_path, mode='train')
val_dataset = some_data(train_path, img_path, mode='valid')
test_dataset = some_data(test_path, img_path, mode='test')
print(train_dataset)
print(val_dataset)
print(test_dataset)
```

### 4.3定义dataloader
```python
# 定义data loader
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=90,      
        shuffle=True,     #打开乱序  False
        num_workers=0
    )

val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=90,   
        shuffle=True,    #打开乱序  False
        num_workers=0
    )
test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=90, 
        shuffle=False,
        num_workers=0
    )
```

## 5.设置计算设备
```python
# GPU计算
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
print(device)
# 此处如果有且配置好cuda环境会打印cuda
```

## 6.模型设置
```python
# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False
# resnet50模型
def res_model(num_classes, feature_extract = False, use_pretrained=True):

    model_ft = models.densenet161(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft
```

## 7.设置超参数
```python
learning_rate = 1e-4   #1e-4
weight_decay = 1e-3
num_epoch = 50
beta = 1              #cutmix参数
model_path = './pre_res_model.ckpt' #保存中间模型数据，方便加载
```

## 8.训练
```python
# 初始化模型，并放在要使用的计算设备上
model = res_model(176)
model = model.to(device)
model.device = device

# 对于分类任务，使用cross-entropy衡量模型性能
criterion = nn.CrossEntropyLoss()

# 初始化optimizer，可以自己微调学习率等超参数
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

# 训练轮数
n_epochs = num_epoch

best_acc = 0.0
# 开始训练
for epoch in range(n_epochs):
    # ---------- Training ----------
    # 设置为训练模式
    model.train() 
    # 用于记录训练过程中的信息
    train_loss = []
    train_accs = []
    # 批量迭代训练集
    for batch in tqdm(train_loader):
        # 一个批量包含图像数据和对应的标签
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        # 前向传播数据，保证数据和模型都在同一设备上
        logits = model(imgs)
        # 计算cross-entropy损失
        # 无需在计算cross-entropy之前应用softmax因为他会自动进行
        loss = criterion(logits, labels)
        
        # 将之前的梯度清零
        optimizer.zero_grad()
        # 计算参数梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        
        # 计算当前batch的精度
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        # 记录损失和精度
        train_loss.append(loss.item())
        train_accs.append(acc)

    # 更新当前的平均损失和平均精度
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    
    
    # ---------- Validation ----------
    # 设备模型为验证模式
    model.eval()
    # 记录验证过程中的信息
    valid_loss = []
    valid_accs = []
    
    # 批量迭代验证集
    for batch in tqdm(val_loader):
        imgs, labels = batch
        # 验证过程无需计算梯度
        with torch.no_grad():
            logits = model(imgs.to(device))
            
        # 计算损失
        loss = criterion(logits, labels.to(device))

        # 计算当前batch的精度
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # 记录损失和精度
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        
    # 更新当前的平均损失和平均精度
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    
    # 如果模型有提升，就更新保存的模型
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), model_path)
        print('saving model with acc {:.3f}'.format(best_acc))
```
补充：官方补充的模型保存方法
该方法是官方推荐的保存模型的方案，保存模型时只保存了已经训练好的模型参数。这种保存方法较于保存整个模型来说更为灵活。这里需要注意的是model.load_state_dict()函数需要的是一个字典类型的输入，所以model.load_state_dict(PATH)是不可行的。这种方法有一个缺点就是需要初始化模型后将保存的模型学习参数赋值给模型。这样你就需要记录你曾经用于初始化模型的参数。
save
```python
torch.save(model.state_dict(), PATH)
```
load
```python
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

## 9.测试并保存结果
```python
# 定义保存地址
saveFileName = './submission.csv'
# 进行预测
# 初始化模型，加载之前保存的权重
model = res_model(176)
model = model.to(device)
model.load_state_dict(torch.load(model_path))

# 开启验证模式
model.eval()

# 保存预测
predictions = []
# 迭代测试集
for batch in tqdm(test_loader):   
    imgs = batch
    with torch.no_grad():
        logits = model(imgs.to(device))
    
    # 以logit最大的类为预测并记录
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

preds = []
for i in predictions:
    preds.append(num_to_class[i])

test_data = pd.read_csv(test_path)
test_data['label'] = pd.Series(preds)
submission = pd.concat([test_data['image'], test_data['label']], axis=1)
submission.to_csv(saveFileName, index=False)
print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!")
```