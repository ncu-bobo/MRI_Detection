import torch
import yaml
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from CustomDataset import CustomDataset
from TransformDataset import TransformDataset
from model.Unet import UNet
import matplotlib.pyplot as plt
from torch.nn import init
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv3d') != -1:
        init.xavier_normal(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)

def train():
    # 读取配置变量
    mriVars = yaml.load(open('variables.yaml', encoding='UTF-8'), Loader=yaml.FullLoader)

    # 构建训练集、测试集
    # 定义转换
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(mriVars['train']['degrees']),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 数据和标签目录
    image_dir = mriVars['data']['data_train'] + '/Set'
    label_dir = mriVars['data']['data_train'] + '/Label'
    # 创建完整的数据集
    full_dataset = CustomDataset(image_dir=image_dir, label_dir=label_dir, transform=None)

    # 确定训练集和测试集的大小 8:2
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # 随机分割数据集
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 为分割后的数据集应用不同的transform
    train_dataset = TransformDataset(train_dataset, transform=transform_train)
    test_dataset = TransformDataset(test_dataset, transform=transform_test)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 定义模型参数
    epochs = mriVars['train']['epochs']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = mriVars['train']['learning_rate']
    model_state_dict = mriVars['train']['model_state_dict']
    res = {'epoch': [], 'loss': [], 'dice': []}

    unet = UNet(256, 256).to(device).apply(weights_init)
    # unet.load_state_dict(torch.load(model_state_dict))
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(unet.parameters(), learning_rate)

    for epoch in range(epochs):
        dt_size = len(train_loader.dataset)
        step = 0
        for x, y in train_loader:
            step += 1
            # x = x[0].to(device)
            # y = y[0].to(device)
            print(x.size())
            print(y.size())
            optimizer.zero_grad()
            outputs = unet(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                res['epoch'].append((epoch + 1) * step)
                res['loss'].append(loss.item())
                print("epoch%d step%d/%d train_loss:%0.3f" % (
                    epoch, step, (dt_size - 1) // train_loader.batch_size + 1, loss.item()),
                      end='')
    plt.plot(res['epoch'], np.squeeze(res['cost']), label='Train cost')
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title("Model: train cost")
    plt.legend()

    plt.plot(res['epoch'], np.squeeze(res), label='Validation cost', color='#FF9966')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title("Model:validation  loss")
    plt.legend()

    plt.savefig("examples.jpg")