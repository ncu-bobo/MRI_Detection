import os

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
import cv2
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
    batch_size = mriVars['train']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 定义模型参数
    epochs = mriVars['train']['epochs']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = mriVars['train']['learning_rate']
    model_state_dict = mriVars['train']['model_state_dict']
    res = {'epoch': [], 'loss': [], 'dice': []}

    unet = UNet(1, 1).to(device).apply(weights_init)
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(unet.parameters(), learning_rate)

    model_path = "./model_store/weights_3.pth"
    if os.path.exists(model_path):
        unet.load_state_dict(torch.load(model_path, map_location=device))
        start_epoch = 2
        print('加载成功！')
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')

    for epoch in range(start_epoch+1, epochs):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)
        dt_size = len(train_loader.dataset)
        step = 0
        for x, y in train_loader:
            step += 1
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = unet(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                res['epoch'].append((epoch + 1) * step)
                res['loss'].append(loss.item())
                print("epoch%d step%d/%d train_loss:%0.3f\n" % (
                    epoch, step, (dt_size - 1) // train_loader.batch_size + 1, loss.item()),
                      end='')
        if (epoch + 1) % 1 == 0:
            torch.save(unet.state_dict(), './model_store/weights_%d.pth' % (epoch + 1))


    save_root = './data/predict'
    # 模型设置为评估模式
    unet.eval()
    # plt.ion()
    index = 0
    with torch.no_grad():
        index = 0
        for x, ground in test_loader:
            index += 1
            print("step: %d/%d" % (index, len(test_loader)))
            x = x.type(torch.FloatTensor)
            y = unet(x)
            x = torch.squeeze(x)
            x = x.unsqueeze(0)
            ground = torch.squeeze(ground)
            # ground = ground.unsqueeze(0)
            img_ground = ground.detach().numpy()
            img_x = x.detach().numpy()
            img_y = torch.squeeze(y).numpy()
            # cv2.imshow('img', img_y)
            src_path = os.path.join(save_root, "predict_%d_s.png" % index)
            save_path = os.path.join(save_root, "predict_%d_o.png" % index)
            ground_path = os.path.join(save_root, "predict_%d_g.png" % index)
            # img_ground.save(ground_path)
            # img_x.save(src_path)
            # plt.imshow(img_y)
            # plt.pause(0.5)
            cv2.imwrite(save_path, img_y * 255)
            cv2.imwrite(ground_path, img_ground * 255)
            print(save_path)
            index = index + 1

        # plt.show()

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


# def test():
#     save_root = './data/predict'
#     # 模型设置为评估模式
#     unet.eval()
#     # plt.ion()
#     index = 0
#     with torch.no_grad():
#         index = 0
#         for x, ground in test_loader:
#             index += 1
#             print("step: %d/%d" % (index, len(test_loader)))
#             x = x.type(torch.FloatTensor)
#             y = unet(x)
#             x = torch.squeeze(x)
#             x = x.unsqueeze(0)
#             ground = torch.squeeze(ground)
#             # ground = ground.unsqueeze(0)
#             img_ground = ground.detach().numpy()
#             img_x = x.detach().numpy()
#             img_y = torch.squeeze(y).numpy()
#             # cv2.imshow('img', img_y)
#             src_path = os.path.join(save_root, "predict_%d_s.png" % index)
#             save_path = os.path.join(save_root, "predict_%d_o.png" % index)
#             ground_path = os.path.join(save_root, "predict_%d_g.png" % index)
#             # img_ground.save(ground_path)
#             # img_x.save(src_path)
#             # plt.imshow(img_y)
#             # plt.pause(0.5)
#             cv2.imwrite(save_path, img_y * 255)
#             cv2.imwrite(ground_path, img_ground * 255)
#             print(save_path)
#             index = index + 1


# 计算Dice系数
def dice_calc(args):
    root = './data/predict'
    nums = len(os.listdir(root)) // 3
    dice = list()
    dice_mean = 0
    for i in range(nums):
        ground_path = os.path.join(root, "predict_%d_g.png" % i)
        predict_path = os.path.join(root, "predict_%d_o.png" % i)
        img_ground = cv2.imread(ground_path)
        img_predict = cv2.imread(predict_path)
        intersec = 0
        x = 0
        y = 0
        for w in range(256):
            for h in range(256):
                intersec += img_ground.item(w, h, 1) * img_predict.item(w, h, 1) / (255 * 255)
                x += img_ground.item(w, h, 1) / 255
                y += img_predict.item(w, h, 1) / 255
        if x + y == 0:
            current_dice = 1
        else:
            current_dice = round(2 * intersec / (x + y), 3)
        dice_mean += current_dice
        dice.append(current_dice)
    dice_mean /= len(dice)
    print(dice)
    print(round(dice_mean, 3))