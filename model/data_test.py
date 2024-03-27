import SimpleITK as sitk
import matplotlib.pyplot as plt
import data_process
import cv2


# 查看初始数据情况
def origin_watch():
    origin = sitk.ReadImage('data/origin/Set/001.nii')  # 读取原始图像数据
    print(origin.GetSize())  # 打印原始图像大小
    mask = sitk.ReadImage('data/origin/Label/001.nii')  # 读取标签数据
    origin = data_process.wwwc(origin, 350, 50)  # 调用wwwc函数，设置窗宽窗位参数
    mask = sitk.Cast(mask, sitk.sitkUInt8)  # 将mask转换为无符号8位整数类型
    # 原图和mask叠加在一起
    new_data = sitk.LabelOverlay(origin, mask, opacity=0.01)
    data = sitk.GetArrayFromImage(new_data)  # 从图像获取数组数据
    d = data.shape[0]

    plt.figure(figsize=(12, 12))  # 创建画布
    for index, i in enumerate(range(int(d*0.6)+2, int(d*0.6)+14)):  # 遍历指定范围的数据
        plt.subplot(4, 3, index+1)  # 创建子图
        plt.imshow(data[i, ...])  # 显示图像数据
        plt.axis('off')  # 关闭坐标轴
        plt.subplots_adjust(left=0.0, bottom=0.0, top=1, right=0.8, wspace=0.0001, hspace=0.0001)  # 调整子图间距
    plt.show()  # 显示图形

# 查看图片情况
def image_watch():
    # 显示某一层喉部图片
    image = cv2.imread('data/data_train/Set/50.jpg', 0)  # 读取原始图像
    label = cv2.imread('data/data_train/Label/50.png', 0)  # 读取标签图像
    plt.figure(figsize=(10, 5))  # 创建画布
    # 第一个子图
    plt.subplot(121)
    plt.imshow(image, 'gray')  # 显示原始图像
    # 第二个子图
    plt.subplot(122)
    # 显示标签图像，乘以255转换为灰度图
    plt.imshow(label * 255, 'gray')
    plt.show()  # 显示图像

if __name__ == '__main__':
    image_watch()