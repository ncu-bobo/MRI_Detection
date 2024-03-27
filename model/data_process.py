import os
from tqdm import tqdm
import time
import yaml
import SimpleITK as sitk
import numpy as np
import cv2
# 读取源数据，并将set与label组合展示
def wwwc(sitkImage, ww, wc):
    # 设置窗宽窗位
    min = int(wc - ww/2.0)  # 计算最小值
    max = int(wc + ww/2.0)  # 计算最大值
    intensityWindow = sitk.IntensityWindowingImageFilter()  # 创建强度窗口滤波器对象
    intensityWindow.SetWindowMaximum(max)  # 设置窗口最大值
    intensityWindow.SetWindowMinimum(min)  # 设置窗口最小值
    sitkImage = intensityWindow.Execute(sitkImage)  # 对图像进行窗宽窗位调整
    return sitkImage

# 对源数据进行调整，找到一个适合的窗口
def winWid_cen(src_path, ww, wc, file_name):
    # 设置窗宽窗位
    min = int(wc - ww / 2.0)  # 计算最小值
    max = int(wc + ww / 2.0)  # 计算最大值
    intensityWindow = sitk.IntensityWindowingImageFilter()  # 创建强度窗口滤波器对象
    intensityWindow.SetWindowMaximum(max)  # 设置窗口最大值
    intensityWindow.SetWindowMinimum(min)  # 设置窗口最小值

    sitkImage = sitk.ReadImage(src_path)  # 读取图像数据
    sitkImage = intensityWindow.Execute(sitkImage)  # 对图像进行窗宽窗位调整
    sitk.WriteImage(sitkImage, file_name)  # 将调整后的图像保存到指定路径
    return sitkImage

def process():
    # 读取配置变量
    mriVars = yaml.load(open('variables.yaml', encoding='UTF-8'), Loader=yaml.FullLoader)

    # 检查数据集是否已经统一窗宽
    data_winwid = mriVars['data']['data_winwid']
    # 若没有，则统一窗宽
    if not os.path.exists(data_winwid):
        print("数据集调整窗宽窗位中...")
        # 创建相关数据目录
        os.mkdir(data_winwid)
        os.makedirs(os.path.join(data_winwid, 'Set'))
        os.makedirs(os.path.join(data_winwid, 'Label'))

        # 遍历指定范围的文件ID
        for file_id in tqdm(range(1, 81)):
            data_origin = mriVars['data']['data_origin'] + r'/Set/%03d.nii' % file_id
            label_origin = mriVars['data']['data_origin'] + r'/Label/%03d.nii' % file_id
            # 调用winWid_cen函数，设置窗宽窗位参数并保存图像
            winWid_cen(data_origin, mriVars['data']['ww'], mriVars['data']['wc'],
                                    mriVars['data']['data_winwid'] + r'/Set/%03d.nii' % file_id)
            sitkImage = sitk.ReadImage(label_origin)  # 读取图像数据
            sitk.WriteImage(sitkImage, mriVars['data']['data_winwid'] + r'/Label/%03d.nii' % file_id)
            time.sleep(0.001)
    else:
        print("数据集已完成窗宽窗位的调整")

    # 检查数据集是否已经统一裁剪
    data_cropping = mriVars['data']['data_cropping']
    # 若没有，则统一裁剪
    if not os.path.exists(data_cropping):
        print("数据集裁剪中...")
        # 创建相关数据目录
        os.mkdir(data_cropping)
        os.makedirs(os.path.join(data_cropping, 'Set'))
        os.makedirs(os.path.join(data_cropping, 'Label'))

        # 遍历原始数据集中的文件
        for ct_file in os.listdir(os.path.join(data_winwid, 'Set')):
            # 读取CT图像
            ct = sitk.ReadImage(data_winwid + '/Set/' + ct_file, sitk.sitkInt16)
            # 将CT图像转换为numpy数组
            ct_array = sitk.GetArrayFromImage(ct)
            # 读取标签图像
            seg = sitk.ReadImage(data_winwid + '/Label/' + ct_file, sitk.sitkInt8)
            # 将标签图像转换为numpy数组
            seg_array = sitk.GetArrayFromImage(seg)

            # 将灰度值在阈值之外的截断掉
            # 大于上限的灰度值设为上限
            ct_array[ct_array > mriVars['data']['ww']] = mriVars['data']['ww']
            # 小于下限的灰度值设为下限
            ct_array[ct_array < mriVars['data']['wc']] = mriVars['data']['wc']

            # 找到喉部区域开始和结束的slice
            z = np.any(seg_array, axis=(1, 2))  # 沿着z轴方向找到非零值
            start_slice, end_slice = np.where(z)[0][[0, -1]]  # 找到非零值的起始和结束位置

            ct_array = ct_array[start_slice - 2:end_slice + 3, :, :]  # 裁剪CT图像数组
            seg_array = seg_array[start_slice - 2:end_slice + 3, :, :]  # 裁剪标签图像数组

            new_ct = sitk.GetImageFromArray(ct_array)  # 根据数组创建新的CT图像
            new_ct.SetDirection(ct.GetDirection())  # 设置方向
            new_ct.SetOrigin(ct.GetOrigin())  # 设置原点

            new_seg = sitk.GetImageFromArray(seg_array)  # 根据数组创建新的标签图像
            new_seg.SetDirection(ct.GetDirection())  # 设置方向
            new_seg.SetOrigin(ct.GetOrigin())  # 设置原点
            # 保存新的CT图像
            sitk.WriteImage(new_ct, data_cropping + '/Set/' + ct_file)
            # 保存新的标签图像
            sitk.WriteImage(new_seg, data_cropping + '/Label/' + ct_file)
    else:
        print("数据集已完成裁剪")

    # 检查是否已生成img格式数据集
    data_train = mriVars['data']['data_train']
    # 若没有，则统一裁剪
    if not os.path.exists(data_train):
        print("数据集格式转换中...")
        # 创建相关数据目录
        os.mkdir(data_train)
        os.makedirs(os.path.join(data_train, 'Set'))
        os.makedirs(os.path.join(data_train, 'Label'))
        last_dict = {81, 82, 83, 84, 85, 86, 87, 88, 89, 90}  # 索引字典
        count = 0

        for f in tqdm(os.listdir(data_cropping+"/Set")):  # 遍历数据路径下的文件
            fname = int(f.split(".")[0])  # 获取文件名对应的数字索引
            # 排除后10个样本
            if fname in last_dict:  # 如果在排除列表中，则跳过当前循环
                continue
            origin_path = data_cropping+"/Set/" + f  # 原始图像路径
            seg_path = data_cropping+"/Label/" + f  # 标签图像路径
            origin_array = sitk.GetArrayFromImage(sitk.ReadImage(origin_path))  # 读取并转换原始图像为数组
            seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))  # 读取并转换标签图像为数组
            for i in range(seg_array.shape[0]):  # 遍历标签图像数组的第一个维度（即z轴方向）
                seg_image = seg_array[i, :, :]  # 获取当前切片的标签图像数组
                seg_image = np.rot90(np.transpose(seg_image, (1, 0)))  # 对标签图像数组进行旋转和转置操作
                origin_image = origin_array[i, :, :]  # 获取当前切片的原始图像数组
                origin_image = np.rot90(np.transpose(origin_image, (1, 0)))  # 对原始图像数组进行旋转和转置操作
                cv2.imwrite(data_train + '/Label/' + str(count) + '.png', seg_image)  # 将处理后的标签图像保存为.png格式
                cv2.imwrite(data_train + '/Set/' + str(count) + '.jpg', origin_image)  # 将处理后的原始图像保存为.jpg格式
                count += 1  # 计数加一
            time.sleep(0.00001)  # 延时0.00001秒
        # 打印转换后的数据量
        print("训练集图像总数："+ str(count))
    else:
        print("数据集格式已转换完毕")