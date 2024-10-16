import matplotlib.pyplot as plt
import csv

import numpy as np
import pandas as pd
import os
import torch

from matplotlib.pyplot import MultipleLocator
from scipy.interpolate import make_interp_spline

def draw_different_reward_functions():
    labels = ['Intersection', 'Roundabout', 'Left turn']
    path = os.path.abspath('..')            # 获取上一级目录
    df = pd.read_csv(path + '/draw_csv/functions_Ablation.csv', index_col = False)

    y_1 = torch.Tensor(df.loc[:, '1'])
    y_2 = torch.Tensor(df.loc[:, '2'])
    y_3 = torch.Tensor(df.loc[:, '3'])
    y_4 = torch.Tensor(df.loc[:, '4'])
    y_our = torch.Tensor(df.loc[:, '5'])

    plt.figure(figsize = (5, 4))
    plt.xlabel('Traffic scenarios')                                         # x轴标题
    plt.ylabel('Success rate')                                          # y轴标题

    x = range(1, 2)
    plt.plot(x, y_1, 'r')               # 红
    plt.plot(x, y_2, 'b')               # 蓝
    plt.plot(x, y_1, 'k-')              # 黑
    plt.plot(x, y_2, 'g-')              # 绿
    plt.plot(x, y_2, 'm')               # 紫

    plt.legend(labels)
    plt.savefig(path + "/draw_results/" + "functions.jpg")
    plt.show()


def draw_different_modules():
    labels = ['Intersection', 'Roundabout', 'Left turn']

    path = os.path.abspath('..')            # 获取上一级目录
    df = pd.read_csv(path + '/draw_csv/modules_Ablation.csv', index_col = False)

    y_1 = torch.Tensor(df.loc[:, '1'])
    y_2 = torch.Tensor(df.loc[:, '2'])
    y_our = torch.Tensor(df.loc[:, '3'])



    plt.figure(figsize = (5, 4))
    x = range(1, 2)
    width = 0.2  # 柱子的宽度
    x = np.arange(len(labels))  # x轴刻度标签位置

    plt.bar(x - width, y_1, width, label = 'w/o DRLM', color ='royalblue')
    plt.bar(x, y_2, width, label = 'w/o EPLM', color ='darkorange')
    plt.bar(x + width, y_our, width, label = 'Ours', color ='forestgreen')

    plt.ylabel('Success rate',  fontsize = 18)
    # plt.xlabel('Traffic scenarios', fontsize = 10)

    plt.xticks(x, labels = labels, fontsize = 18)
    plt.legend(loc = 'center', bbox_to_anchor = (0.5, 1.05), ncol = 3, fontsize = 10)

    plt.savefig(path + "/draw_results/" +"modules.jpg")
    plt.show()

def draw_different_samples():

    labels = ['Intersection', 'Roundabout', 'Left turn']

    path = os.path.abspath('..')            # 获取上一级目录
    df = pd.read_csv(path + '/draw_csv/samples_Ablation.csv', index_col = False)

    y_1 = torch.Tensor(df.loc[:, '1'])
    y_2 = torch.Tensor(df.loc[:, '2'])
    y_our = torch.Tensor(df.loc[:, '3'])


    print('y_1 =', y_1)
    print('y_2 =', y_2)
    print('y_our =', y_our)

    plt.figure(figsize = (8, 6))
    x = range(1, 2)
    width = 0.2  # 柱子的宽度
    x = np.arange(len(labels))  # x轴刻度标签位置

    plt.bar(x - width, y_1, width, label = 'size = 10', color ='royalblue')
    plt.bar(x, y_2, width, label = 'size = 20', color ='darkorange')
    plt.bar(x + width, y_our, width, label = 'size = 40', color ='forestgreen')

    plt.ylabel('Success rate', fontsize = 16)
    # plt.xlabel('Traffic scenarios', fontsize = 14)

    plt.xticks(x, labels = labels, fontsize = 16)
    plt.legend(loc = 'center', bbox_to_anchor = (0.5, 1.05), ncol = 3, fontsize = 12)
    # plt.legend(fontsize = 8)

    plt.savefig(path + "/draw_results/" + "samples.jpg")
    plt.show()



if __name__ == '__main__':
    # draw_different_modules()
    draw_different_samples()