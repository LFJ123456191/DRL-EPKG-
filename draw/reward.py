import matplotlib.pyplot as plt
import csv

import numpy as np
import pandas as pd
import os
import torch

from matplotlib.pyplot import MultipleLocator
from scipy.interpolate import make_interp_spline

def draw_different_rewards():
    path = os.path.abspath('..')            # 获取上一级目录

    plt.figure(figsize = (8, 6))

    x = [2, 4, 6]
    x_name = ['Intersection', 'Roundabout', 'Left turn']
    df = pd.read_csv(path + '/draw_csv/rewards_Ablation.csv', index_col = False)
    y_v = torch.Tensor(df.loc[:, 'without_v'])
    y_a = torch.Tensor(df.loc[:, 'without_a'])
    y_c = torch.Tensor(df.loc[:, 'without_c'])
    y_ours = torch.Tensor(df.loc[:, 'ours'])

    # plt.xlabel('Traffic scenarios', fontsize = 18)  # x轴标题
    plt.ylabel('Success rate', fontsize = 16)  # y轴标题
    plt.xticks(x, x_name, fontsize = 16)

    plt.tick_params(axis = 'y', labelsize = 16)         # 设置纵坐标显示数值


    plt.plot(x, y_v, linestyle ='--', marker = 'o', linewidth = 3 )
    plt.plot(x, y_a, linestyle ='--', marker = 'o', linewidth = 3)
    plt.plot(x, y_c, linestyle ='--', marker = 'o', linewidth = 3)
    plt.plot(x, y_ours, linestyle ='--', marker = 'o', linewidth = 3)

    plt.legend(['w/o v', 'w/o a', 'w/o c', 'Ours'], fontsize = 16)


    for a, b in zip(x, y_v):                                                    # 显示数值
        plt.text(a, b + 0.003, round(b.item() + 0.001, 2), ha = 'center', va = 'bottom', fontsize = 12 )

    for a, b in zip(x, y_a):
        plt.text(a, b + 0.003, round(b.item() + 0.001, 2), ha = 'center', va = 'bottom', fontsize = 12 )

    for a, b in zip(x, y_c):
        plt.text(a, b - 0.01, round(b.item() + 0.001, 2), ha = 'center', va = 'bottom', fontsize = 12 )

    for a, b in zip(x, y_ours):
        plt.text(a, b + 0.003, round(b.item() + 0.001, 2), ha = 'center', va = 'bottom', fontsize = 12 )



    plt.savefig(path + "/draw_results/" + "rewards.jpg")
    plt.show()


if __name__ == '__main__':
    draw_different_rewards()