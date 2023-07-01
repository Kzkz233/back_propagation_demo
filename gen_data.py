# -*- coding: UTF-8 -*-

"""
生成测试用数据
"""

import numpy as np


if __name__ == '__main__':
    num = 10000
    w = np.array([5, -7])
    b = 3
    x = np.random.rand(2, num) * 10
    noise = np.random.normal(size=num)

    y = w @ x + b + noise

    sum = np.concatenate((x, y[np.newaxis, :]), axis=0)
    np.savetxt('data.csv', sum.T, fmt='%f', delimiter=',')  # frame: 文件 array:存入文件的数组
