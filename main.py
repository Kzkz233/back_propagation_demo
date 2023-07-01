# -*- coding: UTF-8 -*-

"""
BR算法组入队考核教学，考察手写反向传播
"""

from model import *
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


if __name__ == '__main__':
    # 超参数
    max_epoch = 100
    lr = 5e-6

    # 加载数据集
    data_df = pd.read_csv('./data.csv', header=None)
    data = data_df.values

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(data[:,0:2], data[:,2], test_size=0.2)

    # 构造网络
    my_net = [MyLinearWithLeakyRelu(input_dim=2, output_dim=256),
              MyDropout(input_dim=256, drop_rate=0.2),
              MyLinear(input_dim=256, output_dim=1)]

    # 训练
    for i in tqdm(range(max_epoch)):
        sum_loss = 0
        count = 0

        for j in range(len(X_train)):
            x = X_train[j:j+1]
            y = y_train[j:j+1]

            # 前向传播
            for tmp_layer in my_net:
                x = tmp_layer.forward(x)

            loss = 1/2 * (x - y) * (x - y)
            last_grad = x - y
            sum_loss += loss
            count += 1
            # 反向传播
            tmp_grade = last_grad
            for tmp_layer in my_net[::-1]:
                tmp_grade = tmp_layer.backward(tmp_grade, lr)

            # 更新网络
            for tmp_layer in my_net:
                tmp_layer.update()

        # 平均train loss
        avg_loss = sum_loss/count
        print('\navg_loss:\t', avg_loss)

    # 测试
    for i in range(len(X_test)):
        x = X_test[i:i + 1]
        y = y_test[i:i + 1]

        for tmp_layer in my_net:
            x = tmp_layer.forward(x)
        print('y_pred:\t', x[0], '\ty_true:\t', y)





