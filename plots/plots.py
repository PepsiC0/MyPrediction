import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


def plot(nodes_id, time_se):


    predicted = np.load("./PEMS04/LSTM/predicted.npy")
    predicted = np.transpose(predicted)
    print(predicted.shape)
    test_y = np.load('./PEMS04/LSTM//test.npy')
    test_y = np.transpose(test_y, (2, 0, 1))[:][:, :, 0]
    print(test_y.shape)

    plot_prediction = predicted[nodes_id][time_se[0]: time_se[1]] # [T1]，将指定节点的，指定时间的数据拿出来
    print(plot_prediction.shape)
    plot_target = test_y[nodes_id][time_se[0]: time_se[1]]  # [T1]，同上


    plt.figure(figsize=(8, 5))
    plt.title('LSTM 预测结果')
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_target, label='real')
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_prediction, label='prediction')
    plt.legend(loc='best')
    plt.grid(visible=True)
    plt.savefig(f'./PEMS04/LSTM/{nodes_id}_predicted.png')
    plt.show()


if __name__ == '__main__':

    plot(nodes_id=120, time_se=[0, 288])
