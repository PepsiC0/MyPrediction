import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


def plot(data_name, model_name, nodes_id, time_se):
    data_name = data_name
    model_name = model_name

    predicted = np.load(f'{data_name}/{model_name}/predicted.npy')
    predicted = predicted[:][:, :, 0]
    test_y = np.load(f'{data_name}/{model_name}/test.npy')
    test_y = test_y[:][:, :, 0]
    # print(test_y.shape)

    plot_prediction = predicted[nodes_id][time_se[0]: time_se[1]]  # [T1]，将指定节点的，指定时间的数据拿出来
    # print(plot_prediction.shape)
    plot_target = test_y[nodes_id][time_se[0]: time_se[1]]  # [T1]，同上

    # a = test_y[306][time_se[0]: time_se[1]]
    # b = test_y[20][time_se[0]: time_se[1]]
    # print(a)
    # print(b)
    plt.figure(figsize=(8, 5))
    plt.title(f'{model_name} 预测结果')
    # plt.plot(test_y[:72], label='real')
    # plt.plot(predicted[:72], label='prediction')
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_target, label='real')
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_prediction, label='prediction')
    plt.legend(loc='best')
    plt.grid(visible=True)
    # plt.plot(a)
    # plt.plot(b)
    plt.savefig(f'{data_name}/{model_name}/{nodes_id}_predicted.png')
    plt.show()


if __name__ == '__main__':

    plot(data_name='PEMS04', model_name='GRU', nodes_id=200, time_se=[0, 288])
