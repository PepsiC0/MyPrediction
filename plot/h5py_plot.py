import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


def visualize_result(data_name, model_name, nodes_id, time_se):
    data_name = data_name
    model_name = model_name
    h5_file = f'{data_name}/{model_name}/result.h5'
    file_obj = h5py.File(h5_file, "r")   # 获得文件对象，这个文件对象有两个keys："predict"和"target"
    print(file_obj["predict"].shape)
    prediction = file_obj["predict"][:][:, :, 0]  # [N, T],切片，最后一维取第0列，所以变成二维了，要是[:, :, :1]那么维度不会缩减
    target = file_obj["target"][:][:, :, 0]  # [N, T],同上
    file_obj.close()

    plot_prediction = prediction[nodes_id][time_se[0]: time_se[1]]  # [T1]，将指定节点的，指定时间的数据拿出来
    plot_target = target[nodes_id][time_se[0]: time_se[1]]  # [T1]，同上

    plt.figure(figsize=(8, 5))
    plt.title(f'{model_name} 预测结果')
    plt.grid(True)
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_target)
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_prediction)

    plt.legend(["target", "prediction"], loc="best")

    plt.axis([0, time_se[1] - time_se[0],
              np.min(np.array([np.min(plot_prediction), np.min(plot_target)])),
              np.max(np.array([np.max(plot_prediction), np.max(plot_target)]))])
    # plt.title('GAT')
    # plt.savefig("h5py_prediction.png")
    plt.savefig(f'{data_name}/{model_name}/{nodes_id}_predicted.png')
    plt.show()

if __name__ == '__main__':

    # visualize_result(data_name='Xian', model_name='GRU', nodes_id=120, time_se=[0, 288])  # PEMS04 节点数：307；时间288为一天
    visualize_result(data_name='Xian', model_name='GRU', nodes_id=30, time_se=[0, 24])  # Xian 节点数：64；时间24为一天
