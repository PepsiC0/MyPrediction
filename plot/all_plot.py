import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


def all_plot(data_name, model_name, nodes_id, time_se):
    data_name = data_name
    model_name = model_name
    # Cheb = np.load(f'{data_name}/Cheb/predicted.npy')[:][:, :, 0]
    GAT = np.load(f'{data_name}/GAT/predicted.npy')[:][:, :, 0]
    GCN = np.load(f'{data_name}/GCN/predicted.npy')[:][:, :, 0]
    GRU = np.load(f'{data_name}/GRU/predicted.npy')[:][:, :, 0]
    LSTM = np.load(f'{data_name}/LSTM/predicted.npy')[:][:, :, 0]
    ASTGCN = np.load(f'{data_name}/ASTGCN/predicted.npy')[:][:, :, 0]
    test_y = np.load(f'{data_name}/LSTM/test.npy')[:][:, :, 0]

    # Cheb_plot = Cheb[nodes_id][time_se[0]: time_se[1]]
    GAT_plot = GAT[nodes_id][time_se[0]: time_se[1]]
    GCN_plot = GCN[nodes_id][time_se[0]: time_se[1]]
    GRU_plot = GRU[nodes_id][time_se[0]: time_se[1]]
    LSTM_plot = LSTM[nodes_id][time_se[0]: time_se[1]]
    ASTGCN_plot = ASTGCN[nodes_id][time_se[0]: time_se[1]]

    test_y_plot = test_y[nodes_id][time_se[0]: time_se[1]]
    plt.figure(figsize=(8, 5))
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), test_y_plot, label='real', color='black')
    # plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), Cheb_plot, label='Cheb')
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), GAT_plot, label='GAT')
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), GCN_plot, label='GCN')
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), GRU_plot, label='GRU')
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), LSTM_plot, label='LSTM')
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), LSTM_plot, label='ASTGCN')
    plt.title(f' 预测结果 ')
    plt.legend(loc='best')
    # plt.grid(visible=True)
    plt.savefig(f'all_predicted.svg')
    plt.show()

if __name__ == '__main__':
    all_plot(data_name='PEMS04', model_name=None, nodes_id=120, time_se=[0, 12 * 24])
