import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
data_name = 'PEMS04'  # PEMS04

Cheb = np.load(f'{data_name}/Cheb/predicted.npy')
GAT = np.load(f'{data_name}/GAT/predicted.npy')
GCN = np.load(f'{data_name}/GCN/predicted.npy')
GRU = np.load(f'{data_name}/GRU/predicted.npy')
LSTM = np.load(f'{data_name}/LSTM/predicted.npy')
test_y = np.load(f'{data_name}/Cheb/test.npy')

plt.figure(figsize=(10, 5))
plt.plot(test_y[:48], label='real')
# plt.plot(Cheb[:48], label='Cheb')
# plt.plot(GAT[:48], label='GAT')
plt.plot(GCN[:48], label='GCN')
plt.plot(GRU[:48], label='GRU')
plt.plot(LSTM[:48], label='LSTM')
plt.title(f' 预测结果 ')
plt.legend(loc='best')
plt.savefig(f'all_predicted.png')
plt.show()
