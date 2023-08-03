import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
model_name = 'LSTM'  # LSTM、GRU ···
data_name = 'PEMS04'

predicted = np.load(f'{data_name}/{model_name}/predicted.npy')
test_y = np.load(f'{data_name}/{model_name}/test.npy')

print(test_y.shape)
# plt.plot(test_y[:][:, :, 0],  label='real')
# plt.plot(predicted[:][:, :, 0], label='prediction')
plt.figure(figsize=(8,5))
plt.plot(test_y[:72], label='real')
plt.plot(predicted[:72], label='prediction')
plt.title(f'{model_name} 预测结果')
plt.legend(loc='best')
plt.savefig(f'{data_name}/{model_name}/predicted.png')
plt.show()
