import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
model_name = 'LSTM'  # LSTM、GRU ···


predicted = np.load(f'{model_name}/predicted.npy')
test_y = np.load(f'{model_name}/test.npy')

# plt.plot(test_y[:][:, :, 0],  label='real')
# plt.plot(predicted[:][:, :, 0], label='prediction')
plt.plot(test_y[:72], label='real')
plt.plot(predicted[:72], label='prediction')
plt.title(f'{model_name} 预测结果')
plt.legend(loc='best')
plt.savefig(f'{model_name}/predicted.png')
plt.show()
