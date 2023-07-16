import torch
import torch.nn as nn
import numpy as np
import os
import logging
from data.Xian.dataset import LoadData
from utils.utils import Evaluation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelTester:
    def __init__(self, model, criterion, model_name):
        self.model = model
        self.criterion = criterion
        self.model_name = model_name

    def test(self, test_data):
        self.model.eval()
        with torch.no_grad():
            MAE, MAPE, RMSE = [], [], []  # 定义三种指标的列表
            total_loss = 0.0
            for data in test_data:
                predicted = self.model(data['flow_x'].to(device))

                loss = self.criterion(predicted, data["flow_y"].to(device))  # 使用MSE计算loss

                total_loss += loss.item()  # 所有的batch的loss累加
                # # 将预测结果转换为NumPy数组
                # predicted = predicted.squeeze().cpu()
                # 将预测结果转换为张量
                predicted = predicted.to(device)
                test_y = data['flow_y'].squeeze().view(-1)
                # 将预测结果转换为NumPy数组
                predicted = predicted.view(-1).cpu().detach().numpy()
                test_y = test_y.cpu().numpy()
                # print(predicted.shape)
                # print(type(predicted))
                # print(test_y.shape)
                # print(type(test_y))
                # mae = np.mean(np.abs(test_y - predicted)).item()
                # rmse = np.sqrt(np.mean((test_y - predicted) ** 2)).item()
                # mape = np.mean(np.abs((test_y - predicted) / test_y)) * 100
                # # print(np.abs((test_y - predicted) / test_y))
                # print(f'MAE: {mae}, RMSE: {rmse}, MAPE: {mape}')

                mae, mape, rmse = Evaluation.total(test_y.reshape(-1), predicted.reshape(-1))  # 变成常向量才能计算这三种指标

                performance = [mae, mape, rmse]
                MAE.append(performance[0])
                MAPE.append(performance[1])
                RMSE.append(performance[2])
                # print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))

                # dataset = test_data.dataset
                predicted = LoadData.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], predicted)
                test_y = LoadData.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], test_y)

                # mae = np.mean(np.abs(test_y - predicted)).item()
                # rmse = np.sqrt(np.mean((test_y - predicted) ** 2)).item()
                # mape = np.mean(np.abs((test_y - predicted) / test_y)) * 100
                # # print(np.abs((test_y - predicted) / test_y))
                # # print(f'MAE: {mae}, RMSE: {rmse}, MAPE: {mape}')

            print("Performance:  MAE {:02.4f}   RMSE {:02.4f}   MAPE {:02.4f}%".format(np.mean(mae), np.mean(mape),
                                                                                       np.mean(rmse * 100)))
            # 保存结果
            save_dir = f'./plot/{self.model_name}'
            pre_dir = 'predicted.npy'
            test_dir = 'test.npy'
            # 创建保存目录（如果不存在）
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, pre_dir), predicted)
            np.save(os.path.join(save_dir, test_dir), test_y)

            # 配置日志
            log_file = f'logs/{self.model_name}/{self.model_name}.log'
            logging.basicConfig(filename=log_file, level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s')
            # 配置日志
            logging.info(f'MAE: {mae}')
            logging.info(f'MSE: {rmse}')
            logging.info(f'MAPE: {mape}')

