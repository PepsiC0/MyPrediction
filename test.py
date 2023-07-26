import torch
import torch.nn as nn
import numpy as np
import os
import logging
from data.Xian.dataset import LoadData
from utils.utils import Evaluation
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelTester:
    def __init__(self, model, criterion, model_name):
        self.model = model
        self.criterion = criterion
        self.model_name = model_name

    def test(self, test_data):
        self.model.eval()
        with torch.no_grad():
            # MAE, MAPE, RMSE = [], [], []  # 定义三种指标的列表
            total_loss = 0.0
            # Target = np.zeros([64, 1, 1])  # [N, T, D],T=1 ＃ 目标数据的维度，用０填充
            # Predict = np.zeros_like(Target)  # [N, T, D],T=1 # 预测数据的维度

            for data in test_data:
                predicted = self.model(data).to(device)
                print(predicted.shape)
                test_y = data['flow_y'].view(data['flow_y'].size(0)*data['flow_y'].size(1), 1)
                loss = self.criterion(predicted, test_y.to(device))  # 使用MSE计算loss
                print(data["flow_y"].shape)
                total_loss += loss.item()  # 所有的batch的loss累加

                # 将预测结果转换为张量
                predicted = predicted.to(device)
                # test_y = data['flow_y'].squeeze().view(-1)
                # test_y = data['flow_y'].squeeze().view(-1)
                # 将预测结果转换为NumPy数组
                # predicted = predicted.view(-1).cpu().detach().numpy()
                predicted = predicted.cpu().detach().numpy()
                print(predicted.shape)
                test_y = test_y.cpu().numpy()
                print(test_y.shape)

                MAE = np.mean(np.abs(test_y - predicted)).item()
                RMSE = np.sqrt(np.mean((test_y - predicted) ** 2)).item()
                MAPE = np.mean(np.abs((test_y - predicted) / test_y)) * 100
                # # print(np.abs((test_y - predicted) / test_y)
                # print(f'MAE: {mae}, RMSE: {rmse}, MAPE: {mape}')

                # mae, mape, rmse = Evaluation.total(test_y.reshape(-1), predicted.reshape(-1))  # 变成常向量才能计算这三种指标
                #
                # performance = [mae, mape, rmse]
                # MAE.append(performance[0])
                # MAPE.append(performance[1])
                # RMSE.append(performance[2])


            # print("Performance:  MAE {:02.4f}   MAPE {:02.4f}   RMSE {:02.4f}%".format(np.mean(MAE), np.mean(MAPE),
            #                                                                            np.mean(RMSE * 100)))
            print("Performance:  MAE {:2.2f}  RMSE {:2.2f}  MAPE {:2.2f}% ".format(np.mean(MAE), np.mean(RMSE),
                                                                                   np.mean(MAPE)))
            # Predict = np.delete(predicted, 0, axis=1)  # 将第0行的0删除，因为开始定义的时候用0填充，但是时间是从1开始的
            # Target = np.delete(test_y, 0, axis=1)
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
            logging.info(f'MAE: {MAE}')
            logging.info(f'MSE: {RMSE}')
            logging.info(f'MAPE: {MAPE}')

    def compute_performance(self, prediction, target, data):  # 计算模型性能
        # 下面的try和except实际上在做这样一件事：当训练+测试模型的时候，数据肯定是经过dataloader的，所以直接赋值就可以了
        # 但是如果将训练好的模型保存下来，然后测试，那么数据就没有经过dataloader，是dataloader型的，需要转换成dataset型。
        try:
            dataset = data.dataset  # 数据为dataloader型，通过它下面的属性.dataset类变成dataset型数据
        except:
            dataset = data  # 数据为dataset型，直接赋值

        # 下面就是对预测和目标数据进行逆归一化，recover_data()函数在上一小节的数据处理中
        #  flow_norm为归一化的基，flow_norm[0]为最大值，flow_norm[1]为最小值
        # prediction.numpy()和target.numpy()是需要逆归一化的数据，转换成numpy型是因为 recover_data()函数中的数据都是numpy型，保持一致
        prediction = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], prediction.numpy())
        target = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], target.numpy())

        # 对三种评价指标写了一个类，这个类封装在另一个文件中，在后面
        mae, mape, rmse = Evaluation.total(target.reshape(-1), prediction.reshape(-1))  # 变成常向量才能计算这三种指标

        performance = [mae, mape, rmse]
        recovered_data = [prediction, target]

        return performance, recovered_data  # 返回评价结果，以及恢复好的数据（为可视化准备的）
