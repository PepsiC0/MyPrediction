import h5py
import torch
import numpy as np
import os
import logging
from data.Xian.dataset import LoadData
from utils.utils import Evaluation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelTester:
    def __init__(self, model, criterion, model_name, data_name, data):
        self.model = model
        self.criterion = criterion
        self.model_name = model_name
        self.data_name = data_name
        self.data = data

    def test(self, test_data):
        self.model.eval()
        with torch.no_grad():
            MAE, RMSE, MAPE = [], [], []  # 定义三种指标的列表
            total_loss = 0.0
            Target = np.zeros([64, 1, 1])  # [N, T, D],T=1 ＃ 目标数据的维度，用０填充 --Xian
            # Target = np.zeros([307, 1, 1])  # [N, T, D],T=1 ＃ 目标数据的维度，用０填充 -- PEMS04
            Predict = np.zeros_like(Target)  # [N, T, D],T=1 # 预测数据的维度
            count = 0
            for data in test_data:
                predicted = self.model(data).to(device)
                test_y = data['flow_y']
                loss = self.criterion(predicted, test_y.to(device))  # 使用MSE计算loss
                total_loss += loss.item()  # 所有的batch的loss累加
                print(total_loss)
                count += 1

                predicted = predicted.transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]
                test_y = test_y.transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]
                predicted = predicted.cpu().detach().numpy()
                test_y = test_y.cpu().numpy()
                performance, data_to_save = self.compute_performance(predicted, test_y,
                                                                     test_data)  # 计算模型的性能，返回评价结果和恢复好的数据

                # 下面这个是每一个batch取出的数据，按batch这个维度进行串联，最后就得到了整个时间的数据，也就是
                # [N, T, D] = [N, T1+T2+..., D]
                Predict = np.concatenate([Predict, data_to_save[0]], axis=1)
                Target = np.concatenate([Target, data_to_save[1]], axis=1)
                # print(predicted)
                # print(Target.shape)

                MAE.append(performance[0])
                RMSE.append(performance[1])
                MAPE.append(performance[2])
                # print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))
                print(f"Test Loss: {100 * total_loss / len(data):02.4f}, Epoch: {count}")

            # print("Performance:  MAE {:02.4f}   MAPE {:02.4f}   RMSE {:02.4f}%".format(np.mean(MAE), np.mean(MAPE),
            #                                                                            np.mean(RMSE * 100)))
            print("Performance:  MAE {:2.2f}  RMSE {:2.2f}  MAPE {:2.2f}% ".format(np.mean(MAE), np.mean(RMSE),
                                                                                   np.mean(MAPE)))
            Predict = np.delete(Predict, 0, axis=1)  # 将第0行的0删除，因为开始定义的时候用0填充，但是时间是从1开始的
            Target = np.delete(Target, 0, axis=1)
            # 保存结果
            save_dir = f'./plot/{self.data_name}/{self.model_name}'
            pre_dir = 'predicted.npy'
            test_dir = 'test.npy'
            result_file = f'./plot/{self.data_name}/{self.model_name}/result.h5'

            # 创建保存目录（如果不存在）
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, pre_dir), Predict)
            np.save(os.path.join(save_dir, test_dir), Target)

            file_obj = h5py.File(result_file, "w")  # 将预测值和目标值保存到文件中，因为要多次可视化看看结果
            file_obj["predict"] = Predict  # [N, T, D]
            file_obj["target"] = Target  # [N, T, D]
            # print(file_obj["predict"].shape)
            # 配置日志
            log_file = f'logs/{self.data_name}/{self.model_name}/{self.model_name}.log'
            logging.basicConfig(filename=log_file, level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s')
            # 配置日志
            logging.info(f'MAE: {np.mean(MAE)}')
            logging.info(f'RMSE: {np.mean(RMSE)}')
            logging.info(f'MAPE: {np.mean(MAPE)}')

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
        prediction = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], prediction)
        target = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], target)

        # 对三种评价指标写了一个类，这个类封装在另一个文件中，在后面
        mae, rmse, mape = Evaluation.total(target.reshape(-1), prediction.reshape(-1))  # 变成常向量才能计算这三种指标

        performance = [mae, rmse, mape]
        recovered_data = [prediction, target]

        return performance, recovered_data  # 返回评价结果，以及恢复好的数据（为可视化准备的）
