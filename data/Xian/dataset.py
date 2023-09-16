import numpy as np
import pandas as pd
import csv
import torch
import numpy as np
from torch.utils.data import Dataset


def get_adjacent_matrix(distance_file: str, num_nodes, data_name, graph_type='connect') -> np.array:
    # 读取邻接矩阵文件
    A = np.zeros([int(num_nodes), int(num_nodes)])
    if data_name == 'Xian':
        data = pd.read_csv(distance_file, index_col=0)
        # 提取邻接矩阵数据
        A = data.values
    elif data_name == 'PEMSO4':
        num_nodes = num_nodes
        # A = np.zeros([int(num_nodes), int(num_nodes)])  # 构造全0的邻接矩阵
        with open(distance_file, "r") as f_d:
            f_d.readline()  # 表头，跳过第一行.
            reader = csv.reader(f_d)  # 读取.csv文件.
            for item in reader:  # 将一行给item组成列表
                if len(item) != 3:  # 长度应为3，不为3则数据有问题，跳过
                    continue
                i, j, distance = int(item[0]), int(item[1]), float(item[2])

                if graph_type == "connect":  # 这个就是将两个节点的权重都设为1，也就相当于不要权重
                    A[i, j], A[j, i] = 1., 1.
                elif graph_type == "distance":  # 这个是有权重，下面是权重计算方法
                    A[i, j] = 1. / distance
                    A[j, i] = 1. / distance
                else:
                    raise ValueError("graph type is not correct (connect or distance)")
        return A
    return A


def get_flow_data(flow_file: str, data_name) -> np.array:
    data = np.load(flow_file)

    # Xian
    if data_name == 'Xian':
        flow_data = data['arr_0'][:, :, 0][:, :, np.newaxis]  # [64, 1464, 1]
    # PEMS04
    elif data_name == 'PEMS04':
        flow_data = data['data'].transpose([1, 0, 2])[:, :, 0][:, :,
                    np.newaxis]  # [N, T, D],transpose就是转置，让节点纬度在第0位，N为节点数，T为时间，D为节点特征

    return flow_data


class LoadData(Dataset):  # 这个就是把读入的数据处理成模型需要的训练数据和测试数据，一个一个样本能读取出来
    def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, train_mode, data_name):
        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0]  # 61-14 = 47, train_data
        self.test_days = divide_days[1]  # 7 * 2 = 14, test_data
        self.history_length = history_length  # 60/60 = 1
        self.time_interval = time_interval  # 1 hour
        self.data_name = data_name

        if data_name == 'Xian':
            self.one_day_length = int(24 / self.time_interval)  # 一整天的数据量
        elif data_name == 'PEMS04':
            self.one_day_length = int(24 * 60 / self.time_interval)  # 一整天的数据量

        self.graph = get_adjacent_matrix(distance_file=data_path[0], num_nodes=num_nodes, data_name=data_name)

        self.flow_norm, self.flow_data = self.pre_process_data(data=get_flow_data(data_path[1], data_name),
                                                               norm_dim=1)  # self.flow_norm为归一化的基

    def __len__(self):  # 表示数据集的长度
        """
        :return: length of dataset (number of samples).
        """
        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length  # 训练的样本数　＝　训练集总长度　－　历史数据长度
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length  # 每个样本都能测试，测试样本数　＝　测试总长度
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):  # 功能是如何取每一个样本 (x, y), index = [0, L1 - 1]这个是根据数据集的长度确定的
        """
        :param index: int, range between [0, length - 1].
        :return:
            graph: torch.tensor, [N, N].
            data_x: torch.tensor, [N, H, D].
            data_y: torch.tensor, [N, 1, D].
        """

        if self.train_mode == "train":
            index = index  # 训练集的数据是从时间０开始的，这个是每一个流量数据，要和样本（ｘ,y）区别
        elif self.train_mode == "test":
            index += self.train_days * self.one_day_length  # 有一个偏移量
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData.slice_data(self.flow_data, self.history_length, index, self.train_mode)  # 这个就是样本（ｘ,y）

        data_x = LoadData.to_tensor(data_x)  # [N, H, D] # 转换成张量
        data_y = LoadData.to_tensor(data_y).unsqueeze(1)  # [N, 1, D]　# 转换成张量，在时间维度上扩维

        return {"graph": LoadData.to_tensor(self.graph), "flow_x": data_x, "flow_y": data_y}  # 组成词典返回

    @staticmethod
    def slice_data(data, history_length, index, train_mode):  # 根据历史长度,下标来划分数据样本
        """
        :param data: np.array, normalized traffic data.
        :param history_length: int, length of history data to be used.
        :param index: int, index on temporal axis.
        :param train_mode: str, ["train", "test"].
        :return:
            data_x: np.array, [N, H, D].
            data_y: np.array [N, D].
        """
        if train_mode == "train":
            start_index = index  # 开始下标就是时间下标本身，这个是闭区间
            end_index = index + history_length  # 结束下标,这个是开区间
        elif train_mode == "test":
            start_index = index - history_length  # 开始下标，这个最后面贴图了，可以帮助理解
            end_index = index  # 结束下标
        else:
            raise ValueError("train model {} is not defined".format(train_mode))

        data_x = data[:, start_index: end_index]  # 在切第二维，不包括end_index
        data_y = data[:, end_index]  # 把上面的end_index取上

        return data_x, data_y

    @staticmethod
    def pre_process_data(data, norm_dim):  # 预处理,归一化
        """
        :param data: np.array,原始的交通流量数据
        :param norm_dim: int,归一化的维度，就是说在哪个维度上归一化,这里是在dim=1时间维度上
        :return:
            norm_base: list, [max_data, min_data], 这个是归一化的基.
            norm_data: np.array, normalized traffic data.
        """
        norm_base = LoadData.normalize_base(data, norm_dim)  # 计算 normalize base
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data)  # 归一化后的流量数据

        return norm_base, norm_data  # 返回基是为了恢复数据做准备的

    @staticmethod
    def normalize_base(data, norm_dim):  # 计算归一化的基
        """
        :param data: np.array, 原始的交通流量数据
        :param norm_dim: int, normalization dimension.归一化的维度，就是说在哪个维度上归一化,这里是在dim=1时间维度上
        :return:
            max_data: np.array
            min_data: np.array
        """
        max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D], keepdims=True就保持了纬度一致
        min_data = np.min(data, norm_dim, keepdims=True)

        return max_data, min_data  # 返回最大值和最小值

    @staticmethod
    def normalize_data(max_data, min_data, data):  # 计算归一化的流量数据，用的是最大值最小值归一化法
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, original traffic data without normalization.
        :return:
            np.array, normalized traffic data.
        """
        mid = min_data
        base = max_data - min_data
        normalized_data = (data - mid) / base

        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data):  # 恢复数据时使用的，为可视化比较做准备的
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, normalized data.
        :return:
            recovered_data: np.array, recovered data.
        """
        mid = min_data
        base = max_data - min_data

        recovered_data = data * base + mid

        return recovered_data  # 这个就是原始的数据

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)


if __name__ == '__main__':

    data_name = 'Xian'  # Xian、PEMS04

    if data_name == 'Xian':
        data_path = [f"adjacency_matrix_xian.csv", f"{data_name}.npz"]
        time_interval = 1
        history_length = 1
        num_nodes = 64
        divide_days = [47, 14]
    elif data_name == 'PEMS04':
        data_path = [f"../{data_name}/{data_name}.csv", f"../{data_name}/{data_name}.npz"]
        time_interval = 5
        history_length = 6
        num_nodes = 307
        divide_days = [45, 14]
    data = LoadData(data_path=data_path, num_nodes=num_nodes, divide_days=divide_days,
                    time_interval=time_interval, history_length=history_length,
                    train_mode="train", data_name=data_name)

    print(len(data))
    print(data[20])
    # print(data[0]["graph"])
    # print(data[0]["graph"].shape)
    # # print(test_data[0]["flow_y"])
    # print(data[0]["flow_x"].size())
    # print(data[0]["flow_y"].size())
