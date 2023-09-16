import os
import random
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import csv
import numpy as np
import torch

# from torch.utils.data import Dataset,DataLoader

mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


def log_string(log, string):
    """打印log"""
    log.write(string + '\n')
    log.flush()
    print(string)


def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    """Disable cudnn to maximize reproducibility 禁用cudnn以最大限度地提高再现性"""
    torch.cuda.cudnn_enabled = False
    """
    cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用
    如果设置为torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法
    然后再设置：torch.backends.cudnn.benchmark = True，当这个flag为True时，将会让程序在开始时花费一点额外时间，
    为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    但由于其是使用非确定性算法，这会让网络每次前馈结果略有差异,如果想要避免这种结果波动，可以将下面的flag设置为True
    """
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


""""
生成邻接矩阵
"""


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

def get_adjacency_matrix(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None):
    """
    :param distance_df_filename: str, csv边信息文件路径
    :param num_of_vertices:int, 节点数量
    :param type_:str, {connectivity, distance}
    :param id_filename:str 节点信息文件， 有的话需要构建字典
    """
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 建立映射列表
        df = pd.read_csv(distance_df_filename)
        for row in df.values:
            if len(row) != 3:
                continue
            i, j = int(row[0]), int(row[1])
            A[id_dict[i], id_dict[j]] = 1
            A[id_dict[j], id_dict[i]] = 1

        return A

    df = pd.read_csv(distance_df_filename)
    for row in df.values:
        if len(row) != 3:
            continue
        i, j, distance = int(row[0]), int(row[1]), float(row[2])
        if type_ == 'connectivity':
            A[i, j] = 1
            A[j, i] = 1
        elif type == 'distance':
            A[i, j] = 1 / distance
            A[j, i] = 1 / distance
        else:
            raise ValueError("type_ error, must be "
                             "connectivity or distance!")

    return A


class Evaluation(object):
    def __init__(self):
        pass

    @staticmethod
    def mae_(target, output):
        return np.mean(np.abs(target - output))

    @staticmethod
    def mape_(target, output):
        return np.mean(np.abs(target - output) / (target + 5)) * 100  # 加５是因为target有可能为0，当然只要不太大，加几都行

    @staticmethod
    def rmse_(target, output):
        return np.sqrt(np.mean(np.power(target - output, 2)))

    @staticmethod
    def total(target, output):
        mae = Evaluation.mae_(target, output)
        rmse = Evaluation.rmse_(target, output)
        mape = Evaluation.mape_(target, output)

        return mae, rmse, mape


def visualize_result(h5_file, nodes_id, time_se, visualize_file):
    file_obj = h5py.File(h5_file, "r")  # 获得文件对象，这个文件对象有两个keys："predict"和"target"
    prediction = file_obj["predict"][:][:, :, 0]  # [N, T],切片，最后一维取第0列，所以变成二维了，要是[:, :, :1]那么维度不会缩减
    target = file_obj["target"][:][:, :, 0]  # [N, T],同上
    file_obj.close()

    plot_prediction = prediction[nodes_id][time_se[0]: time_se[1]]  # [T1]，将指定节点的，指定时间的数据拿出来
    plot_target = target[nodes_id][time_se[0]: time_se[1]]  # [T1]，同上

    plt.figure()
    plt.title('预测结果')
    plt.grid(True, linestyle="-.", linewidth=0.5)
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_prediction, ls="-", marker=" ", color="r")
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_target, ls="-", marker=" ", color="b")

    plt.legend(["prediction", "target"], loc="best")

    plt.axis([0, time_se[1] - time_se[0],
              np.min(np.array([np.min(plot_prediction), np.min(plot_target)])),
              np.max(np.array([np.max(plot_prediction), np.max(plot_target)]))])
    # plt.title('GAT')
    plt.savefig(visualize_file + ".png")


"""指标"""


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)

    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()

    return mae, mape, rmse


"""
数据加载器
"""


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        数据加载器
        :param xs:训练数据
        :param ys:标签数据
        :param batch_size:batch大小
        :param pad_with_last_sample:剩余数据不够时，是否复制最后的sample以达到batch大小
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        """洗牌"""
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """标准转换器"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class NScaler:
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class MinMax01Scaler:
    """最大最小值01转换器"""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class MinMax11Scaler:
    """最大最小值11转换器"""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


def load_dataset(dataset_dir, normalizer, batch_size, valid_batch_size=None, test_batch_size=None, column_wise=False):
    """
    加载数据集
    :param dataset_dir: 数据集目录
    :param normalizer: 归一方式
    :param batch_size: batch大小
    :param valid_batch_size: 验证集batch大小
    :param test_batch_size: 测试集batch大小
    :param column_wise: 是指列元素的级别上进行归一，否则是全样本取值
    """
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    if normalizer == 'max01':
        if column_wise:
            minimum = data['x_train'].min(axis=0, keepdims=True)
            maximum = data['x_train'].max(axis=0, keepdims=True)
        else:
            minimum = data['x_train'].min()
            maximum = data['x_train'].max()

        scaler = MinMax01Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax01 Normalization')

    elif normalizer == 'max11':
        if column_wise:
            minimum = data['x_train'].min(axis=0, keepdims=True)
            maximum = data['x_train'].max(axis=0, keepdims=True)
        else:
            minimum = data['x_train'].min()
            maximum = data['x_train'].max()

        scaler = MinMax11Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax11 Normalization')

    elif normalizer == 'std':
        if column_wise:
            mean = data['x_train'].mean(axis=0, keepdims=True)  # 获得每列元素的均值、标准差
            std = data['x_train'].std(axis=0, keepdims=True)
        else:
            mean = data['x_train'].mean()
            std = data['x_train'].std()

        scaler = StandardScaler(mean, std)
        print('Normalize the dataset by Standard Normalization')

    elif normalizer == 'None':
        scaler = NScaler()
        print('Does not normalize the dataset')
    else:
        raise ValueError

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    graph = get_adjacent_matrix(distance_file='../data/PEMS04/PEMS04.csv', num_nodes=307, data_name='PEMS04')

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler

    # return {"graph": graph, "train_x": data["x_train"], "train_y": data["y_train"], "val_x": data["x_val"],
    #         "val_y": data["y_val"], "test_x": data["x_test"], "test_y": data["y_test"], "scaler": scaler}
    return data

if __name__ == '__main__':
    # A = get_adjacent_matrix(distance_file='../data/PEMS04/PEMS04.csv', num_nodes=307, data_name='PEMS04')
    A = get_adjacency_matrix(distance_df_filename='../data/PEMS04/PEMS04.csv', num_of_vertices=307)
    print(A.shape)
    print(A)

    # data = load_dataset('../data/processed/PEMS04/', 'std', batch_size=64, valid_batch_size=64,
    #                     test_batch_size=64)

    # train_x = DataLoader(data['train_x'], data['train_y'], batch_size=64)

    # print(data["graph"])
    # print(data['train_x'].shape)
    # print(data['train_y'].shape)
    # print(data['val_x'].shape)
    # print(data['val_y'].shape)
    # print(data['test_y'].shape)
    # print(data['test_y'].shape)

    # print(train_x)
