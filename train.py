# import torch
# import time
# from torch.utils.tensorboard import SummaryWriter
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# class ModelTrainer:
#     def __init__(self, model, data_name, model_name, num_epochs, criterion, optimizer, data):
#         self.model = model
#         self.data_name = data_name
#         self.model_name = model_name
#         self.num_epochs = num_epochs
#         self.criterion = criterion
#         self.optimizer = optimizer
#         # self.data = data
#         self.data = data
#         print(len(data))
#
#     def train(self, train_data):
#
#         # 设置日志文件的路径和文件名
#         log_dir = f'logs/{self.data_name}/{self.model_name}'  # 指定日志文件夹的路径和名称
#         # 创建TensorBoard的SummaryWriter
#         writer = SummaryWriter(log_dir=log_dir)
#
#         # 训练模型
#         self.model.train()
#
#         best_val_loss = float('inf')
#         current_patience = 0
#
#         for epoch in range(self.num_epochs):
#             start_time = time.time()  # 记录当前时间
#             epoch_loss = 0.0
#             count = 0
#             for data in train_data:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]],一次把一个batch的训练数据取出来
#                 self.model.zero_grad()
#                 count += 1
#                 outputs = self.model(data)
#                 # train_y = data['flow_y']  # [64, 307, 6, 1]
#                 train_y = data['train_y']  # [64, 307, 6, 1]
#                 print(train_y.size())
#                 # loss = criterion(outputs, data['flow_y'].to(device))
#                 loss = self.criterion(outputs, train_y.to(device))
#                 epoch_loss += loss.item()  # 这里是把一个epoch的损失都加起来，最后再除训练数据长度，用平均loss来表示
#
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#             # 记录损失值到TensorBoard
#             writer.add_scalar('Loss', loss.item(), epoch + 1)
#
#             end_time = time.time()  # 记录结束时间
#             epoch_time = end_time - start_time  # 计算epoch所花费的时间
#
#             # # Validation loss calculation
#             # val_loss = self.validate(val_data)
#             # writer.add_scalar('Validation Loss', val_loss, epoch + 1)
#             #
#             # if val_loss < best_val_loss:
#             #     best_val_loss = val_loss
#             #     current_patience = 0
#             # else:
#             #     current_patience += 1
#             #
#             # if current_patience >= patience:
#             #     print("早停！{}个纪元没有改进。".format(patience))
#             #     break
#
#             # print(epoch_loss)
#             if (epoch + 1) % 1 == 0:
#                 print(
#                     f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss / len(data):02.4f}, Time: {epoch_time:.2f} s')
#
#         # 保存训练好的模型
#         save_path = f'saved_models/{self.data_name}/{self.model_name}.pth'
#         torch.save(self.model.state_dict(), save_path)
#         # 关闭TensorBoard的SummaryWriter
#         writer.close()
#
#     def validate(self, val_data):
#         self.model.eval()
#         val_loss = 0.0
#
#         with torch.no_grad():
#             for data in val_data:
#                 outputs = self.model(data)
#                 val_y = data['flow_y'].to(device)
#                 loss = self.criterion(outputs, val_y)
#                 val_loss += loss.item()
#
#         return val_loss / len(val_data)
