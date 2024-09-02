import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import os
import csv
import json
import argparse
from config import *

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device=DEVICE):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(self.device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        out = nn.functional.softmax(out, dim=1)
        return out


class ExerciseData(Dataset):
    def __init__(self, path):
        self.data_path = path

        self.data_list = []
        self.category_2_idx = {}
        self.idx_2_category = {}

        # ignore .DS_Store file on Mac OS
        dir_list = [f for f in os.listdir(path) if not f.startswith('.')]

        for idx, cls in enumerate(dir_list):
            self.category_2_idx[cls] = idx
            self.idx_2_category[idx] = cls
            for csv_files in os.listdir(os.path.join(path, cls)):
                if not csv_files.endswith('.csv'):
                    continue
                with open(os.path.join(path, cls, csv_files), "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        row = [eval(element) for element in row]
                        self.data_list.append({'poses': row, 'label': idx})
        print('Successfully collected data')

    def __getitem__(self, item):
        input_data = self.data_list[item]['poses']
        input_data = torch.tensor(input_data)
        input_data = input_data.reshape(5, INPUT_DIM)  # 5 -> get_data_from_video.py: collect_data(data_len=5)
        x_mean, x_std = torch.mean(input_data), torch.std(input_data)
        input_data = (input_data - x_mean) / x_std  # max-min scaler

        label = torch.zeros((1, len(self.category_2_idx)))
        label[0, self.data_list[item]['label']] = 1

        return input_data, label

    def __len__(self):
        return len(self.data_list)


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--device', default='cuda:0', type=str, help='Training device')
#     parser.add_argument('--data_path', default=r'./data_without_resize', type=str, help='Path to input data')
#     parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
#     parser.add_argument('--epoch', default=150, type=int, help='Training epoch')
#     parser.add_argument('--save_dir', default='./checkpoint/without_resize', type=str, help='Path to save checkpoint')
#     args = parser.parse_args()
#     return args


def train(save_path, csv_path, pt_file='best_model.pt', batch_size=4, epoch=150, device=DEVICE):
    device = torch.device(device)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataset = ExerciseData(csv_path)
    with open(os.path.join(save_path, 'idx_2_category.json'), 'w') as file:
        file.write(json.dumps(dataset.idx_2_category))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    model = LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(epoch):
        total_loss = 0
        best_model_loss = 99999
        for seq_data, labels in dataloader:
            seq_data = seq_data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predict = model(seq_data)
            loss = loss_function(predict, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss = total_loss / len(dataloader)
        print(f'epoch: {i:3} loss: {total_loss:10.8f}')
        if best_model_loss > total_loss:
            best_model_loss = total_loss
            # save_path = os.path.join(save_path, pt_file)
            # print(save_path)
            torch.save(model.state_dict(), './checkpoint/best5.pt')


if __name__ == '__main__':
    train(save_path='./checkpoint',  # 保存pt文件的路径
          csv_path='./data',  # csv文件所在路径
          # pt_file='best_model3.pt',  # 保存pt文件名; 在<save_path>路径下
          batch_size=4,
          epoch=10)

