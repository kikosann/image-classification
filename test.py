import torch
import argparse

import torchvision.models
from torchvision import transforms
import os
from PIL import Image

file_path = './test'
trained_model = 'model_state_dict_epoch(resnet18).pt'
mapping = {}
all_files = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 选择设备

class dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, segment=False, transform=False):
        self.file_list = file_list
        self.transform = transform
        self.segment = segment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item = self.file_list[idx][0]
        label = self.file_list[idx][1]
        img = Image.open(item).convert('RGB')
        img = adjust_colors(img)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.functional.resize(img, (resize_target, resize_target))
            img = transforms.functional.to_tensor(img)
        return img, label


def adjust_colors(img):
    img = transforms.functional.adjust_brightness(img, 2)
    img = transforms.functional.adjust_contrast(img, 1.1)
    img = transforms.functional.adjust_saturation(img, 1.1)
    return img

tmp_list = [[file_path + '/' + file, 0]for file in os.listdir(file_path)]
print(tmp_list)

# for idx, img in enumerate(os.listdir(file_path)):
#     # print(img)
#     mapping[idx] = img
#     tmp_list = [[file_path + '/' + file, 0] for file in os.listdir(file_path)]
#     all_files.extend(tmp_list)
# print(all_files)
# print(mapping)
resize_target = 224  # 由于图片大小不同 指定为固定尺寸64
batch_size = 32
data_transform = transforms.Compose([
    transforms.transforms.Resize((resize_target, resize_target)),
    transforms.ToTensor()
])
dataset = dataset(tmp_list, segment=False, transform=data_transform)
#
# test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
#
# # for idx, data in enumerate(test_loader):
# #     print(data[0].shape)
# #     print(data[1].shape)
# #     if idx == 2:
# #         break
#
# # 加载模型
# model = torchvision.models.resnet18()
# model.fc = torch.nn.Linear(512,12)
# model.load_state_dict(torch.load(trained_model))
#
# model = model.to(device)
#  # 损失函数
#
#
# # test
# with torch.no_grad():
#     for ind, (img, cls) in enumerate(test_loader):
#         model.eval()
#         x1, y1 = img.to(device), cls.to(device)
#         y_pred1 = model(x1)
#         _, predicted = torch.max(y_pred1.data, 1)
#         print('ind=',ind)
#         print(predicted)
