import torch
import torchvision
import numpy as np
import os
import random
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from vgg11 import VGG
from resnet18 import ResNet18,resnet18
from se_resnet18 import se_resnet_18
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', default=True, type=bool, choices=[True, False], help='True/False')
    parser.add_argument('--select_model', default='se_resnet18', type=str, choices=['resnet18', 'vgg11', 'se_resnet18'],
                        help='select model')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('--batchsize', default=32, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs(iteration)')
    args = parser.parse_args()


    # 数据集封装Dataset转DataLoader
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


    file_path = './train'
    mapping = {}
    all_files = []
    for label, directory in enumerate(os.listdir(file_path)):
        mapping[label] = directory
        tmp_list = [[file_path + '/' + directory + '/' + file, label] for file in
                    os.listdir(file_path + '/' + directory)]
        all_files.extend(tmp_list)
    print(all_files)
    random.shuffle(all_files)  # 将all_files的顺序打乱
    resize_target = 224  # 由于图片大小不同 指定为固定尺寸64
    batch_size = args.batchsize
    data_transform = transforms.Compose([
        transforms.transforms.Resize((resize_target, resize_target)),
        transforms.ToTensor()
    ])
    dataset = dataset(all_files, segment=False, transform=data_transform)
    split = int(np.floor(0.4 * len(dataset)))  # 设置val所占比例0.4
    indices = list(range(len(dataset)))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler_random = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler_random = torch.utils.data.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler_random)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler_random)

    # 模型加载
    vgg = VGG('VGG11')
    # print(vgg)
    # ResNet18 = ResNet18()
    ResNet18 = resnet18()
    # print(ResNet18)

    Se_resnet18 = se_resnet_18()
    # print(Se_resnet18)

    # 是否GPU
    if args.use_gpu == True:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 选择设备
    else:
        device = torch.device('cpu')
    print('device:', device)

    # 模型选择
    if args.select_model == 'resnet18':
        Model = torchvision.models.resnet18()
    if args.select_model == 'vgg11':
        Model = torchvision.models.vgg11()
    if args.select_model == 'se_resnet18':
        Model = Se_resnet18

    model = Model.to(device)
    criterion = torch.nn.CrossEntropyLoss()  # 损失函数
    print('criterion:', criterion)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)  # 优化器
    print('optimizer:', optimizer)

    # 训练部分
    final_train_loss = []
    final_train_acc = []
    final_valid_loss = []
    final_valid_acc = []
    best_acc = 0

    import time

    start = time.time()

    for epoch in range(1, args.num_epochs + 1):
        start1 = time.time()
        print('----------------------epoch = %s------------------------' % epoch)
        total_step = 0
        train_loss_list = []
        train_acc_list = []
        for ind, (img, cls) in enumerate(train_loader):
            model.train()
            x, y = img.to(device), cls.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            _, predicted = torch.max(y_pred.data, 1)
            acc = (predicted == y).sum() / len(y)
            train_loss_list.append(loss.item())
            train_acc_list.append(acc.item())
            total_step += 1
            if total_step % 10 == 0:
                print('**epoch=', epoch, '**train_loss=', loss.item(), '**acc=', acc.item(),
                      '**batch / num of batch  =  ',
                      total_step, '/', len(train_loader))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_valid_loss = 0
        valid_loss_list = []
        valid_acc_list = []
        with torch.no_grad():
            total_step1 = 0
            for ind, (img, cls) in enumerate(valid_loader):
                model.eval()
                x1, y1 = img.to(device), cls.to(device)
                y_pred1 = model(x1)
                loss = criterion(y_pred1, y1)  # 每次for循环得到一个batch的loss

                _, predicted = torch.max(y_pred1.data, 1)
                acc = (predicted == y1).sum() / len(y1)

                valid_loss_list.append(loss.item())
                valid_acc_list.append(acc.item())

                total_step1 += 1
                if total_step1 % 10 == 0:
                    print('**epoch=', epoch, '**valid_loss=', loss.item(), '**valid_acc=', acc.item(),
                          '**step / num of batch  =  ', total_step1, '/', len(valid_loader))
        train_loss = np.mean(train_loss_list)
        valid_loss = np.mean(valid_loss_list)
        train_acc = np.mean(train_acc_list)
        valid_acc = np.mean(valid_acc_list)

        if valid_acc >= best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), 'model_state_dict_epoch.pt')
            print('已保存第%s个epoch的模型' % epoch)
        print('epoch=', epoch, 'mean_train_loss=', train_loss)
        print('epoch=', epoch, 'mean_train_acc=', train_acc)
        print('epoch=', epoch, 'mean_valid_loss=', valid_loss)
        print('epoch=', epoch, 'mean_valid_acc=', valid_acc)
        final_train_loss.append(train_loss)
        final_train_acc.append(train_acc)
        final_valid_loss.append(valid_loss)
        final_valid_acc.append(valid_acc)
        # writer.add_scalar('train_loss', train_loss, epoch)  # 把loss值写入summary writer
        # writer.add_scalar('valid_loss', valid_loss, epoch)
        # writer1.add_scalar('train_acc', train_acc, epoch)
        # writer1.add_scalar('valid_acc', valid_acc, epoch)
        print('本epoch运行时长%s' % (time.time() - start1))
    end = time.time()
    print('运行时长%s' % (end - start))

    # 可视化
    plt.rc('font', family='Times New Roman')

    epochs = range(args.num_epochs)
    plt.plot(epochs, final_train_acc, 'r', label='Training Acc')
    plt.plot(epochs, final_valid_acc, 'b', linewidth=1, label='Validation Acc')
    ax = plt.subplot(111)
    # 设置刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # 设置坐标标签字体大小
    ax.set_xlabel('iter', fontsize=18)
    ax.set_ylabel('Acc', fontsize=18)
    ax.legend(fontsize=15)
    plt.savefig('accuracy迭代曲线', dpi=600)
