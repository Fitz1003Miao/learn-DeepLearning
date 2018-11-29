import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import os
import AlexNet
from PIL import Image

CLASS_LIST = ['cat', 'dog']

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.data_path = [os.path.join(folder, file) for file in os.listdir(folder)]
        self.trans = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    def __getitem__(self, index):
        img_path = self.data_path[index]
        label = CLASS_LIST.index(os.path.basename(img_path).split('.')[0])
        img = self.trans(Image.open(img_path))
        return img, label
    def __len__(self):
        return len(self.data_path)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        import math
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def main():
    train_dataset = MyDataSet('../data/train')
    val_dataset = MyDataSet('../data/val')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = 0)

    model = AlexNet.AlexNet(num_classes = len(CLASS_LIST))
    model.apply(weight_init)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        print('USE GPU')
    else:
        print('USE CPU')

    criterion = nn.CrossEntropyLoss(size_average = True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    for epoch in range(1, 100):
        model.train()
        for batch_idx, (data, target) in enumerate(train_dataloader):
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            pred = torch.max(F.softmax(output), 1)[1]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccu: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.data[0], (target == pred).sum().item() / target.size(0)))

            # if batch_idx % 10 == 0 and batch_idx != 0:
            #     val_total_loss = 0
            #     val_pred_correct = 0
            #     for batch_val_idx, (val_data, val_target) in enumerate(val_dataloader):
            #         if use_gpu:
            #             val_data, val_target = val_data.cuda(), val_target.cuda()
            #         val_output = model(val_data)
            #         val_pred = torch.max(F.softmax(val_output), 1)[1]
            #         val_loss = criterion(val_output, val_target)
            #         val_total_loss += val_loss
            #         val_pred_correct += (val_target == val_pred).sum().item()
            #     print('Val: Loss: {:.6f}, Acc: {:.6f}'.format(val_total_loss / len(val_dataloader), val_pred_correct / val_dataset.__len__()))

if __name__ == "__main__":
    main()